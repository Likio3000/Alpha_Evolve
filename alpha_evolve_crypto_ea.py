import os
import glob
import math
import random
import sys
import textwrap

import numpy as np
import pandas as pd

"""
alphaevolve_crypto_ea.py   (v3 – diversity + nicer logging)
========================================================
•   Duplicate filter – keeps only unique expression strings.
•   Correlation‑penalty – discourages highly correlated winners (|ρ|>0.9).
•   Pretty console print: generation, fitness, IC, size, depth, expr (truncated).

Run:
    python alphaevolve_crypto_ea.py [generations] [seed]
"""

###############################################################################
# CONFIG ######################################################################
###############################################################################
DATA_DIR = "./data"
POP_SIZE = 128
N_GENERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TOURNAMENT_K = 5
P_MUT = 0.3
P_CROSS = 0.5
ELITE_KEEP = 8
MAX_DEPTH = 6
PAR_SIM_PENALTY = 0.20  # bigger ⇒ favour simpler trees
CORR_PENALTY_W = 0.10  # weight for correlation penalty (per |ρ|)
DUPLICATE_HOF_SZ = 50  # max stored alphas for corr‑penalty / duplication
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
random.seed(SEED)
np.random.seed(SEED)
###############################################################################

###############################################################################
# 1. LOAD OHLC CSVs ###########################################################
###############################################################################


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("time").reset_index(drop=True)
    df["ret_fwd"] = df["close"].pct_change().shift(-1)
    for w in (5, 10, 20, 30):
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"vol_{w}"] = df["close"].rolling(w).std(ddof=0)
    df["range"] = df["high"] - df["low"]
    df.dropna(inplace=True)
    return df


COINS = {
    os.path.basename(f).split(".")[0]: load_csv(f)
    for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
}
print("Loaded", len(COINS), "coins →", ", ".join(COINS.keys()))

###############################################################################
# 2. EXPRESSION‑TREE PRIMITIVES ###############################################
###############################################################################
UNARY = {"tanh": np.tanh, "sign": np.sign, "neg": np.negative, "abs": np.abs}
BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": lambda a, b: np.divide(a, np.where(b == 0, 1e-12, b)),
}
TERMINALS = (
    ["close", "open", "high", "low", "range"]
    + [f"ma_{w}" for w in (5, 10, 20, 30)]
    + [f"vol_{w}" for w in (5, 10, 20, 30)]
)


class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    # evaluation
    def eval(self, df):
        if self.label in UNARY:
            return UNARY[self.label](self.children[0].eval(df))
        if self.label in BINARY:
            return BINARY[self.label](
                self.children[0].eval(df), self.children[1].eval(df)
            )
        return df[self.label].values

    # helpers
    def copy(self):
        return Node(self.label, [c.copy() for c in self.children])

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        return 1 + (max(c.depth() for c in self.children) if self.children else 0)

    def __str__(self):
        if not self.children:
            return self.label
        if self.label in UNARY:
            return f"{self.label}({self.children[0]})"
        return f"({self.children[0]} {self.label} {self.children[1]})"


###############################################################################
# 3. GP OPERATORS #############################################################
###############################################################################


def random_tree(depth=0):
    if depth >= MAX_DEPTH or (depth > 1 and random.random() < 0.3):
        return Node(random.choice(TERMINALS))
    if random.random() < 0.5:
        return Node(random.choice(list(UNARY)), [random_tree(depth + 1)])
    return Node(
        random.choice(list(BINARY)), [random_tree(depth + 1), random_tree(depth + 1)]
    )


def mutate(node):
    if random.random() < 0.1:
        return random_tree()
    node.children = [mutate(c) for c in node.children]
    return node


def crossover(a, b):
    if random.random() < 0.5:
        return b.copy()
    if not a.children or not b.children:
        return a.copy()
    child = a.copy()
    idx = random.randrange(len(child.children))
    child.children[idx] = crossover(child.children[idx], random.choice(b.children))
    return child


###############################################################################
# 4. FITNESS – IC + penalties #################################################
###############################################################################
_cache = {}
HOF_expr = []  # unique expressions
HOF_sigs = []  # corresponding stacked signals (concatenated coins)


def _concat_signal(sig_dict):
    return np.concatenate([v for v in sig_dict.values()])


def evaluate(node: Node):
    exp_str = str(node)
    if exp_str in _cache:
        return _cache[exp_str]

    sig_by_coin = {}
    ic_vals = []
    for sym, df in COINS.items():
        try:
            s = node.eval(df)
            sig_by_coin[sym] = s[:-1]
            ic = np.corrcoef(s[:-1], df["ret_fwd"].values[:-1])[0, 1]
            ic_vals.append(0.0 if math.isnan(ic) else ic)
        except Exception:
            ic_vals.append(0.0)
            sig_by_coin[sym] = np.zeros(len(df) - 1)
    mean_ic = float(np.mean(ic_vals))

    # parsimony
    score = mean_ic - PAR_SIM_PENALTY / node.size()

    # correlation penalty vs Hall‑of‑Fame
    if HOF_sigs:
        cand = _concat_signal(sig_by_coin)
        corrs = [abs(np.corrcoef(cand, ref)[0, 1]) for ref in HOF_sigs]
        high_corr = [c for c in corrs if not math.isnan(c) and c > 0.9]
        if high_corr:
            score -= CORR_PENALTY_W * np.mean(high_corr)
    _cache[exp_str] = (score, mean_ic)
    return _cache[exp_str]


###############################################################################
# 5. EVOLVE LOOP ##############################################################
###############################################################################


def evolve():
    pop = [random_tree() for _ in range(POP_SIZE)]
    for gen in range(N_GENERATIONS):
        fits = [evaluate(ind)[0] for ind in pop]
        best_idx = int(np.argmax(fits))
        best = pop[best_idx]
        best_fit, best_ic = evaluate(best)
        print(
            f"Gen {gen:3d} | fit {best_fit:+.4f} | IC {best_ic:+.4f} | sz {best.size():3d} | dp {best.depth():2d} | "
            + textwrap.shorten(str(best), 120)
        )

        # update Hall‑of‑Fame unique list
        b_str = str(best)
        if b_str not in HOF_expr:
            HOF_expr.append(b_str)
            HOF_sigs.append(
                _concat_signal({sym: best.eval(df)[:-1] for sym, df in COINS.items()})
            )
            if len(HOF_expr) > DUPLICATE_HOF_SZ:
                HOF_expr.pop(0)
                HOF_sigs.pop(0)

        # elitism
        elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[
            :ELITE_KEEP
        ]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < POP_SIZE:
            a = max(random.sample(range(POP_SIZE), TOURNAMENT_K), key=lambda i: fits[i])
            b = max(random.sample(range(POP_SIZE), TOURNAMENT_K), key=lambda i: fits[i])
            child = (
                crossover(pop[a], pop[b])
                if random.random() < P_CROSS
                else pop[a].copy()
            )
            if random.random() < P_MUT:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
    # deduplicate final top‑20
    scored = [(ind, evaluate(ind)[0]) for ind in pop]
    scored.sort(key=lambda t: t[1], reverse=True)
    uniq = []
    top = []
    for n, sc in scored:
        if str(n) not in uniq:
            uniq.append(str(n))
            top.append((n, sc))
        if len(top) == 20:
            break
    return top


if __name__ == "__main__":
    top = evolve()
    print("\n================ UNIQUE TOP 20 ================")
    for i, (expr, fit) in enumerate(top, 1):
        ic = evaluate(expr)[1]
        print(
            f"#{i:02d} | fit {fit:+.4f} | IC {ic:+.4f} | sz {expr.size():3d} | dp {expr.depth():2d}\n   {expr}\n"
        )
