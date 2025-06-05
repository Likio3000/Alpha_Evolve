# AlphaEvolve – Detailed Implementation & Reproduction Guide

*Based on **“AlphaEvolve: A Learning Framework to Discover Novel Alphas in Quantitative Investment”***

---

## 1  Overview

AlphaEvolve is an AutoML‑style evolutionary framework that **evolves stock‑prediction models (“alphas”)** combining the strengths of both classical formulaic alphas and data‑driven machine‑learning alphas. The goal is to automatically generate *weakly‑correlated* alphas with **high Sharpe ratio** while keeping model complexity low.

Key ingredients:

1. **Rich search space** – Scalars, vectors and matrices + domain‑specific operators.
2. **Evolutionary search** – Population‑based mutation, IC‑based selection.
3. **Selective domain‑knowledge injection** – RelationOps capture sector/industry relations **only when helpful**.
4. **Redundancy‑aware pruning** – Graph traversal removes dead code and fingerprints alphas to avoid re‑evaluation.

---

## 2  Data & Problem Setup

| Symbol | Meaning                                                   |
| ------ | --------------------------------------------------------- |
| *K*    | # stocks (tasks) (≈ 1026 after filtering)                 |
| *f*    | # feature types (13)                                      |
| *w*    | Look‑back window (days), = 13                             |
| *S*    | All samples (5‑year NASDAQ 2013‑2017 = 1220 trading days) |

### 2.1 Feature matrix  **X ∈ ℝ<sup>f×w</sup>**

1. Moving averages (MA‑5, 10, 20, 30).
2. Volatility of close prices over the same horizons.
3. Raw OHLCV: *open, high, low, close* (volume unavailable).

### 2.2 Label

Daily return
$\text{return}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$

### 2.3 Data split

| Split | Days |
| ----- | ---- |
| Train |  988 |
| Val   |  116 |
| Test  |  116 |

---

## 3  Alpha Representation

Every alpha is **a program with three functions**:

```text
Setup()   # one‑off initialisation per stock
Predict() # executed each day; must write scalar s1 as prediction
Update()  # runs only during training to update long‑term parameters
```

### 3.1 Operands

| Kind   | Symbol | Example                              |
| ------ | ------ | ------------------------------------ |
| Scalar | sᵢ     | daily return, intermediate constants |
| Vector | vᵢ     | OHLCV row, MA series                 |
| Matrix | mᵢ     | feature window X                     |

Special operands: **m0** (input X), **s0** (ground‑truth return), **s1** (prediction).

### 3.2 Operator families

| Family            | Notes                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| Basic math        | `+, −, ×, ÷, pow, log, exp, sin, cos, tan, abs, min, max, std` …       |
| **ExtractionOps** | *GetScalarOp*, *GetVectorOp* → pull row/col/value from *m0*            |
| **RelationOps**   | *RankOp*, *RelationRankOp*, *RelationDemeanOp* (sector/industry aware) |

---

## 4  Evolutionary Algorithm

### 4.1 Meta‑hyper‑parameters

| Parameter          | Value (paper)                         |
| ------------------ | ------------------------------------- |
| Population size    | **100**                               |
| Tournament size    | **10**                                |
| Mutation prob / op | **0.9**                               |
| Max operations     | Setup ≤ 21, Predict ≤ 21, Update ≤ 45 |
| Max operands       | 10 scalars, 16 vectors, 4 matrices    |

### 4.2 Mutation operators

1. **Point** – randomise operand or operator.
2. **Insert op** – inject fresh random operation.
3. **Delete op** – remove an existing operation.

Pseudocode

```python
for round in range(budget_rounds):
    parent = tournament_select(pop)
    child  = mutate(copy(parent))
    pop.append(child)
    if len(pop) > POP_SIZE:
        pop.pop(0)  # age‑based eviction
```

### 4.3 Fitness

Information Coefficient (IC)
$\text{IC} = \frac1N \sum_{t=1}^N \text{corr}(\hat{y}_t, y_t)$
where $\hat{y}_t$ is the vector of predictions for the *K* stocks on day *t*.

Selection uses *higher IC*; final model chosen by **Sharpe ratio** on Val.

---

## 5  Pruning & Fingerprinting

### 5.1 Why

Random mutations quickly create *dead code* → slows evaluation.

### 5.2 Algorithm

1. **Build DAG**: operands = nodes, ops = edges; root = last *s1*.
2. **Reverse DFS** from *s1* back to *m0*; mark reachable ops.
3. **Remove** unreachable ops → shorter program.
4. **Serialise remaining ops** (operator + operand IDs) → 128‑bit hash.
5. **Cache**\[hash] = fitness to skip re‑evaluation.

Complexity ≈ *O*(#ops) per alpha (negligible vs evaluation).

---

## 6  Selective Domain Knowledge

`RelationOps` query **peer stocks in same sector/industry** *only when chosen by evolution* – avoids hard‑coding noisy relations.

Example (paper Fig. 4):

```text
s3 = norm(m0)               # per‑stock feature
s2 = rank_sector(s3)        # relation‑aware ranking
s1 = heaviside(s2)          # use in prediction
```

---

## 7  End‑to‑End Training Loop

1. **One epoch** of training per evaluation (speed vs accuracy trade‑off).
2. For each stock: call *Setup*, then daily `Predict` (collect preds) & `Update` (train time only).
3. Compute IC on Val; hand back to EA.
4. After search converges: retrain best alpha longer, compute Sharpe on Test.

---

## 8  Baseline Re‑implementation

| Baseline     | Key points                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| Genetic Algo | Formula trees. Use DEAP or similar; crossover 0.4, point‑mut. 0.4, subtree‑mut. 0.01.                   |
| Rank‑LSTM    | 2‑layer LSTM → FC rank head; tune seq\_len ∈ {4,8,16,32}, units ∈ {32,64,128,256}, λ ∈ {0.01,0.1,1,10}. |
| RSR          | Graph + LSTM; same tuning; sector graph adjacency.                                                      |

Replicate 5 seeds, report mean ± sd.

---

## 9  Hyperparameters & Cut‑off Strategy

After each “round” (≈ 60 h wall‑clock) ⇒ keep α with highest Sharpe; **future rounds forbid IC‑correlation > 15 %** with any kept α.

Implementation tip: maintain rolling *K×R* IC matrix; cheap rank‑based approximation works.

---

## 10  Python Architecture Suggestions

### 10.1 Core classes

```python
class Operand:  # scalar / vector / matrix wrapper
    ...

class Operation:
    op_name: str
    inputs: list[int]
    output: int

class Alpha:
    setup:    list[Operation]
    predict:  list[Operation]
    update:   list[Operation]
```

### 10.2 Execution engine

* Numpy for vectorisation (matrix norms, heaviside).
* Per‑stock state dict holds operand arrays; reused across days.
* JIT (numba) or torch for extra speed if needed.

### 10.3 Search harness

* `multiprocessing.Pool` over alphas; share read‑only features.
* Global **LRU cache** maps fingerprint→fitness.
* Log every accepted alpha to SQLite/Parquet for auditability.

---

## 11  Pruning Implementation Snippet

```python
def prune(alpha: Alpha) -> Alpha:
    useful = set(['s1'])  # start from prediction
    for func in (alpha.predict, alpha.setup, alpha.update):
        for op in reversed(func):
            if op.output in useful:
                useful.update(op.inputs)
            else:
                op.mark_pruned()
    alpha.remove_pruned()
    return alpha
```

---

## 12  Performance Tips

* **Vectorise** operators – avoid Python loops inside daily simulation.
* **Couple train/val splits** into a single batched run to exploit cache locality.
* **Early stopping** in mutation: if any intermediate operand becomes *NaN/Inf*, assign worst fitness.

---

## 13  Reproducing Paper Numbers

1. Download NASDAQ OHLCV 2013‑01‑02 → 2017‑12‑29.
2. Apply *min‑price* and *sample‑count* filters ⇒ 1026 stocks.
3. Build 13‑feature tensor (*f*×*w*) per stock per day.
4. Run EA for 4 rounds + 1 competition round (≈ 5‑7 days on 64 CPUs).
5. Confirm best alpha Sharpe ≈ 21.3, IC ≈ 0.067 on Test.

---

## 14  Reference

Can Cui et al. “AlphaEvolve: A Learning Framework to Discover Novel Alphas in Quantitative Investment.” 2021. fileciteturn0file0
