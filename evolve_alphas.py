from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple
import math
from multiprocessing import Pool, cpu_count
import numpy as np
import logging

from alpha_framework import AlphaProgram, TypeId, CROSS_SECTIONAL_FEATURE_VECTOR_NAMES
import alpha_framework.program_logic_generation as plg
import alpha_framework.program_logic_variation as plv
from evolution_components import (
    initialize_data,
    evaluate_program,
    initialize_evaluation_cache,
    initialize_hof,
    add_program_to_hof,
    update_correlation_hof,
    get_final_hof_programs,
    print_generation_summary,
    clear_hof,
    pbar,
)
from evolution_components import data_handling as dh_module
from evolution_components import hall_of_fame_manager as hof_module
from evolution_components import evaluation_logic as el_module
from utils.context import make_eval_context_from_globals, EvalContext
from evolution_components import diagnostics as diag

from config import EvoConfig  # New import
from utils.context import make_eval_context_from_dir

###############################################################################
# CLI & CONFIG REMOVED ########################################################
###############################################################################
# _parse_cli function REMOVED
# Global args object initialization REMOVED

_RNG = np.random.default_rng()
_CTX: EvalContext | None = None
_WORKER_CTX: EvalContext | None = None

def _pool_init(data_dir: str, strategy: str, min_common_points: int, eval_lag: int, sector_mapping: dict):
    """Initializer for worker processes to build an EvalContext once per worker."""
    from utils.context import make_eval_context_from_dir as _mk
    from evolution_components import data_handling as _dh
    global _WORKER_CTX
    try:
        # Precompute column matrices in each worker for vector features
        # Avoid heavy precompute in workers to reduce init overhead and memory.
        cols = None
        _WORKER_CTX = _mk(
            data_dir=data_dir,
            strategy=strategy,
            min_common_points=min_common_points,
            eval_lag=eval_lag,
            dh_module=_dh,
            sector_mapping=sector_mapping,
            precompute_columns=cols,
        )
    except Exception:
        _WORKER_CTX = None


def _sync_evolution_configs_from_config(cfg: EvoConfig):  # Renamed and signature changed
    global _RNG
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _RNG = np.random.default_rng(cfg.seed)
    # No longer initialize global data; EvalContext will be used instead
    el_module.configure_evaluation(
        parsimony_penalty=cfg.parsimony_penalty,
        max_ops=cfg.max_ops,
        xs_flatness_guard=cfg.xs_flat_guard,
        temporal_flatness_guard=cfg.t_flat_guard,
        early_abort_bars=cfg.early_abort_bars,
        early_abort_xs=cfg.early_abort_xs,
        early_abort_t=cfg.early_abort_t,
        flat_bar_threshold=cfg.flat_bar_threshold,
        scale_method=cfg.scale,
        sharpe_proxy_weight=cfg.sharpe_proxy_w,
        ic_std_penalty_weight=cfg.ic_std_penalty_w,
        turnover_penalty_weight=cfg.turnover_penalty_w,
        ic_tstat_weight=cfg.ic_tstat_w,
        use_train_val_splits=cfg.use_train_val_splits,
        train_points=cfg.train_points,
        val_points=cfg.val_points,
        sector_neutralize=cfg.sector_neutralize,
        winsor_p=cfg.winsor_p,
        parsimony_jitter_pct=cfg.parsimony_jitter_pct,
        # Provide fixed weights for logging/secondary fitness (no ramp)
        fixed_sharpe_proxy_weight=cfg.sharpe_proxy_w,
        fixed_ic_std_penalty_weight=cfg.ic_std_penalty_w,
        fixed_turnover_penalty_weight=cfg.turnover_penalty_w,
        fixed_corr_penalty_weight=cfg.corr_penalty_w,
        fixed_ic_tstat_weight=cfg.ic_tstat_w,
        hof_corr_mode=getattr(cfg, "hof_corr_mode", "flat"),
        temporal_decay_half_life=getattr(cfg, "temporal_decay_half_life", 0.0),
        cv_k_folds=getattr(cfg, "cv_k_folds", 0),
        cv_embargo=getattr(cfg, "cv_embargo", 0),
    )
    initialize_hof(
        max_size=cfg.hof_size,
        keep_dupes=cfg.keep_dupes_in_hof,
        corr_penalty_weight=cfg.corr_penalty_w,
        corr_cutoff=cfg.corr_cutoff
    )
    initialize_evaluation_cache(cfg.eval_cache_size)


FEATURE_VARS: Dict[str, TypeId] = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
# Constant scalar features are deliberately excluded to align with the
# reference paper.  They remain accessible via SCALAR_FEATURE_NAMES if needed.

INITIAL_STATE_VARS: Dict[str, TypeId] = {
    "prev_s1_vec": "vector",
    "rolling_mean_custom": "vector"
}

# ─── bias random generation and mutation towards vector-returning ops ───
VECTOR_OPS_BIAS = 0.3
plg.VECTOR_OPS_BIAS = VECTOR_OPS_BIAS
plv.VECTOR_OPS_BIAS = VECTOR_OPS_BIAS

def _random_prog(cfg: EvoConfig) -> AlphaProgram:
    return AlphaProgram.random_program(
        FEATURE_VARS,
        INITIAL_STATE_VARS,
        max_total_ops=cfg.max_ops,
        max_setup_ops=cfg.max_setup_ops,
        max_predict_ops=cfg.max_predict_ops,
        max_update_ops=cfg.max_update_ops,
        ops_split_jitter=getattr(cfg, "ops_split_jitter", 0.0),
        rng=_RNG,
    )


def _mutate_prog(p: AlphaProgram, cfg: EvoConfig) -> AlphaProgram:
    return p.mutate(
        FEATURE_VARS,
        INITIAL_STATE_VARS,
        max_total_ops=cfg.max_ops,
        max_setup_ops=cfg.max_setup_ops,
        max_predict_ops=cfg.max_predict_ops,
        max_update_ops=cfg.max_update_ops,
        rng=_RNG,
    )

def _eval_worker(args) -> Tuple[int, el_module.EvalResult]:
    idx, prog = args
    try:
        # Prefer per-worker context if present, otherwise fall back to parent context
        ctx = _WORKER_CTX if _WORKER_CTX is not None else _CTX
        result = evaluate_program(
            prog,
            dh_module,
            hof_module,
            INITIAL_STATE_VARS,
            return_preds=True,
            ctx=ctx,
        )
        return idx, result
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Evaluation error for program idx=%s fp=%s: %s",
            idx,
            getattr(prog, "fingerprint", "<unknown>"),
            e,
        )
        # Return a sentinel EvalResult with -inf fitness so it is ignored
        return idx, el_module.EvalResult(
            fitness=float('-inf'),
            mean_ic=0.0,
            sharpe_proxy=0.0,
            parsimony_penalty=0.0,
            correlation_penalty=0.0,
            processed_predictions=None,
        )

###############################################################################
# EVOLVE LOOP ##############################################################
###############################################################################

def evolve_with_context(cfg: EvoConfig, ctx: EvalContext) -> List[Tuple[AlphaProgram, float]]:
    """Evolve using an explicit EvalContext (context-first API).

    This avoids any implicit module-level data. The traditional evolve() helper
    will build a context from cfg and delegate to this function.
    """
    _sync_evolution_configs_from_config(cfg)
    try:
        diag.reset()
    except Exception:
        pass
    # Store the provided context for local workers and sequential eval
    global _CTX
    _CTX = ctx

    logger = logging.getLogger(__name__)

    pop: List[AlphaProgram] = [_random_prog(cfg) for _ in range(cfg.pop_size)]
    # Optional multi-fidelity context for a cheaper first-pass evaluation
    mf_ctx: EvalContext | None = None
    if getattr(cfg, "mf_enabled", False):
        try:
            mf_frac = float(getattr(cfg, "mf_initial_fraction", 0.4))
            mf_ctx = slice_eval_context(ctx, eval_fraction=mf_frac)
        except Exception:
            mf_ctx = None
    gen_eval_times_history: List[float] = []

    try:
        prev_best_fit: float = float('-inf')
        no_improve_gens: int = 0
        prev_q25: float | None = None
        q25_deteriorate_streak: int = 0
        for gen in range(cfg.generations):
            # Anneal correlation penalty and optional eval weights to encourage exploration early
            # Ramp linearly over a configurable portion of the run
            try:
                ramp_gens_cfg = int(max(1, round(cfg.generations * float(getattr(cfg, "ramp_fraction", 1.0/3.0)))))
            except Exception:
                ramp_gens_cfg = cfg.generations // 3 if cfg.generations > 0 else 5
            ramp_gens = max(getattr(cfg, "ramp_min_gens", 5), ramp_gens_cfg)
            ramp = min(1.0, (gen + 1) / ramp_gens)
            try:
                hof_module.set_correlation_penalty(weight=cfg.corr_penalty_w * ramp, cutoff=cfg.corr_cutoff)
            except Exception:
                pass
            try:
                # Re-apply eval configuration to adjust weights dynamically
                el_module.configure_evaluation(
                    parsimony_penalty=cfg.parsimony_penalty,
                    max_ops=cfg.max_ops,
                    xs_flatness_guard=cfg.xs_flat_guard,
                    temporal_flatness_guard=cfg.t_flat_guard,
                    early_abort_bars=cfg.early_abort_bars,
                    early_abort_xs=cfg.early_abort_xs,
                    early_abort_t=cfg.early_abort_t,
                    flat_bar_threshold=cfg.flat_bar_threshold,
                    scale_method=cfg.scale,
                    sharpe_proxy_weight=cfg.sharpe_proxy_w * ramp,
                    ic_std_penalty_weight=cfg.ic_std_penalty_w * ramp,
                    turnover_penalty_weight=cfg.turnover_penalty_w * ramp,
                    ic_tstat_weight=cfg.ic_tstat_w * ramp,
                    use_train_val_splits=cfg.use_train_val_splits,
                    train_points=cfg.train_points,
                    val_points=cfg.val_points,
                    sector_neutralize=cfg.sector_neutralize,
                    winsor_p=cfg.winsor_p,
                    hof_corr_mode=getattr(cfg, "hof_corr_mode", "flat"),
                    temporal_decay_half_life=getattr(cfg, "temporal_decay_half_life", 0.0),
                    parsimony_jitter_pct=cfg.parsimony_jitter_pct,
                    # Fixed weights stay at targets
                    fixed_sharpe_proxy_weight=cfg.sharpe_proxy_w,
                    fixed_ic_std_penalty_weight=cfg.ic_std_penalty_w,
                    fixed_turnover_penalty_weight=cfg.turnover_penalty_w,
                    fixed_corr_penalty_weight=cfg.corr_penalty_w,
                    fixed_ic_tstat_weight=cfg.ic_tstat_w,
                )
            except Exception:
                pass
            # Reset per-generation diagnostics for clearer inspection
            if hasattr(el_module, "reset_eval_stats"):
                el_module.reset_eval_stats()
            if hasattr(el_module, "reset_eval_events"):
                el_module.reset_eval_events()
            logger.info(
                "Gen %s/%s | Starting evaluation of %s programs",
                gen + 1,
                cfg.generations,
                len(pop),
            )
            t_start_gen = time.perf_counter()
            eval_results: List[Tuple[int, el_module.EvalResult]] = []
            pop_fitness_scores = np.full(cfg.pop_size, -np.inf)

            # Helper to choose selection score based on configuration
            def _sel_score(res: el_module.EvalResult) -> float:
                sel = getattr(cfg, "selection_metric", "ramped")
                base_score: float
                # Auto strategy: ramped early, fixed after ramp completes
                if sel == "auto":
                    use_fixed = (ramp >= 0.999)
                    if use_fixed:
                        fs = getattr(res, "fitness_static", None)
                        base_score = float(fs) if fs is not None and np.isfinite(fs) else float(res.fitness)
                    else:
                        base_score = float(res.fitness)
                elif sel == "fixed":
                    fs = getattr(res, "fitness_static", None)
                    base_score = float(fs) if fs is not None and np.isfinite(fs) else float(res.fitness)
                elif sel == "ic":
                    base_score = float(res.mean_ic)
                elif sel == "phased":
                    # Early phase: pure IC, mid: ramped, late: fixed
                    if (gen < int(getattr(cfg, "ic_phase_gens", 0))):
                        base_score = float(res.mean_ic)
                    elif ramp < 0.999:
                        base_score = float(res.fitness)
                    else:
                        fs = getattr(res, "fitness_static", None)
                        base_score = float(fs) if fs is not None and np.isfinite(fs) else float(res.fitness)
                else:
                    # default ramped fitness
                    base_score = float(res.fitness)

                # Optional novelty boost: reward low correlation w.r.t. HOF
                nb_w = float(getattr(cfg, "novelty_boost_w", 0.0))
                if nb_w > 0.0:
                    try:
                        proc = getattr(res, "processed_predictions", None)
                        if proc is not None and proc.size > 0:
                            flat = proc.ravel()
                            mean_corr = hof_module.get_mean_corr_component_with_hof(flat)
                            # Boost by (1 - corr); if HOF empty, mean_corr=0 → neutral boost
                            base_score = base_score + nb_w * float(1.0 - mean_corr)
                    except Exception:
                        pass
                # Optional structural novelty boost: reward opcode diversity vs HOF
                ns_w = float(getattr(cfg, "novelty_struct_w", 0.0))
                if ns_w > 0.0:
                    try:
                        # Candidate opcode set
                        ops = [*getattr(pop[i], 'setup', []), *getattr(pop[i], 'predict_ops', []), *getattr(pop[i], 'update_ops', [])]
                        cand_set = {getattr(o, 'opcode', '') for o in ops if hasattr(o, 'opcode')}
                        hof_sets = hof_module.get_hof_opcode_sets(None)
                        if hof_sets:
                            # Jaccard distance vs closest HOF program
                            dists = []
                            for s in hof_sets:
                                if not s and not cand_set:
                                    dists.append(0.0)
                                    continue
                                inter = len(cand_set.intersection(s))
                                union = max(1, len(cand_set.union(s)))
                                jaccard = inter / union
                                dists.append(1.0 - jaccard)
                            novelty_struct = max(0.0, float(max(dists)))
                            base_score = base_score + ns_w * novelty_struct
                    except Exception:
                        pass
                return base_score

            # Multiprocessing can fail in restricted environments; fall back to sequential when workers == 1.
            # If multi-fidelity is enabled and we're not using a pool, perform a cheap pass then promote top-K for full eval.
            if mf_ctx is not None and (cfg.workers or 0) <= 1:
                # Sequential multi-fidelity evaluation
                iterator = pbar(range(len(pop)), desc=f"Gen {gen+1}/{cfg.generations} [mf-cheap]", disable=cfg.quiet, total=cfg.pop_size)
                _CTX = mf_ctx  # type: ignore
                tmp_results: List[Tuple[int, el_module.EvalResult]] = []
                for i in iterator:
                    _, result = _eval_worker((i, pop[i]))
                    tmp_results.append((i, result))
                try:
                    promote_frac = float(getattr(cfg, "mf_promote_fraction", 0.3))
                    promote_n = max(int(round(cfg.pop_size * promote_frac)), int(getattr(cfg, "mf_min_promote", 8)))
                except Exception:
                    promote_n = max(8, cfg.pop_size // 3)
                def _sel_score_local(res: el_module.EvalResult) -> float:
                    sel = getattr(cfg, "selection_metric", "ramped")
                    if sel == "ic":
                        return float(res.mean_ic)
                    if sel == "fixed":
                        fs = getattr(res, "fitness_static", None)
                        return float(fs) if fs is not None and np.isfinite(fs) else float(res.fitness)
                    if sel == "phased" and (gen < int(getattr(cfg, "ic_phase_gens", 0))):
                        return float(res.mean_ic)
                    return float(res.fitness)
                tmp_results.sort(key=lambda t: _sel_score_local(t[1]), reverse=True)
                promote_idx = {i for (i, _) in tmp_results[:promote_n]}
                _CTX = ctx  # type: ignore
                iterator2 = pbar(range(len(pop)), desc=f"Gen {gen+1}/{cfg.generations} [mf-full]", disable=cfg.quiet, total=cfg.pop_size)
                for i in iterator2:
                    if i in promote_idx:
                        _, result = _eval_worker((i, pop[i]))
                    else:
                        result = next((r for (j, r) in tmp_results if j == i), None)
                        if result is None:
                            _, result = _eval_worker((i, pop[i]))
                    eval_results.append((i, result))
                    pop_fitness_scores[i] = _sel_score(result)
            elif (cfg.workers or 0) > 1:
                with Pool(
                    processes=cfg.workers or cpu_count(),
                    initializer=_pool_init,
                    initargs=(
                        cfg.data_dir,
                        cfg.max_lookback_data_option,
                        cfg.min_common_points,
                        cfg.eval_lag,
                        cfg.sector_mapping,
                    ),
                ) as pool:
                    results_iter = pool.imap_unordered(_eval_worker, enumerate(pop))
                    bar = pbar(results_iter, desc=f"Gen {gen+1}/{cfg.generations}", disable=cfg.quiet, total=cfg.pop_size)
                    completed = 0
                    for i, result in bar:
                        eval_results.append((i, result))
                        pop_fitness_scores[i] = _sel_score(result)
                        completed += 1
                        logger.debug(
                            "g%d p%03d fit=%+.4f IC=%+.4f ops=%d",
                            gen + 1,
                            i,
                            result.fitness,
                            result.mean_ic,
                            pop[i].size,
                        )
                        if not cfg.quiet and hasattr(bar, 'set_postfix_str'):
                            valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                            best_score_so_far = np.max(valid_scores) if valid_scores.size > 0 else -np.inf
                            bar.set_postfix_str(f"BestFit: {best_score_so_far:.4f}")
                        # Emit live progress JSON for dashboards
                        try:
                            import json as _json
                            valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                            best_score_so_far = float(np.max(valid_scores)) if valid_scores.size > 0 else float('-inf')
                            median_so_far = float(np.median(valid_scores)) if valid_scores.size > 0 else float('nan')
                            elapsed = float(time.perf_counter() - t_start_gen)
                            frac = completed / max(1, cfg.pop_size)
                            eta = (elapsed / frac - elapsed) if frac > 1e-6 else None
                            logger.info(
                                "PROGRESS %s",
                                _json.dumps({
                                    "type": "gen_progress",
                                    "gen": int(gen + 1),
                                    "completed": int(completed),
                                    "total": int(cfg.pop_size),
                                    "best": best_score_so_far,
                                    "median": median_so_far,
                                    "elapsed_sec": elapsed,
                                    "eta_sec": float(eta) if eta is not None else None,
                                }),
                            )
                        except Exception:
                            pass
            else:
                # Sequential evaluation
                iterator = pbar(range(len(pop)), desc=f"Gen {gen+1}/{cfg.generations}", disable=cfg.quiet, total=cfg.pop_size)
                for idx_in_seq, i in enumerate(iterator, start=1):
                    _, result = _eval_worker((i, pop[i]))
                    eval_results.append((i, result))
                    pop_fitness_scores[i] = _sel_score(result)
                    logger.debug(
                        "g%d p%03d fit=%+.4f IC=%+.4f ops=%d",
                        gen + 1,
                        i,
                        result.fitness,
                        result.mean_ic,
                        pop[i].size,
                    )
                    if not cfg.quiet and hasattr(iterator, 'set_postfix_str'):
                        valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                        best_score_so_far = np.max(valid_scores) if valid_scores.size > 0 else -np.inf
                        iterator.set_postfix_str(f"BestFit: {best_score_so_far:.4f}")
                    # Emit live progress JSON for dashboards
                    try:
                        import json as _json
                        valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                        best_score_so_far = float(np.max(valid_scores)) if valid_scores.size > 0 else float('-inf')
                        median_so_far = float(np.median(valid_scores)) if valid_scores.size > 0 else float('nan')
                        elapsed = float(time.perf_counter() - t_start_gen)
                        frac = idx_in_seq / max(1, cfg.pop_size)
                        eta = (elapsed / frac - elapsed) if frac > 1e-6 else None
                        logger.info(
                            "PROGRESS %s",
                            _json.dumps({
                                "type": "gen_progress",
                                "gen": int(gen + 1),
                                "completed": int(idx_in_seq),
                                "total": int(cfg.pop_size),
                                "best": best_score_so_far,
                                "median": median_so_far,
                                "elapsed_sec": elapsed,
                                "eta_sec": float(eta) if eta is not None else None,
                            }),
                        )
                    except Exception:
                        pass
            
            gen_eval_time = time.perf_counter() - t_start_gen
            if gen_eval_time > 0:
                gen_eval_times_history.append(gen_eval_time)

            logger.info(
                "Gen %s | Evaluation completed in %.1fs",
                gen + 1,
                gen_eval_time,
            )

            # Emit evaluation diagnostics
            if hasattr(el_module, "get_eval_stats"):
                stats = el_module.get_eval_stats()
                # Emit a concise INFO summary for quick loop feedback
                logger.info(
                    "Eval stats g%d | cache hits %d, misses %d, no_feat %d, nan_inf %d, all_zero %d, early_xs %d, early_t %d, early_flatbar %d",
                    gen + 1,
                    stats.get("cache_hits", 0),
                    stats.get("cache_misses", 0),
                    stats.get("rejected_no_feature_vec", 0),
                    stats.get("rejected_nan_or_inf", 0),
                    stats.get("rejected_all_zero", 0),
                    stats.get("early_abort_xs", 0),
                    stats.get("early_abort_t", 0),
                    stats.get("early_abort_flatbar", 0),
                )
                # Record richer diagnostics for optional post-run analysis
                try:
                    # Per-gen population distribution and a snapshot of top-K
                    K = 5
                    events = el_module.get_eval_events() if hasattr(el_module, "get_eval_events") else []
                    # Sort a copy for diagnostics to reflect current selection metric
                    tmp_sorted = sorted(eval_results, key=lambda x: _sel_score(x[1]), reverse=True)
                    valid_scores = [r[1].fitness for r in tmp_sorted if np.isfinite(r[1].fitness)]
                    q = None
                    if valid_scores:
                        arr = np.array(valid_scores)
                        q = {
                            "best": float(np.max(arr)),
                            "p95": float(np.quantile(arr, 0.95)),
                            "p75": float(np.quantile(arr, 0.75)),
                            "median": float(np.median(arr)),
                            "p25": float(np.quantile(arr, 0.25)),
                            "count": int(arr.size),
                        }
                    top_summary = []
                    for idx_in_pop, res in tmp_sorted[:K]:
                        prog = pop[idx_in_pop]
                        top_summary.append({
                            "fingerprint": prog.fingerprint,
                            "fitness": float(res.fitness),
                            "fitness_fixed": float(getattr(res, "fitness_static", float("nan"))),
                            "mean_ic": float(res.mean_ic),
                            "ic_std": float(getattr(res, "ic_std", 0.0)),
                            "turnover": float(getattr(res, "turnover_proxy", 0.0)),
                            "parsimony": float(res.parsimony_penalty),
                            "corr_pen": float(res.correlation_penalty),
                            "ops": int(prog.size),
                            "program": prog.to_string(max_len=180),
                        })
                    diag.record_generation(
                        generation=gen + 1,
                        eval_stats=stats,
                        eval_events=events,
                        best=(top_summary[0] if top_summary else {}),
                    )
                    # Enrich entry with additional optional fields for downstream reporting
                    diag.enrich_last(
                        pop_quantiles=(q or {}),
                        topK=top_summary,
                        ramp={
                            "corr_w": float(cfg.corr_penalty_w * ramp),
                            "ic_std_w": float(cfg.ic_std_penalty_w * ramp),
                            "turnover_w": float(cfg.turnover_penalty_w * ramp),
                            "sharpe_w": float(cfg.sharpe_proxy_w * ramp),
                        },
                        gen_eval_seconds=float(gen_eval_time),
                    )
                    # Optional Pareto front summary for diagnostics
                    try:
                        if getattr(cfg, "moea_enabled", False):
                            def _objs(res: el_module.EvalResult) -> tuple[float, float, float, float]:
                                return (
                                    float(res.mean_ic),
                                    float(res.sharpe_proxy),
                                    float(-res.turnover_proxy),
                                    float(-res.parsimony_penalty),
                                )
                            # Build naive first-front (non-dominated set)
                            objs = [(_i, _objs(_r)) for _i, _r in eval_results if np.isfinite(_r.fitness)]
                            # Simple O(n^2) check for front 0
                            pareto = []
                            for i, oi in objs:
                                dominated = False
                                for j, oj in objs:
                                    if i == j:
                                        continue
                                    if all(oj[k] >= oi[k] for k in range(len(oi))) and any(oj[k] > oi[k] for k in range(len(oi))):
                                        dominated = True
                                        break
                                if not dominated:
                                    pareto.append((i, oi))
                            pf = [{
                                "idx": int(i),
                                "fp": pop[i].fingerprint,
                                "obj": {"ic": oi[0], "sh": oi[1], "neg_turn": oi[2], "neg_complex": oi[3]},
                                "ops": int(pop[i].size),
                            } for (i, oi) in pareto[: min(10, len(pareto))]]
                            diag.enrich_last(pareto_front=pf, pareto_size=len(pf))
                    except Exception:
                        pass
                    # Novelty vs HOF for best-of-gen (mean Spearman component)
                    try:
                        if tmp_sorted:
                            best_res = tmp_sorted[0][1]
                            pp = getattr(best_res, 'processed_predictions', None)
                            mean_corr = None
                            if pp is not None and pp.size > 0:
                                mean_corr = hof_module.get_mean_corr_component_with_hof(pp.ravel())
                            diag.enrich_last(novelty={"hof_mean_corr_best": float(mean_corr) if mean_corr is not None else 0.0})
                    except Exception:
                        pass
                    # Include a compact HOF snapshot for provenance
                    try:
                        diag.enrich_last(hof=hof_module.snapshot(limit=10))
                        # Also include opcode sets for structural heatmaps (as lists for JSON)
                        try:
                            hof_sets = hof_module.get_hof_opcode_sets(limit=10)
                            hof_sets_list = [list(s) for s in hof_sets]
                            diag.enrich_last(hof_opcodes=hof_sets_list)
                        except Exception:
                            pass
                        # Include correlation matrix among HOF rank prediction vectors (latest up to limit)
                        try:
                            fps, corr_mat = hof_module.get_rank_corr_matrix(limit=10, absolute=True)
                            diag.enrich_last(hof_rank_corr={"fps": fps, "matrix": corr_mat})
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Add population histogram for live charts
                    try:
                        if valid_scores and q is not None:
                            arr = np.array(valid_scores)
                            lo = float(np.min(arr))
                            hi = float(np.max(arr))
                            if hi <= lo:
                                hi = lo + 1e-6
                            bins = np.linspace(lo, hi, 21)
                            counts, edges = np.histogram(arr, bins=bins)
                            diag.enrich_last(pop_hist={"edges": edges.tolist(), "counts": counts.astype(int).tolist()})
                    except Exception:
                        pass
                    # Emit a structured per-generation diagnostic line for live dashboards
                    try:
                        import json as _json
                        last_entry = diag.get_all()[-1] if hasattr(diag, "get_all") and diag.get_all() else None
                        if last_entry is not None:
                            logging.getLogger(__name__).info("DIAG %s", _json.dumps(last_entry))
                    except Exception:
                        pass
                except Exception:
                    pass

            # Sort by the configured selection score but keep EvalResult intact for logging
            eval_results.sort(key=lambda x: _sel_score(x[1]), reverse=True)

            # Add top-K from this generation into the HOF to increase saved diversity.
            if eval_results and eval_results[0][1].fitness > -np.inf:
                k = max(1, int(getattr(cfg, "hof_per_gen", 1)))
                added = 0
                for rank_idx, (prog_idx_k, res_k) in enumerate(eval_results):
                    if not np.isfinite(res_k.fitness):
                        continue
                    try:
                        prog_k = pop[prog_idx_k]
                        metrics_k = el_module.evaluate_program(
                            prog_k, dh_module, hof_module, INITIAL_STATE_VARS, return_preds=True, ctx=_CTX
                        )
                        add_program_to_hof(prog_k, metrics_k, gen)
                        if metrics_k.processed_predictions is not None:
                            update_correlation_hof(prog_k.fingerprint, metrics_k.processed_predictions)
                        added += 1
                        if added >= k:
                            break
                    except Exception:
                        continue

            print_generation_summary(gen, pop, eval_results)

            if not eval_results or eval_results[0][1].fitness <= -float('inf'):
                logger.info(
                    "Gen %s | No valid programs. Restarting population and HOF.",
                    gen + 1,
                )
                pop = [_random_prog(cfg) for _ in range(cfg.pop_size)]
                initialize_evaluation_cache(cfg.eval_cache_size)
                clear_hof()
                gen_eval_times_history.clear()
                continue

            eta_str = ""
            if gen_eval_times_history:
                avg_gen_time = np.mean(gen_eval_times_history)
                remaining_gens = cfg.generations - (gen + 1)
                if avg_gen_time > 0 and remaining_gens > 0:
                    eta_seconds = remaining_gens * avg_gen_time
                    eta_str = f" | ETA {time.strftime('%Hh%Mm%Ss', time.gmtime(eta_seconds))}"
            
            best_prog_idx, best_metrics = eval_results[0]
            best_fit = best_metrics.fitness
            best_ic = best_metrics.mean_ic
            best_program_obj = pop[best_prog_idx]

            # Track progress for adaptive rates
            if best_fit > prev_best_fit + 1e-6:
                no_improve_gens = 0
                prev_best_fit = best_fit
            else:
                no_improve_gens += 1
            best_fixed = getattr(best_metrics, "fitness_static", None)
            if best_fixed is not None and np.isfinite(best_fixed):
                logger.info(
                    "Gen %3d BestFit %+7.4f FixedFit %+7.4f MeanIC %+7.4f Ops %2d EvalTime %.1fs%s\n  └─ %s",
                    gen + 1,
                    best_fit,
                    best_fixed,
                    best_ic,
                    best_program_obj.size,
                    gen_eval_time,
                    eta_str,
                    best_program_obj.to_string(max_len=100),
                )
            else:
                logger.info(
                    "Gen %3d BestFit %+7.4f MeanIC %+7.4f Ops %2d EvalTime %.1fs%s\n  └─ %s",
                    gen + 1,
                    best_fit,
                    best_ic,
                    best_program_obj.size,
                    gen_eval_time,
                    eta_str,
                    best_program_obj.to_string(max_len=100),
                )

            logger.debug(
                "Gen %s | Building new population from elites and offspring",
                gen + 1,
            )

            new_pop: List[AlphaProgram] = []
            # Optional Pareto-based elite selection (NSGA-II style fronts)
            if getattr(cfg, "moea_enabled", False):
                def _objectives(res: el_module.EvalResult) -> tuple[float, float, float, float]:
                    return (
                        float(res.mean_ic),                 # maximize
                        float(res.sharpe_proxy),            # maximize
                        float(-res.turnover_proxy),         # minimize turnover
                        float(-res.parsimony_penalty),      # minimize complexity
                    )
                def _nondominated_sort(objs: list[tuple[float, ...]]):
                    n = len(objs)
                    S = [set() for _ in range(n)]
                    n_dom = [0] * n
                    fronts: list[list[int]] = []
                    for p in range(n):
                        for q in range(n):
                            if p == q:
                                continue
                            op, oq = objs[p], objs[q]
                            if all(op[i] >= oq[i] for i in range(len(op))) and any(op[i] > oq[i] for i in range(len(op))):
                                S[p].add(q)
                            elif all(oq[i] >= op[i] for i in range(len(op))) and any(oq[i] > op[i] for i in range(len(op))):
                                n_dom[p] += 1
                        if n_dom[p] == 0:
                            if not fronts:
                                fronts.append([])
                            fronts[0].append(p)
                    i_f = 0
                    while i_f < len(fronts):
                        next_front: list[int] = []
                        for p in fronts[i_f]:
                            for q in S[p]:
                                n_dom[q] -= 1
                                if n_dom[q] == 0:
                                    next_front.append(q)
                        if next_front:
                            fronts.append(next_front)
                        i_f += 1
                    return fronts
                def _crowding(front: list[int], objs: list[tuple[float, ...]]):
                    if not front:
                        return {}
                    m = len(objs[0])
                    dist = {i: 0.0 for i in front}
                    for k in range(m):
                        front_sorted = sorted(front, key=lambda i: objs[i][k])
                        dist[front_sorted[0]] = float("inf")
                        dist[front_sorted[-1]] = float("inf")
                        vals = [objs[i][k] for i in front_sorted]
                        vmin, vmax = vals[0], vals[-1]
                        rng = (vmax - vmin) if (vmax > vmin) else 1.0
                        for j in range(1, len(front_sorted) - 1):
                            prev_v = objs[front_sorted[j - 1]][k]
                            next_v = objs[front_sorted[j + 1]][k]
                            dist[front_sorted[j]] += (next_v - prev_v) / rng
                    return dist
                # Build objective vectors aligned by index
                # Default to using only valid results
                objs_list: list[tuple[float, ...]] = [None] * len(pop)  # type: ignore
                valid_idx: list[int] = []
                for idx_in_pop, res in eval_results:
                    if np.isfinite(res.fitness):
                        objs_list[idx_in_pop] = _objectives(res)
                        valid_idx.append(idx_in_pop)
                # Map to a compact list for sorting
                idx_map = {idx: k for k, idx in enumerate(valid_idx)}
                compact_objs = [objs_list[i] for i in valid_idx]
                fronts_local = _nondominated_sort(compact_objs)
                # Elite fraction cap
                elite_cap = max(0, min(cfg.pop_size, int(round(cfg.pop_size * float(getattr(cfg, "moea_elite_frac", 0.2))))))
                picked = 0
                for f in fronts_local:
                    # Convert compact indices back to population indices
                    front_pop_idx = [valid_idx[i] for i in f]
                    cd = _crowding([idx_map[i] for i in front_pop_idx], compact_objs)
                    for i in sorted(front_pop_idx, key=lambda x: cd.get(idx_map.get(x, -1), 0.0), reverse=True):
                        new_pop.append(pop[i].copy())
                        picked += 1
                        if picked >= elite_cap:
                            break
                    if picked >= elite_cap:
                        break
                # Fallback to at least one elite
                if not new_pop and valid_idx:
                    new_pop.append(pop[valid_idx[0]].copy())

            # Diversity proxy and adaptive breeding rates
            try:
                unique_fps = len({p.fingerprint for p in pop})
                diversity_ratio = unique_fps / max(1, len(pop))
            except Exception:
                diversity_ratio = 1.0

            # Stagnation factor grows when no improvement; patience ~ 5 gens or 10% of total
            patience = max(5, cfg.generations // 10 if cfg.generations > 0 else 5)
            stagnation_factor = min(1.0, no_improve_gens / patience)

            # Increase exploration when stagnating or when diversity is low
            p_mut_eff = max(0.0, min(1.0, cfg.p_mut * (1.0 + 0.5 * stagnation_factor)))
            fresh_rate_eff = max(0.0, min(1.0, cfg.fresh_rate * (1.0 + (1.0 - diversity_ratio))))

            # Soft-restart heuristic: boost fresh intake if lower quartile deteriorates over gens
            valid_scores_for_q = [r[1].fitness for r in eval_results if np.isfinite(r[1].fitness)]
            if valid_scores_for_q:
                arrq = np.array(valid_scores_for_q)
                q25 = float(np.quantile(arrq, 0.25))
                if prev_q25 is not None and q25 < prev_q25 - 1e-6:
                    q25_deteriorate_streak += 1
                else:
                    q25_deteriorate_streak = 0
                prev_q25 = q25
                # Up to +0.2 extra fresh when deteriorating for several gens
                fresh_rate_eff = min(1.0, fresh_rate_eff + 0.05 * q25_deteriorate_streak)

            # Anneal crossover upwards to recombine mature building blocks
            p_cross_eff = max(0.0, min(1.0, cfg.p_cross + 0.5 * ramp))

            # Rank-based sampling weights for tournaments (softmax over normalized ranks)
            valid_indices_for_tournament = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]
            valid_scores = [pop_fitness_scores[i] for i in valid_indices_for_tournament]
            rank_weights = None
            if valid_indices_for_tournament:
                # Higher rank -> higher weight. Normalize to [0,1].
                order = np.argsort(valid_scores)
                ranks = np.empty(len(valid_scores), dtype=float)
                ranks[order] = np.arange(len(valid_scores))
                rank_norm = ranks / max(1, len(valid_scores) - 1)
                # Temperature schedule: floor + (target - floor) * ramp
                beta_floor = float(getattr(cfg, 'rank_softmax_beta_floor', 0.0))
                beta_target = float(getattr(cfg, 'rank_softmax_beta_target', 2.0))
                beta = beta_floor + (beta_target - beta_floor) * float(ramp)
                # Use exp(beta * rank_norm) so top ranks dominate as beta grows
                w = np.exp(beta * rank_norm)
                # Guard against inf/nan
                if np.all(np.isfinite(w)) and np.any(w > 0):
                    rank_weights = w.tolist()
            
            # Legacy scalar elites when MOEA is disabled
            if not getattr(cfg, "moea_enabled", False):
                elites_added_fingerprints = set()
                for res_idx, res_metrics in eval_results:
                    if res_metrics.fitness <= -float('inf'):
                        continue
                    prog_candidate = pop[res_idx]
                    fp_cand = prog_candidate.fingerprint
                    if fp_cand not in elites_added_fingerprints or cfg.keep_dupes_in_hof: 
                        new_pop.append(prog_candidate.copy())
                        elites_added_fingerprints.add(fp_cand)
                    if len(new_pop) >= cfg.elite_keep:
                        break
            
            if not new_pop and eval_results and eval_results[0][1].fitness > -float('inf'):
                 new_pop.append(pop[eval_results[0][0]].copy())

            while len(new_pop) < cfg.pop_size:
                if _RNG.random() < fresh_rate_eff:
                    new_pop.append(_random_prog(cfg))
                    continue
                
                valid_indices_for_tournament = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]
                if not valid_indices_for_tournament: 
                    new_pop.extend([_random_prog(cfg) for _ in range(cfg.pop_size - len(new_pop))])
                    break

                num_to_sample = cfg.tournament_k * 2
                if len(valid_indices_for_tournament) > 0:
                    if rank_weights is not None:
                        # Weighted sampling (with replacement) by rank-based weights
                        tournament_indices_pool = random.choices(valid_indices_for_tournament, weights=rank_weights, k=num_to_sample)
                    else:
                        # Fallback to uniform sampling
                        if len(valid_indices_for_tournament) < num_to_sample:
                            tournament_indices_pool = random.choices(valid_indices_for_tournament, k=num_to_sample)
                        else:
                            tournament_indices_pool = random.sample(valid_indices_for_tournament, num_to_sample)
                else:
                    new_pop.append(_random_prog(cfg))
                    continue

                if getattr(cfg, "moea_enabled", False):
                    # Binary tournament by selection rank (fallback to scalar if undefined)
                    def _score_for_tour(i_tour: int) -> float:
                        # Use scalar selection score as fallback
                        return float(pop_fitness_scores[i_tour])
                    parent1_idx = max(tournament_indices_pool[:cfg.tournament_k], key=_score_for_tour)
                    parent2_idx = max(tournament_indices_pool[cfg.tournament_k:], key=_score_for_tour)
                else:
                    parent1_idx = max(tournament_indices_pool[:cfg.tournament_k], key=lambda i_tour: pop_fitness_scores[i_tour])
                    parent2_idx = max(tournament_indices_pool[cfg.tournament_k:], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent_a, parent_b = pop[parent1_idx], pop[parent2_idx]

                child: AlphaProgram
                if _RNG.random() < p_cross_eff:
                    child = parent_a.crossover(
                        parent_b,
                        max_setup_ops=cfg.max_setup_ops,
                        max_predict_ops=cfg.max_predict_ops,
                        max_update_ops=cfg.max_update_ops,
                        rng=_RNG
                    )
                else:
                    child = parent_a.copy() if _RNG.random() < 0.5 else parent_b.copy()
                
                if _RNG.random() < p_mut_eff:
                    child = _mutate_prog(child, cfg)
                
                new_pop.append(child)
            pop = new_pop

            logger.debug(
                "Gen %s | Next generation population prepared (%s programs)",
                gen + 1,
                len(pop),
            )

    except KeyboardInterrupt:
        logger.info(
            "[Ctrl‑C] Evolution stopped early. Processing current HOF..."
        )
    
    final_top_programs_with_ic = get_final_hof_programs()
    return final_top_programs_with_ic


def evolve(cfg: EvoConfig) -> List[Tuple[AlphaProgram, float]]:
    """Compatibility helper: build EvalContext from disk and delegate to context API."""
    # Precompute column matrices for all cross-sectional vector features (strip _t)
    try:
        from alpha_framework.alpha_framework_types import CROSS_SECTIONAL_FEATURE_VECTOR_NAMES as _VECN
        cols = [c.replace("_t", "") for c in _VECN if c != "sector_id_vector"]
    except Exception:
        cols = None
    ctx = make_eval_context_from_dir(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag,
        dh_module=dh_module,
        sector_mapping=cfg.sector_mapping,
        precompute_columns=cols,
    )
    return evolve_with_context(cfg, ctx)
