from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple
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
from evolution_components import diagnostics as diag

from config import EvoConfig  # New import

###############################################################################
# CLI & CONFIG REMOVED ########################################################
###############################################################################
# _parse_cli function REMOVED
# Global args object initialization REMOVED

_RNG = np.random.default_rng()


def _sync_evolution_configs_from_config(cfg: EvoConfig):  # Renamed and signature changed
    global _RNG
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _RNG = np.random.default_rng(cfg.seed)
    
    initialize_data(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag
    )
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
        use_train_val_splits=cfg.use_train_val_splits,
        train_points=cfg.train_points,
        val_points=cfg.val_points,
        sector_neutralize=cfg.sector_neutralize,
        winsor_p=cfg.winsor_p,
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
        result = evaluate_program(
            prog,
            dh_module,
            hof_module,
            INITIAL_STATE_VARS
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

def evolve(cfg: EvoConfig) -> List[Tuple[AlphaProgram, float]]: 
    _sync_evolution_configs_from_config(cfg)
    try:
        diag.reset()
    except Exception:
        pass

    logger = logging.getLogger(__name__)

    pop: List[AlphaProgram] = [_random_prog(cfg) for _ in range(cfg.pop_size)]
    gen_eval_times_history: List[float] = []

    try:
        for gen in range(cfg.generations):
            # Anneal correlation penalty and optional eval weights to encourage exploration early
            # Ramp linearly over first third of the run (min 5 gens)
            ramp_gens = max(5, cfg.generations // 3 if cfg.generations > 0 else 5)
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
                    use_train_val_splits=cfg.use_train_val_splits,
                    train_points=cfg.train_points,
                    val_points=cfg.val_points,
                    sector_neutralize=cfg.sector_neutralize,
                    winsor_p=cfg.winsor_p,
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

            # Multiprocessing can fail in restricted environments; fall back to sequential when workers == 1
            if (cfg.workers or 0) > 1:
                with Pool(processes=cfg.workers or cpu_count()) as pool:
                    results_iter = pool.imap_unordered(_eval_worker, enumerate(pop))
                    bar = pbar(results_iter, desc=f"Gen {gen+1}/{cfg.generations}", disable=cfg.quiet, total=cfg.pop_size)
                    for i, result in bar:
                        eval_results.append((i, result))
                        pop_fitness_scores[i] = result.fitness
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
            else:
                # Sequential evaluation
                iterator = pbar(range(len(pop)), desc=f"Gen {gen+1}/{cfg.generations}", disable=cfg.quiet, total=cfg.pop_size)
                for i in iterator:
                    _, result = _eval_worker((i, pop[i]))
                    eval_results.append((i, result))
                    pop_fitness_scores[i] = result.fitness
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
                    valid_scores = [r[1].fitness for r in eval_results if np.isfinite(r[1].fitness)]
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
                    for idx_in_pop, res in eval_results[:K]:
                        prog = pop[idx_in_pop]
                        top_summary.append({
                            "fingerprint": prog.fingerprint,
                            "fitness": float(res.fitness),
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
                    # Also stash distribution and top-K under synthetic keys for downstream report scripts
                    if hasattr(diag, "_GEN_DIAGNOSTICS"):
                        try:
                            diag._GEN_DIAGNOSTICS[-1]["pop_quantiles"] = q or {}
                            diag._GEN_DIAGNOSTICS[-1]["topK"] = top_summary
                            diag._GEN_DIAGNOSTICS[-1]["ramp"] = {"corr_w": float(cfg.corr_penalty_w * ramp),
                                                                   "ic_std_w": float(cfg.ic_std_penalty_w * ramp),
                                                                   "turnover_w": float(cfg.turnover_penalty_w * ramp),
                                                                   "sharpe_w": float(cfg.sharpe_proxy_w * ramp)}
                            diag._GEN_DIAGNOSTICS[-1]["gen_eval_seconds"] = float(gen_eval_time)
                        except Exception:
                            pass
                except Exception:
                    pass

            eval_results.sort(key=lambda x: x[1].fitness, reverse=True)

            if eval_results and eval_results[0][1].fitness > -np.inf:
                best_prog_idx_this_gen, _ = eval_results[0]
                best_program_instance_this_gen = pop[best_prog_idx_this_gen]
                # Re-evaluate winner once with predictions so we can update the HOF correlation store.
                best_metrics_this_gen = el_module.evaluate_program(
                    best_program_instance_this_gen,
                    dh_module,
                    hof_module,
                    INITIAL_STATE_VARS,
                    return_preds=True,
                )
                best_preds_matrix_this_gen = best_metrics_this_gen.processed_predictions


                add_program_to_hof(best_program_instance_this_gen, best_metrics_this_gen, gen)

                if best_preds_matrix_this_gen is not None:
                    update_correlation_hof(best_program_instance_this_gen.fingerprint, best_preds_matrix_this_gen)

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
            logger.info(
                "Gen %3d BestThisGenFit %+7.4f MeanIC %+7.4f Ops %2d EvalTime %.1fs%s\n  └─ %s",
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
                if rng.random() < cfg.fresh_rate:
                    new_pop.append(_random_prog(cfg, rng))
                    continue
                
                valid_indices_for_tournament = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]
                if not valid_indices_for_tournament: 
                    new_pop.extend([_random_prog(cfg, rng) for _ in range(cfg.pop_size - len(new_pop))])
                    break

                num_to_sample = cfg.tournament_k * 2
                if len(valid_indices_for_tournament) < num_to_sample and len(valid_indices_for_tournament) > 0 :
                    tournament_indices_pool = random.choices(valid_indices_for_tournament, k=num_to_sample)
                elif len(valid_indices_for_tournament) >= num_to_sample:
                    tournament_indices_pool = random.sample(valid_indices_for_tournament, num_to_sample)
                else: 
                    new_pop.append(_random_prog(cfg, rng))
                    continue

                parent1_idx = max(tournament_indices_pool[:cfg.tournament_k], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent2_idx = max(tournament_indices_pool[cfg.tournament_k:], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent_a, parent_b = pop[parent1_idx], pop[parent2_idx]

                child: AlphaProgram
                if rng.random() < cfg.p_cross:
                    child = parent_a.crossover(
                        parent_b,
                        max_setup_ops=cfg.max_setup_ops,
                        max_predict_ops=cfg.max_predict_ops,
                        max_update_ops=cfg.max_update_ops,
                        rng=rng
                    )
                else:
                    child = parent_a.copy() if rng.random() < 0.5 else parent_b.copy()
                
                if rng.random() < cfg.p_mut:
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
