from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Deque
from collections import deque
from multiprocessing import Pool, cpu_count
import numpy as np
import logging

from alpha_framework import (
    AlphaProgram,
    TypeId,
    CROSS_SECTIONAL_FEATURE_VECTOR_NAMES,
    CROSS_SECTIONAL_FEATURE_MATRIX_NAMES,
)
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
    pbar
)
from evolution_components import data_handling as dh_module
from evolution_components import hall_of_fame_manager as hof_module
from evolution_components import evaluation_logic as el_module

from config import EvoConfig # New import

###############################################################################
# CLI & CONFIG REMOVED ########################################################
###############################################################################
# _parse_cli function REMOVED
# Global args object initialization REMOVED

def _sync_evolution_configs_from_config(cfg: EvoConfig): # Renamed and signature changed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    initialize_data(
        data_dir=cfg.data_dir,
        strategy=cfg.max_lookback_data_option,
        min_common_points=cfg.min_common_points,
        eval_lag=cfg.eval_lag
    )
    dh_module.configure_feature_scaling(cfg.feature_scale_method)
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
        sharpe_proxy_weight=cfg.sharpe_proxy_w
    )
    initialize_hof(
        max_size=cfg.hof_size,
        keep_dupes=cfg.keep_dupes_in_hof,
        corr_penalty_weight=cfg.corr_penalty_w,
        corr_cutoff=cfg.corr_cutoff
    )
    initialize_evaluation_cache(cfg.eval_cache_size)


FEATURE_VARS: Dict[str, TypeId] = {
    name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES
}
FEATURE_VARS.update({name: "matrix" for name in CROSS_SECTIONAL_FEATURE_MATRIX_NAMES})
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

def _random_prog(cfg: EvoConfig) -> AlphaProgram: # Signature changed
    return AlphaProgram.random_program(
        FEATURE_VARS,
        INITIAL_STATE_VARS,
        max_total_ops=cfg.max_ops,
        max_setup_ops=cfg.max_setup_ops,
        max_predict_ops=cfg.max_predict_ops,
        max_update_ops=cfg.max_update_ops,
        max_scalar_operands=cfg.max_scalar_operands,
        max_vector_operands=cfg.max_vector_operands,
        max_matrix_operands=cfg.max_matrix_operands,
    )

def _mutate_prog(p: AlphaProgram, cfg: EvoConfig) -> AlphaProgram: # Signature changed
    return p.mutate(
        FEATURE_VARS,
        INITIAL_STATE_VARS,
        max_total_ops=cfg.max_ops,
        max_setup_ops=cfg.max_setup_ops,
        max_predict_ops=cfg.max_predict_ops,
        max_update_ops=cfg.max_update_ops,
        max_scalar_operands=cfg.max_scalar_operands,
        max_vector_operands=cfg.max_vector_operands,
        max_matrix_operands=cfg.max_matrix_operands,
    )

def _eval_worker(args) -> Tuple[int, el_module.EvalResult]:
    idx, prog = args
    result = evaluate_program(
        prog,
        dh_module,
        hof_module,
        INITIAL_STATE_VARS
    )
    return idx, result

###############################################################################
# EVOLVE LOOP ##############################################################
###############################################################################

def evolve(cfg: EvoConfig) -> List[Tuple[AlphaProgram, float]]: # Signature changed
    _sync_evolution_configs_from_config(cfg)

    logger = logging.getLogger(__name__)

    pop: List[AlphaProgram] = [_random_prog(cfg) for _ in range(cfg.pop_size)]
    gen_eval_times_history: List[float] = []

    p_mut = cfg.p_mut
    p_cross = cfg.p_cross
    fitness_window: Deque[float] = deque(maxlen=5)
    improvement_threshold = 1e-3

    try:
        for gen in range(cfg.generations):
            logger.info(
                "Gen %s/%s | Starting evaluation of %s programs",
                gen + 1,
                cfg.generations,
                len(pop),
            )
            t_start_gen = time.perf_counter()
            eval_results: List[Tuple[int, el_module.EvalResult]] = []
            pop_fitness_scores = np.full(cfg.pop_size, -np.inf)

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
            
            gen_eval_time = time.perf_counter() - t_start_gen
            if gen_eval_time > 0:
                gen_eval_times_history.append(gen_eval_time)

            logger.info(
                "Gen %s | Evaluation completed in %.1fs",
                gen + 1,
                gen_eval_time,
            )

            eval_results.sort(key=lambda x: x[1].fitness, reverse=True)

            if eval_results and eval_results[0][1].fitness > -np.inf:
                best_prog_idx_this_gen, best_metrics_this_gen = eval_results[0]
                best_preds_matrix_this_gen = best_metrics_this_gen.processed_predictions
                best_program_instance_this_gen = pop[best_prog_idx_this_gen]

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
                if random.random() < cfg.fresh_rate:
                    new_pop.append(_random_prog(cfg))
                    continue
                
                valid_indices_for_tournament = [k_idx for k_idx, s_k in enumerate(pop_fitness_scores) if s_k > -np.inf]
                if not valid_indices_for_tournament: 
                    new_pop.extend([_random_prog(cfg) for _ in range(cfg.pop_size - len(new_pop))])
                    break

                num_to_sample = cfg.tournament_k * 2
                if len(valid_indices_for_tournament) < num_to_sample and len(valid_indices_for_tournament) > 0 :
                    tournament_indices_pool = random.choices(valid_indices_for_tournament, k=num_to_sample)
                elif len(valid_indices_for_tournament) >= num_to_sample:
                    tournament_indices_pool = random.sample(valid_indices_for_tournament, num_to_sample)
                else: 
                    new_pop.append(_random_prog(cfg))
                    continue

                parent1_idx = max(tournament_indices_pool[:cfg.tournament_k], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent2_idx = max(tournament_indices_pool[cfg.tournament_k:], key=lambda i_tour: pop_fitness_scores[i_tour])
                parent_a, parent_b = pop[parent1_idx], pop[parent2_idx]

                child: AlphaProgram
                if random.random() < p_cross:
                    child = parent_a.crossover(
                        parent_b,
                        max_setup_ops=cfg.max_setup_ops,
                        max_predict_ops=cfg.max_predict_ops,
                        max_update_ops=cfg.max_update_ops,
                    )
                else:
                    child = parent_a.copy() if random.random() < 0.5 else parent_b.copy()
                
                if random.random() < p_mut:
                    child = _mutate_prog(child, cfg)
                
                new_pop.append(child)
            pop = new_pop

            logger.debug(
                "Gen %s | Next generation population prepared (%s programs)",
                gen + 1,
                len(pop),
            )

            fitness_window.append(best_fit)
            if len(fitness_window) == fitness_window.maxlen:
                improvement = fitness_window[-1] - fitness_window[0]
                if cfg.adaptive_mutation:
                    if improvement < improvement_threshold:
                        p_mut = min(p_mut * 1.1, 0.95)
                    else:
                        p_mut = max(p_mut * 0.9, 0.01)
                if cfg.adaptive_crossover:
                    if improvement < improvement_threshold:
                        inc = p_cross * 1.1 if p_cross > 0 else 0.1
                        p_cross = min(inc, 0.95)
                    else:
                        p_cross = max(p_cross * 0.9, 0.0)
                logger.info(
                    "Gen %s | Adaptive probabilities -> p_mut=%.3f p_cross=%.3f",
                    gen + 1,
                    p_mut,
                    p_cross,
                )

    except KeyboardInterrupt:
        logger.info(
            "[Ctrl‑C] Evolution stopped early. Processing current HOF..."
        )
    
    final_top_programs_with_ic = get_final_hof_programs()
    return final_top_programs_with_ic
