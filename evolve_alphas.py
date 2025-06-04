from __future__ import annotations
import random
import time
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import numpy as np

from alpha_framework import AlphaProgram, TypeId, CROSS_SECTIONAL_FEATURE_VECTOR_NAMES, SCALAR_FEATURE_NAMES
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
    el_module.configure_evaluation(
        parsimony_penalty=cfg.parsimony_penalty,
        max_ops=cfg.max_ops,
        xs_flatness_guard=cfg.xs_flat_guard,
        temporal_flatness_guard=cfg.t_flat_guard,
        early_abort_bars=cfg.early_abort_bars,
        early_abort_xs=cfg.early_abort_xs,
        early_abort_t=cfg.early_abort_t,
        scale_method=cfg.scale
    )
    initialize_hof(
        max_size=cfg.hof_size,
        keep_dupes=cfg.keep_dupes_in_hof,
        corr_penalty_weight=cfg.corr_penalty_w,
        corr_cutoff=cfg.corr_cutoff
    )
    initialize_evaluation_cache()


FEATURE_VARS: Dict[str, TypeId] = {name: "vector" for name in CROSS_SECTIONAL_FEATURE_VECTOR_NAMES}
FEATURE_VARS.update({name: "scalar" for name in SCALAR_FEATURE_NAMES})
if "const_1" not in FEATURE_VARS:
    FEATURE_VARS["const_1"] = "scalar"
if "const_neg_1" not in FEATURE_VARS:
    FEATURE_VARS["const_neg_1"] = "scalar"

INITIAL_STATE_VARS: Dict[str, TypeId] = {
    "prev_s1_vec": "vector",
    "rolling_mean_custom": "vector"
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(np.random.randint(0, 2**32))


def _evaluate_prog_for_pool(args: Tuple[int, AlphaProgram]) -> Tuple[int, float, float, Optional[np.ndarray]]:
    idx, prog = args
    return (
        idx,
        *evaluate_program(prog, dh_module, hof_module, INITIAL_STATE_VARS)
    )

def _random_prog(cfg: EvoConfig) -> AlphaProgram: # Signature changed
    return AlphaProgram.random_program(
        FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=cfg.max_ops, rng=_rng()
    )

def _mutate_prog(p: AlphaProgram, cfg: EvoConfig) -> AlphaProgram: # Signature changed
    return p.mutate(
        FEATURE_VARS, INITIAL_STATE_VARS, max_total_ops=cfg.max_ops, rng=_rng()
    )

###############################################################################
# EVOLVE LOOP ##############################################################
###############################################################################

def evolve(cfg: EvoConfig) -> List[Tuple[AlphaProgram, float]]: # Signature changed
    _sync_evolution_configs_from_config(cfg) 

    pop: List[AlphaProgram] = [_random_prog(cfg) for _ in range(cfg.pop_size)]
    gen_eval_times_history: List[float] = []

    try:
        for gen in range(cfg.generations):
            t_start_gen = time.perf_counter()
            eval_results: List[Tuple[int, float, float, Optional[np.ndarray]]] = []
            pop_fitness_scores = np.full(cfg.pop_size, -np.inf)

            bar = pbar(range(cfg.pop_size), desc=f"Gen {gen+1}/{cfg.generations}", disable=cfg.quiet)

            if cfg.n_workers > 1:
                with mp.Pool(processes=cfg.n_workers) as pool:
                    result_iter = pool.imap_unordered(_evaluate_prog_for_pool, enumerate(pop))
                    for _ in bar:
                        i, score, mean_ic, processed_preds_matrix = next(result_iter)
                        eval_results.append((i, score, mean_ic, processed_preds_matrix))
                        pop_fitness_scores[i] = score
                        if not cfg.quiet and hasattr(bar, 'set_postfix_str'):
                            valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                            best_score_so_far = np.max(valid_scores) if valid_scores.size > 0 else -np.inf
                            bar.set_postfix_str(f"BestFit: {best_score_so_far:.4f}")
            else:
                for i in bar:
                    prog = pop[i]
                    score, mean_ic, processed_preds_matrix = evaluate_program(
                        prog,
                        dh_module,
                        hof_module,
                        INITIAL_STATE_VARS
                    )
                    eval_results.append((i, score, mean_ic, processed_preds_matrix))
                    pop_fitness_scores[i] = score
                    if not cfg.quiet and hasattr(bar, 'set_postfix_str'):
                        valid_scores = pop_fitness_scores[pop_fitness_scores > -np.inf]
                        best_score_so_far = np.max(valid_scores) if valid_scores.size > 0 else -np.inf
                        bar.set_postfix_str(f"BestFit: {best_score_so_far:.4f}")
            
            gen_eval_time = time.perf_counter() - t_start_gen
            if gen_eval_time > 0:
                gen_eval_times_history.append(gen_eval_time)

            eval_results.sort(key=lambda x: x[1], reverse=True)

            if eval_results and eval_results[0][1] > -np.inf:
                best_prog_idx_this_gen, best_fit_this_gen, best_ic_this_gen, best_preds_matrix_this_gen = eval_results[0]
                best_program_instance_this_gen = pop[best_prog_idx_this_gen]
                
                add_program_to_hof(best_program_instance_this_gen, best_fit_this_gen, best_ic_this_gen, best_preds_matrix_this_gen)

                if best_preds_matrix_this_gen is not None:
                    update_correlation_hof(best_program_instance_this_gen.fingerprint, best_preds_matrix_this_gen)

            print_generation_summary(gen, pop, eval_results)

            if not eval_results or eval_results[0][1] <= -float('inf'):
                print(f"Gen {gen+1:3d} | No valid programs. Restarting population and HOF.")
                pop = [_random_prog(cfg) for _ in range(cfg.pop_size)]
                initialize_evaluation_cache() 
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
            
            best_prog_idx, best_fit, best_ic, _ = eval_results[0]
            best_program_obj = pop[best_prog_idx]
            print(
                f"Gen {gen+1:3d} BestThisGenFit {best_fit:+.4f} MeanIC {best_ic:+.4f} Ops {best_program_obj.size:2d} EvalTime {gen_eval_time:.1f}s{eta_str}\n"
                f"  └─ {best_program_obj.to_string(max_len=100)}"
            )

            new_pop: List[AlphaProgram] = []
            
            elites_added_fingerprints = set()
            for res_idx, res_score, _, _ in eval_results:
                if res_score <= -float('inf'):
                    continue
                prog_candidate = pop[res_idx]
                fp_cand = prog_candidate.fingerprint
                if fp_cand not in elites_added_fingerprints or cfg.keep_dupes_in_hof: 
                    new_pop.append(prog_candidate.copy())
                    elites_added_fingerprints.add(fp_cand)
                if len(new_pop) >= cfg.elite_keep:
                    break
            
            if not new_pop and eval_results and eval_results[0][1] > -float('inf'):
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
                if random.random() < cfg.p_cross:
                    child = parent_a.crossover(parent_b, rng=_rng())
                else:
                    child = parent_a.copy() if random.random() < 0.5 else parent_b.copy()
                
                if random.random() < cfg.p_mut:
                    child = _mutate_prog(child, cfg)
                
                new_pop.append(child)
            pop = new_pop
            
    except KeyboardInterrupt:
        print("\n[Ctrl‑C] Evolution stopped early. Processing current HOF...")
    
    final_top_programs_with_ic = get_final_hof_programs()
    return final_top_programs_with_ic
