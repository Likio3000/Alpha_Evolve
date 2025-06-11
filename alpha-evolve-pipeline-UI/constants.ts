
import { ParameterDefinition, ParameterSection } from './types';

// Defaults are set to match run_pipeline_all_args.sh where specified,
// otherwise they fall back to config.py or sensible UI defaults.
export const PARAMETER_DEFINITIONS: ParameterDefinition[] = [
  // Positional
  { id: 'generations', label: 'Generations', type: 'number', cliFlag: '', defaultValue: 15, isPositional: true, description: 'Number of generations to run the evolution.' },
  
  // Evolution Core
  { id: 'seed', label: 'Seed', type: 'number', cliFlag: '--seed', defaultValue: 42, description: 'Random seed for reproducibility.' },
  { id: 'pop_size', label: 'Population Size', type: 'number', cliFlag: '--pop_size', defaultValue: 100, description: 'Number of individuals in the population.' },
  { id: 'tournament_k', label: 'Tournament K', type: 'number', cliFlag: '--tournament_k', defaultValue: 10, description: 'Size of the tournament selection.' },
  { id: 'p_mut', label: 'Mutation Probability', type: 'number', cliFlag: '--p_mut', defaultValue: 0.9, step: 0.01, min: 0, max: 1, description: 'Probability of mutating an individual.' },
  { id: 'p_cross', label: 'Crossover Probability', type: 'number', cliFlag: '--p_cross', defaultValue: 0.4, step: 0.01, min: 0, max: 1, description: 'Probability of performing crossover.' },
  { id: 'elite_keep', label: 'Elite Keep', type: 'number', cliFlag: '--elite_keep', defaultValue: 6, description: 'Number of elite individuals to keep.' },
  { id: 'fresh_rate', label: 'Fresh Rate', type: 'number', cliFlag: '--fresh_rate', defaultValue: 0.25, step: 0.01, min: 0, max: 1, description: 'Rate of introducing fresh individuals.' },
  
  // Complexity & Similarity Guards
  { id: 'max_ops', label: 'Max Operations (Total)', type: 'number', cliFlag: '--max_ops', defaultValue: 87, description: 'Maximum total operations in a program.' },
  { id: 'max_setup_ops', label: 'Max Setup Operations', type: 'number', cliFlag: '--max_setup_ops', defaultValue: 21, description: 'Maximum operations in the setup block.' },
  { id: 'max_predict_ops', label: 'Max Predict Operations', type: 'number', cliFlag: '--max_predict_ops', defaultValue: 21, description: 'Maximum operations in the predict block.' },
  { id: 'max_update_ops', label: 'Max Update Operations', type: 'number', cliFlag: '--max_update_ops', defaultValue: 45, description: 'Maximum operations in the update block.' },
  { id: 'max_scalar_operands', label: 'Max Scalar Operands', type: 'number', cliFlag: '--max_scalar_operands', defaultValue: 10 },
  { id: 'max_vector_operands', label: 'Max Vector Operands', type: 'number', cliFlag: '--max_vector_operands', defaultValue: 16 },
  { id: 'max_matrix_operands', label: 'Max Matrix Operands', type: 'number', cliFlag: '--max_matrix_operands', defaultValue: 4 },
  { id: 'parsimony_penalty', label: 'Parsimony Penalty', type: 'number', cliFlag: '--parsimony_penalty', defaultValue: 0.002, step: 0.0001, description: 'Penalty for program complexity.' },
  { id: 'corr_penalty_w', label: 'Correlation Penalty Weight', type: 'number', cliFlag: '--corr_penalty_w', defaultValue: 0.25, step: 0.01, description: 'Weight for correlation penalty.' },
  { id: 'corr_cutoff', label: 'Correlation Cutoff', type: 'number', cliFlag: '--corr_cutoff', defaultValue: 0.15, step: 0.01, description: 'Correlation cutoff for Hall of Fame.' },
  
  // Evaluation Specifics
  { id: 'xs_flat_guard', label: 'XS Flat Guard', type: 'number', cliFlag: '--xs_flat_guard', defaultValue: 0.02, step: 0.001, description: 'Cross-sectional flatness guard threshold.' },
  { id: 't_flat_guard', label: 'Temporal Flat Guard', type: 'number', cliFlag: '--t_flat_guard', defaultValue: 0.005, step: 0.001, description: 'Temporal flatness guard threshold.' },
  { id: 'early_abort_bars', label: 'Early Abort Bars', type: 'number', cliFlag: '--early_abort_bars', defaultValue: 60, description: 'Number of bars for early abort check.' },
  { id: 'early_abort_xs', label: 'Early Abort XS', type: 'number', cliFlag: '--early_abort_xs', defaultValue: 0.02, step: 0.001, description: 'Cross-sectional threshold for early abort.' },
  { id: 'early_abort_t', label: 'Early Abort Temporal', type: 'number', cliFlag: '--early_abort_t', defaultValue: 0.005, step: 0.001, description: 'Temporal threshold for early abort.' },
  { id: 'hof_size', label: 'Hall of Fame Size', type: 'number', cliFlag: '--hof_size', defaultValue: 20 },
  { id: 'scale', label: 'Scale Method', type: 'select', cliFlag: '--scale', defaultValue: 'zscore', options: [{value: 'zscore', label: 'Z-Score'}, {value: 'rank', label: 'Rank'}, {value: 'sign', label: 'Sign'}], description: 'Method for scaling signals.' },
  { id: 'eval_cache_size', label: 'Evaluation Cache Size', type: 'number', cliFlag: '--eval_cache_size', defaultValue: 128 },
  { id: 'sharpe_proxy_w', label: 'Sharpe Proxy Weight', type: 'number', cliFlag: '--sharpe_proxy_w', defaultValue: 0.0, step: 0.01, description: 'Weight for Sharpe ratio proxy in fitness. (Default from config.py as not in run_pipeline_all_args.sh)' },
  { id: 'keep_dupes_in_hof', label: 'Keep Duplicates in HOF', type: 'boolean', cliFlag: '--keep_dupes_in_hof', defaultValue: false, description: 'Allow duplicate programs in Hall of Fame. (Default from config.py as not in run_pipeline_all_args.sh)' },
  { id: 'flat_bar_threshold', label: 'Flat Bar Threshold', type: 'number', cliFlag: '--flat_bar_threshold', defaultValue: 0.25, step: 0.01, description: 'Threshold for considering a bar flat during evaluation. (Default from config.py as not in run_pipeline_all_args.sh)' },

  // Data Handling
  { id: 'data_dir', label: 'Data Directory', type: 'text', cliFlag: '--data_dir', defaultValue: './data', description: 'Directory containing market data CSVs.' },
  { id: 'max_lookback_data_option', label: 'Max Lookback Data Option', type: 'select', cliFlag: '--max_lookback_data_option', defaultValue: 'common_1200', options: [
    {value: 'common_1200', label: 'Common 1200'}, 
    {value: 'specific_long_10k', label: 'Specific Long 10k'}, 
    {value: 'full_overlap', label: 'Full Overlap'}
  ], description: 'Strategy for handling data lookback.' },
  { id: 'min_common_points', label: 'Min Common Points', type: 'number', cliFlag: '--min_common_points', defaultValue: 1200, description: 'Minimum common data points across symbols.' },
  { id: 'eval_lag', label: 'Evaluation Lag', type: 'number', cliFlag: '--eval_lag', defaultValue: 1, description: 'Lag for evaluation (e.g., predict at T, evaluate with returns at T+lag).' },
  
  // Backtesting
  { id: 'top_to_backtest', label: 'Top Alphas to Backtest', type: 'number', cliFlag: '--top', defaultValue: 10, description: 'Number of top alphas from HOF to backtest.' },
  { id: 'fee', label: 'Fee (bps)', type: 'number', cliFlag: '--fee', defaultValue: 1.0, step: 0.1, description: 'Round-trip commission in basis points.' },
  { id: 'hold', label: 'Holding Period (bars)', type: 'number', cliFlag: '--hold', defaultValue: 1, description: 'Number of bars to hold positions.' },
  { id: 'long_short_n', label: 'Long/Short N', type: 'number', cliFlag: '--long_short_n', defaultValue: 0, description: 'Trade only top/bottom N symbols (0 for all).' },
  { id: 'annualization_factor', label: 'Annualization Factor', type: 'number', cliFlag: '--annualization_factor', defaultValue: 2190.0, step: 1.0, description: 'Factor to annualize metrics (e.g., 365*6 for 4H crypto). (Script default: not set explicitly; UI provides a common one for crypto)' },

  // Execution & Logging
  { id: 'workers', label: 'Workers', type: 'number', cliFlag: '--workers', defaultValue: 2, description: 'Number of worker processes for parallel evaluation.' },
  { id: 'debug_prints', label: 'Enable Debug Prints', type: 'boolean', cliFlag: '--debug_prints', defaultValue: true, description: 'Enable verbose debug printing. (Set to true as per run_pipeline_all_args.sh)' },
  { id: 'run_baselines', label: 'Run Baselines', type: 'boolean', cliFlag: '--run_baselines', defaultValue: true, description: 'Train and evaluate baseline models. (Set to true as per run_pipeline_all_args.sh)' },
  { id: 'retrain_baselines', label: 'Retrain Baselines', type: 'boolean', cliFlag: '--retrain_baselines', defaultValue: false, description: 'Force retraining of baselines even if cached. (Script default: not set, implies false from run_pipeline.py)' },
  { id: 'log_level', label: 'Log Level', type: 'select', cliFlag: '--log-level', defaultValue: 'INFO', options: [
    {value: 'DEBUG', label: 'DEBUG'}, {value: 'INFO', label: 'INFO'}, {value: 'WARNING', label: 'WARNING'}, {value: 'ERROR', label: 'ERROR'}
  ], description: 'Logging verbosity level. (Script default: not set, implies INFO from run_pipeline.py)' },
  { id: 'log_file', label: 'Log File Path', type: 'text', cliFlag: '--log-file', defaultValue: '', description: 'Optional file to write logs to (leave empty for no file). (Script default: not set, implies None from run_pipeline.py)' },
  { id: 'quiet', label: 'Quiet Mode', type: 'boolean', cliFlag: '--quiet', defaultValue: false, description: 'Reduce log verbosity (overrides log-level to WARNING if true). (Script default: not set, implies false from run_pipeline.py)' },
];

export const SECTIONS: ParameterSection[] = [
  {
    title: 'General & Evolution Core',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['generations', 'seed', 'pop_size', 'tournament_k', 'p_mut', 'p_cross', 'elite_keep', 'fresh_rate'].includes(p.id))
  },
  {
    title: 'Complexity & Similarity Guards',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['max_ops', 'max_setup_ops', 'max_predict_ops', 'max_update_ops', 'max_scalar_operands', 'max_vector_operands', 'max_matrix_operands', 'parsimony_penalty', 'corr_penalty_w', 'corr_cutoff'].includes(p.id))
  },
  {
    title: 'Evaluation Specifics',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['xs_flat_guard', 't_flat_guard', 'early_abort_bars', 'early_abort_xs', 'early_abort_t', 'hof_size', 'scale', 'eval_cache_size', 'sharpe_proxy_w', 'keep_dupes_in_hof', 'flat_bar_threshold'].includes(p.id))
  },
  {
    title: 'Data Handling',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['data_dir', 'max_lookback_data_option', 'min_common_points', 'eval_lag'].includes(p.id))
  },
  {
    title: 'Backtesting',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['top_to_backtest', 'fee', 'hold', 'long_short_n', 'annualization_factor'].includes(p.id))
  },
  {
    title: 'Execution & Logging',
    parameters: PARAMETER_DEFINITIONS.filter(p => ['workers', 'debug_prints', 'run_baselines', 'retrain_baselines', 'log_level', 'log_file', 'quiet'].includes(p.id))
  }
];

export const INITIAL_PARAMETERS = PARAMETER_DEFINITIONS.reduce((acc, param) => {
  acc[param.id] = param.defaultValue;
  return acc;
}, {} as Record<string, string | number | boolean>);
