import { ParameterDefinition, ParameterSection } from './types';

export const ITER_PARAM_DEFINITIONS: ParameterDefinition[] = [
  // Iterative core
  { id: 'iters', label: 'Iterations', type: 'number', cliFlag: '--iters', defaultValue: 2, description: 'Number of improvement rounds.' },
  { id: 'gens', label: 'Generations per Run', type: 'number', cliFlag: '--gens', defaultValue: 10, description: 'Generations per pipeline run.' },
  { id: 'base_config', label: 'Base Config (TOML/YAML)', type: 'text', cliFlag: '--base_config', defaultValue: '', description: 'Path to a base config file.' },
  { id: 'data_dir', label: 'Data Directory', type: 'text', cliFlag: '--data_dir', defaultValue: '', description: 'Override data directory for both evolution and backtest.' },
  { id: 'bt_top', label: 'Backtest Top N', type: 'number', cliFlag: '--bt_top', defaultValue: 10, description: 'Top N alphas to backtest per run.' },
  { id: 'no_clean', label: 'Do Not Clean Runs Dir', type: 'boolean', cliFlag: '--no-clean', defaultValue: false, description: 'Keep existing pipeline_runs_cs artefacts.' },
  { id: 'dry_run', label: 'Dry Run', type: 'boolean', cliFlag: '--dry-run', defaultValue: false, description: 'Validate wiring without heavy compute.' },

  // Sweep options
  { id: 'sweep_capacity', label: 'Sweep Capacity (Grid)', type: 'boolean', cliFlag: '--sweep-capacity', defaultValue: false, description: 'Grid sweep fresh_rate/pop_size/hof_per_gen.' },
  { id: 'seeds', label: 'Seeds (comma-separated)', type: 'text', cliFlag: '--seeds', defaultValue: '', description: 'Optional comma-separated seeds for sweep.' },
  { id: 'out_summary', label: 'Sweep Summary CSV', type: 'text', cliFlag: '--out-summary', defaultValue: '', description: 'Optional path to write sweep summary CSV.' },

  // Common passthrough knobs (forwarded to run_pipeline)
  { id: 'selection_metric', label: 'Selection Metric', type: 'select', cliFlag: '--selection_metric', defaultValue: 'auto', options: [
    { value: 'ramped', label: 'ramped' },
    { value: 'fixed', label: 'fixed' },
    { value: 'ic', label: 'ic' },
    { value: 'auto', label: 'auto' },
    { value: 'phased', label: 'phased' },
  ], description: 'Selection criterion during evolution.' },
  { id: 'ramp_fraction', label: 'Ramp Fraction', type: 'number', cliFlag: '--ramp_fraction', defaultValue: 0.33, step: 0.01 },
  { id: 'ramp_min_gens', label: 'Ramp Min Gens', type: 'number', cliFlag: '--ramp_min_gens', defaultValue: 5 },
  { id: 'novelty_boost_w', label: 'Novelty Boost Weight', type: 'number', cliFlag: '--novelty_boost_w', defaultValue: 0.02, step: 0.01 },
  { id: 'novelty_struct_w', label: 'Structural Novelty Weight', type: 'number', cliFlag: '--novelty_struct_w', defaultValue: 0.0, step: 0.01 },
  { id: 'hof_corr_mode', label: 'HOF Correlation Mode', type: 'select', cliFlag: '--hof_corr_mode', defaultValue: 'flat', options: [
    { value: 'flat', label: 'flat' },
    { value: 'per_bar', label: 'per_bar' },
  ]},
  { id: 'ic_tstat_w', label: 'IC t-stat Weight', type: 'number', cliFlag: '--ic_tstat_w', defaultValue: 0.0, step: 0.1 },
  { id: 'temporal_decay_half_life', label: 'Temporal Decay Half-life', type: 'number', cliFlag: '--temporal_decay_half_life', defaultValue: 0, description: 'Bars half-life for recency weighting (0 disables).' },
  { id: 'rank_softmax_beta_floor', label: 'Rank Softmax Beta Floor', type: 'number', cliFlag: '--rank_softmax_beta_floor', defaultValue: 0.0, step: 0.1 },
  { id: 'rank_softmax_beta_target', label: 'Rank Softmax Beta Target', type: 'number', cliFlag: '--rank_softmax_beta_target', defaultValue: 2.0, step: 0.1 },
  { id: 'corr_penalty_w', label: 'Correlation Penalty Weight', type: 'number', cliFlag: '--corr_penalty_w', defaultValue: 0.35, step: 0.01 },
];

export const ITER_SECTIONS: ParameterSection[] = [
  {
    title: 'Iterative Loop',
    parameters: [
      'iters', 'gens', 'base_config', 'data_dir', 'bt_top', 'no_clean', 'dry_run',
    ].map(id => ITER_PARAM_DEFINITIONS.find(p => p.id === id)!).filter(Boolean),
  },
  {
    title: 'Sweep Options',
    parameters: [
      'sweep_capacity', 'seeds', 'out_summary'
    ].map(id => ITER_PARAM_DEFINITIONS.find(p => p.id === id)!).filter(Boolean),
  },
  {
    title: 'Passthrough (Evolution Knobs)',
    parameters: [
      'selection_metric', 'ramp_fraction', 'ramp_min_gens', 'novelty_boost_w', 'novelty_struct_w', 'hof_corr_mode',
      'ic_tstat_w', 'temporal_decay_half_life', 'rank_softmax_beta_floor', 'rank_softmax_beta_target', 'corr_penalty_w'
    ].map(id => ITER_PARAM_DEFINITIONS.find(p => p.id === id)!).filter(Boolean),
  },
];

export const ITER_INITIALS: Record<string, string | number | boolean> = Object.fromEntries(
  ITER_PARAM_DEFINITIONS.map(def => [def.id, def.defaultValue])
);

