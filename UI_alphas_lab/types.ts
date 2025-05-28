
export interface SimulationParams {
  generations: number;
  seed: number;
  pop_size: number;
  max_lookback_data_option: 'common_1200' | 'specific_long_10k' | 'full_overlap';
  min_common_points: number;
  eval_lag: number;
  top_to_backtest: number;
  fee: number; // bps
  hold: number; // bars
  scale: 'zscore' | 'rank' | 'sign';
  fresh_rate: number; // Probability (0.0 to 1.0) for novelty injection
  // Less critical for simulation prompts but good to have
  data_dir: string;
  tournament_k: number;
  p_mut: number;
  p_cross: number;
  elite_keep: number;
  max_ops: number;
  parsimony_penalty: number;
  corr_penalty_w: number;
  corr_cutoff: number;
  hof_size: number;
}

export interface BacktestMetrics { // Remains useful for individual alpha metrics if parsed
  sharpeRatio: number;
  annualizedReturnPercent: number;
  maxDrawdownPercent: number;
}

export interface ActualRunUserInput {
  consoleOutput: string;
  personalOpinion: string;
}

export interface ParameterSuggestion {
  suggestedParams: SimulationParams;
  justification: string;
}

export interface PythonFile {
  name: string;
  path: string;
  content: string;
}

export enum ActiveView {
  Code = 'Code',
  ExperimentLoop = 'ExperimentLoop', // Renamed from Simulate
}

export interface AnalyzedIteration {
  id: string; // Unique ID for React keys
  timestamp: Date;
  paramsUsed: SimulationParams;        // Parameters that were analyzed for this iteration
  userInput: ActualRunUserInput;      // User's input related to paramsUsed
  geminiResponse: ParameterSuggestion; // Gemini's suggestion AFTER analyzing paramsUsed and userInput
}
