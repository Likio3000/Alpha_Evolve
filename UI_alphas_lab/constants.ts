
import { SimulationParams } from './types';

export const OP_REGISTRY_KEYS: string[] = [
    "add", "sub", "mul", "div", "tanh", "sign", "neg", "abs", "log", "sqrt",
    "power", "min_val", "max_val", "cs_mean", "cs_std", "cs_rank",
    "cs_demean", "vec_add_scalar", "vec_mul_scalar", "vec_div_scalar",
    "matmul_mv", "matmul_mm", "transpose", "get_feature_vector",
    "get_stock_vector", "assign_vector", "assign_scalar", "assign_matrix"
];

export const CROSS_SECTIONAL_FEATURE_VECTOR_NAMES_CONST: string[] = [
    "opens_t", "highs_t", "lows_t", "closes_t", "ranges_t",
    "ma5_t", "ma10_t", "ma20_t", "ma30_t",
    "vol5_t", "vol10_t", "vol20_t", "vol30_t"
];

export const SCALAR_FEATURE_NAMES_CONST: string[] = ["const_1", "const_neg_1"];
export const FINAL_PREDICTION_VECTOR_NAME_CONST = "s1_predictions_vector";

export const PARAMETER_DESCRIPTIONS: { [key in keyof SimulationParams]?: string } & { positional_generations?: string } = {
  positional_generations: "Number of evolutionary iterations the algorithm will run. This is a positional argument for run_pipeline.py.",
  generations: "Number of evolutionary iterations the algorithm will run.",
  seed: "Random Number Generator seed (used by Python's random module, NumPy) for ensuring reproducibility of results in both evolution and back-testing.",
  pop_size: "Size of the population (number of AlphaPrograms) in each generation during evolution.",
  tournament_k: "In tournament selection for breeding, K individuals are randomly chosen, and the fittest among them is selected as a parent.",
  p_mut: "Probability (0.0 to 1.0) that a child AlphaProgram will undergo mutation after crossover or cloning.",
  p_cross: "Probability (0.0 to 1.0) that two selected parents will undergo crossover to produce a child. If not, the fitter parent is cloned.",
  elite_keep: "Number of the top unique AlphaPrograms from the current generation that are directly copied to the next generation (elitism).",
  max_ops: "Maximum number of operations (instructions) an AlphaProgram can contain. This acts as a hard cap for parsimony.",
  parsimony_penalty: "A penalty applied to an AlphaProgram's fitness score to discourage overly complex programs. Calculated as: penalty_value * (program_size / max_ops).",
  corr_penalty_w: "Weight of the penalty applied if a new AlphaProgram's prediction time series is highly correlated (above corr_cutoff) with any program already in the Hall-of-Fame.",
  corr_cutoff: "Correlation threshold (0.0 to 1.0) above which two AlphaPrograms are considered too similar, triggering the corr_penalty_w.",
  hof_size: "Hall-of-Fame size. The number of best unique programs from the evolution process that are saved (pickled) to disk.",
  max_lookback_data_option: "Strategy for aligning historical data for multiple symbols: 'common_1200' (align on last 1200 common bars), 'specific_long_10k' (use symbols with at least 10k bars, then align), 'full_overlap' (align on maximum possible common history).",
  min_common_points: "Minimum number of common historical data points required for symbols after alignment. Acts as a guard against too-short time series.",
  data_dir: "Directory path where the input OHLCV CSV files for each symbol are located.",
  eval_lag: "Lag (in bars) for evaluation. Predict at time 't', and the alpha is graded against the forward return observed at 't + eval_lag'. This same lag is typically used in the back-tester.",
  scale: "Method to scale raw trading signals cross-sectionally per bar before calculating positions or IC: 'zscore' (normalize to mean 0, std 1), 'rank' (convert to ranks in [-1, +1]), 'sign' (take the sign of the signal).",
  top_to_backtest: "Number of the best AlphaPrograms from the evolved Hall-of-Fame that will be subsequently run through the back-tester for detailed performance simulation.",
  fee: "Round-trip commission fee in basis points (bps) for back-testing (e.g., 1.0 means 0.01%). Applied on position changes.",
  hold: "Holding period in bars for the back-tester. A value of 1 means positions are re-evaluated and potentially re-balanced at each bar.",
  fresh_rate: "Probability (0.0 to 1.0) that a slot in the new population is filled by a completely random program, injecting novelty. 0 disables it."
};
