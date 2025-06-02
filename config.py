# config.py
from dataclasses import dataclass

@dataclass
class EvoConfig:
    # Parameters with defaults from consultant's Option 1 example
    pop_size: int = 96
    fresh_rate: float = 0.12  # a.k.a. novelty injection rate
    p_mut: float = 0.55
    max_ops: int = 24         # Used for AlphaProgram generation & parsimony in evaluation
    parsimony_penalty: float = 0.02 # For evaluation config
    tournament_k: int = 3
    corr_penalty_w: float = 0.35  # For HOF config
    corr_cutoff: float = 0.15   # For HOF config

    # Other parameters, using sensible defaults (often from existing scripts)
    # 'generations' is positional in run_pipeline.py, EvoConfig field will be populated by parser.
    # Argparse will require 'generations' if it's positional without a default in the parser.
    # However, giving a default here for programmatic use or future flexibility.
    generations: int = 10 
    seed: int = 42
    data_dir: str = "./data"
    # Choices for max_lookback_data_option: 'common_1200', 'specific_long_10k', 'full_overlap'
    max_lookback_data_option: str = 'common_1200' 
    min_common_points: int = 1200
    quiet: bool = False
    
    p_cross: float = 0.6
    elite_keep: int = 4
    hof_size: int = 20        # Num top programs to save from evolution
    eval_lag: int = 1         # Lag for IC calculation (evolution) & signal lag (backtest)
    # Choices for scale: "zscore", "rank", "sign"
    scale: str = "zscore"     

    # Evaluation specific constants (from evolve_alphas.py defaults)
    xs_flat_guard: float = 5e-2
    t_flat_guard: float = 5e-2
    early_abort_bars: int = 20
    early_abort_xs: float = 5e-2
    early_abort_t: float = 5e-2
    keep_dupes_in_hof: bool = False

    # Backtesting specific arguments (from run_pipeline.py)
    top_to_backtest: int = 10 # Num best programs from evolution to backtest
    fee: float = 1.0          # Round-trip commission in bps for backtest
    hold: int = 1             # Holding period in bars for backtest