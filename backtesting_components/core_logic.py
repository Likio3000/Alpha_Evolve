from __future__ import annotations
import numpy as np
import pandas as pd  # For DataFrame rolling in hold period
from typing import TYPE_CHECKING, Dict, List, Any, OrderedDict as OrderedDictType

from evolution_components.data_handling import get_sector_groups, get_features_at_time

if TYPE_CHECKING:
    from alpha_framework import AlphaProgram # Use the actual class from the framework

# Constants that might be needed from alpha_framework if not passed explicitly
# For now, assume SCALAR_FEATURE_NAMES and CROSS_SECTIONAL_FEATURE_VECTOR_NAMES are passed or globally available
# in the calling context (e.g. backtest_evolved_alphas.py)
# For cleaner design, these should ideally be passed to backtest_cross_sectional_alpha if they can vary.

# Helper function to scale signals
def _scale_signal_cross_sectionally(raw_signal_vector: np.ndarray, method: str) -> np.ndarray:
    if raw_signal_vector.size == 0:
        return raw_signal_vector
    
    clean_signal_vector = np.nan_to_num(raw_signal_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "sign":
        scaled = np.sign(clean_signal_vector)
    elif method == "rank":
        if clean_signal_vector.size <= 1:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            temp = clean_signal_vector.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(clean_signal_vector))
            scaled = (ranks / (len(clean_signal_vector) - 1 + 1e-9)) * 2.0 - 1.0
    else: # Default: z-score
        mu = np.nanmean(clean_signal_vector) 
        sd = np.nanstd(clean_signal_vector)
        if sd < 1e-9 :
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - mu) / sd
    
    return np.clip(scaled, -1, 1)

# Helper function for max drawdown
def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-9)

    if drawdown.size == 0 or not np.any(drawdown):
        return 0.0

    # Drawdowns are negative percentages.  Return the magnitude as a positive
    # number so callers don't have to negate the value.
    return float(-np.min(drawdown))

# Main backtesting function
def backtest_cross_sectional_alpha(
    prog: AlphaProgram,
    aligned_dfs: OrderedDictType[str, pd.DataFrame],
    common_time_index: pd.DatetimeIndex,
    stock_symbols: List[str],
    n_stocks: int,
    fee_bps: float,
    lag: int, 
    hold: int, 
    scale_method: str,
    initial_state_vars_config: Dict[str, str], # e.g. {"prev_s1_vec": "vector"}
    scalar_feature_names: List[str], # Pass these from calling script
    cross_sectional_feature_vector_names: List[str], # Pass these
    debug_prints: bool = False, # For optional debug prints
    annualization_factor: float = (365 * 6) # Default for 4H bars, 365 days (crypto)
) -> Dict[str, Any]:
    
    program_state: Dict[str, Any] = prog.new_state()
    for s_name, s_type in initial_state_vars_config.items():
        if s_name not in program_state:
            if s_type == "vector":
                program_state[s_name] = np.zeros(n_stocks)
            elif s_type == "matrix":
                program_state[s_name] = np.zeros((n_stocks, n_stocks))
            else:
                program_state[s_name] = 0.0

    raw_signals_over_time: List[np.ndarray] = []

    # sector ids are constant across time; compute once for efficiency
    sector_groups_vec = get_sector_groups(stock_symbols).astype(float)
    if np.all(sector_groups_vec == 0):
        raise RuntimeError("sector_id_vector is all zeros – check data handling")

    # For reproducibility of AlphaProgram.eval if it has stochastic elements (unlikely for these ops)
    # np.random.seed(current_seed) # Seed is usually handled at a higher level (main script)

    for t_idx, timestamp in enumerate(common_time_index):
        if t_idx == len(common_time_index) - 1:
            break

        features_at_t = get_features_at_time(
            timestamp, aligned_dfs, stock_symbols, sector_groups_vec
        )
        
        try:
            signal_vector_t = prog.eval(features_at_t, program_state, n_stocks)
            signal_vector_t = np.nan_to_num(signal_vector_t, nan=0.0, posinf=0.0, neginf=0.0) 
            raw_signals_over_time.append(signal_vector_t)
        except Exception:
            raw_signals_over_time.append(np.zeros(n_stocks))
    
    if not raw_signals_over_time:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": 0, "Error": "No signals generated"}

    signal_matrix = np.array(raw_signals_over_time)

    if debug_prints:
        print(f"Debug (core_logic): Raw signal_matrix σ_cross_sectional (first 5): {signal_matrix.std(axis=1)[:5]}")
        print(">>> signal_matrix[:5,:5] std:", np.std(signal_matrix[:5,:], axis=1, ddof=0))
        print(">>> first few rows of signal_matrix:", signal_matrix[:3,:4])

    target_positions_matrix = np.zeros_like(signal_matrix)
    for t in range(signal_matrix.shape[0]):
        scaled_signal_t = _scale_signal_cross_sectionally(signal_matrix[t, :], scale_method)
        mean_signal_t = np.mean(scaled_signal_t)
        centered_signal_t = scaled_signal_t - mean_signal_t
        sum_abs_centered_signal = np.sum(np.abs(centered_signal_t))
        if sum_abs_centered_signal > 1e-9:
            neutralized_signal_t = centered_signal_t / sum_abs_centered_signal
        else:
            neutralized_signal_t = np.zeros_like(centered_signal_t)
        target_positions_matrix[t, :] = neutralized_signal_t

    if debug_prints:
        print(f"Debug (core_logic): Target_positions_matrix σ_cross_sectional (first 5): {target_positions_matrix.std(axis=1)[:5]}")

    if hold > 1:
        df_target_pos = pd.DataFrame(target_positions_matrix)
        df_held_pos = df_target_pos.rolling(window=hold, min_periods=1).mean()
        held_positions_temp = df_held_pos.values
        for t in range(held_positions_temp.shape[0]): # Re-neutralize after rolling
            current_pos_t = held_positions_temp[t,:]
            mean_pos_t = np.mean(current_pos_t)
            centered_pos_t = current_pos_t - mean_pos_t
            sum_abs_centered_pos = np.sum(np.abs(centered_pos_t))
            target_positions_matrix[t,:] = (centered_pos_t / sum_abs_centered_pos) if sum_abs_centered_pos > 1e-9 else np.zeros_like(centered_pos_t)
    
    actual_positions = np.zeros_like(target_positions_matrix)
    if lag > 0 and target_positions_matrix.shape[0] > lag:
        actual_positions[lag:, :] = target_positions_matrix[:-lag, :]
    elif lag == 0:
        actual_positions = target_positions_matrix
    
    ret_fwd_matrix = np.zeros_like(signal_matrix)
    idx_for_returns = common_time_index[:signal_matrix.shape[0]]
    for i, sym in enumerate(stock_symbols):
        ret_fwd_values = aligned_dfs[sym]["ret_fwd"].loc[idx_for_returns].values
        ret_fwd_matrix[:, i] = np.nan_to_num(ret_fwd_values, nan=0.0, posinf=0.0, neginf=0.0)

    daily_portfolio_returns = np.sum(actual_positions * ret_fwd_matrix, axis=1)
    
    pos_diff = np.diff(actual_positions, axis=0, prepend=np.zeros((1, n_stocks)))
    abs_pos_diff_sum = np.sum(np.abs(pos_diff), axis=1)
    transaction_costs = (fee_bps * 1e-4) * abs_pos_diff_sum
    daily_portfolio_returns_net = daily_portfolio_returns - transaction_costs

    if not np.any(actual_positions) or not np.any(daily_portfolio_returns_net):
        return {
            "Sharpe": 0.0,
            "AnnReturn": 0.0,
            "AnnVol": 0.0,
            "MaxDD": 0.0,
            "Turnover": 0.0,
            "Bars": len(daily_portfolio_returns_net),
            "Error": "No trades executed",
        }
    
    if debug_prints and len(daily_portfolio_returns_net) > 0:
        mean_ret_calc = np.mean(daily_portfolio_returns_net)
        std_ret_calc = np.std(daily_portfolio_returns_net, ddof=0)
        print(f"DEBUG (core_logic): PnL mean {mean_ret_calc:.6e} std {std_ret_calc:.6e}")

    if len(daily_portfolio_returns_net) < 2:
        return {"Sharpe": 0.0, "AnnReturn": 0.0, "AnnVol": 0.0, "MaxDD": 0.0, "Turnover": 0.0, "Bars": len(daily_portfolio_returns_net)}

    equity_curve = np.cumprod(1 + daily_portfolio_returns_net)
    mean_ret = np.mean(daily_portfolio_returns_net)
    std_ret = np.std(daily_portfolio_returns_net, ddof=0)

    sharpe_ratio = (mean_ret / (std_ret + 1e-9)) * np.sqrt(annualization_factor)
    total_return_val = equity_curve[-1] - 1.0
    num_years = len(daily_portfolio_returns_net) / annualization_factor
    annualized_return = ((1.0 + total_return_val) ** (1.0 / num_years)) - 1.0 if num_years > 0 else 0.0
    annualized_volatility = std_ret * np.sqrt(annualization_factor)
    max_dd = _max_drawdown(equity_curve)
    avg_daily_turnover_fraction = np.mean(abs_pos_diff_sum) / 2.0

    return {
        "Sharpe": sharpe_ratio,
        "AnnReturn": annualized_return,
        "AnnVol": annualized_volatility,
        "MaxDD": max_dd,
        "Turnover": avg_daily_turnover_fraction,
        "Bars": len(daily_portfolio_returns_net)
    }
