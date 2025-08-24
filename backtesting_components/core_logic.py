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
    elif method == "madz" or method == "mad":
        med = np.nanmedian(clean_signal_vector)
        mad = np.nanmedian(np.abs(clean_signal_vector - med))
        scale = 1.4826 * mad
        if scale < 1e-9:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (clean_signal_vector - med) / scale
    elif method == "winsor":
        p = 0.02
        lo = np.nanquantile(clean_signal_vector, p)
        hi = np.nanquantile(clean_signal_vector, 1.0 - p)
        w = np.clip(clean_signal_vector, lo, hi)
        mu = np.nanmean(w)
        sd = np.nanstd(w)
        if sd < 1e-9:
            scaled = np.zeros_like(clean_signal_vector)
        else:
            scaled = (w - mu) / sd
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
    long_short_n: int,
    initial_state_vars_config: Dict[str, str], # e.g. {"prev_s1_vec": "vector"}
    scalar_feature_names: List[str], # Pass these from calling script
    cross_sectional_feature_vector_names: List[str], # Pass these
    debug_prints: bool = False, # For optional debug prints
    annualization_factor: float = (365 * 6), # Default for 4H bars, 365 days (crypto)
    stop_loss_pct: float = 0.0,
    # Optional risk controls
    sector_neutralize_positions: bool = False,
    volatility_target: float = 0.0,
    volatility_lookback: int = 30,
    max_leverage: float = 2.0,
    min_leverage: float = 0.25,
    dd_limit: float = 0.0,
    dd_reduction: float = 0.5,
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
        if long_short_n > 0:
            k = min(long_short_n, n_stocks // 2)
            order = np.argsort(scaled_signal_t)
            long_idx = order[-k:]
            short_idx = order[:k]
            ls_vector = np.zeros_like(scaled_signal_t)
            ls_vector[long_idx] = 1.0
            ls_vector[short_idx] = -1.0
            scaled_signal_t = ls_vector
        # Center cross-section and optionally sector-neutralize
        centered_signal_t = scaled_signal_t - np.mean(scaled_signal_t)
        neutralized_signal_t = centered_signal_t
        if sector_neutralize_positions:
            # Demean within sector buckets
            groups = get_sector_groups(stock_symbols).astype(int)
            for gid in np.unique(groups):
                mask = groups == gid
                if np.any(mask):
                    neutralized_signal_t[mask] -= np.mean(neutralized_signal_t[mask])
        # Final L1 neutralization
        sum_abs_centered_signal = np.sum(np.abs(neutralized_signal_t))
        if sum_abs_centered_signal > 1e-9:
            neutralized_signal_t = neutralized_signal_t / sum_abs_centered_signal
        else:
            neutralized_signal_t = np.zeros_like(neutralized_signal_t)
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
    close_t_matrix = np.zeros_like(signal_matrix)
    next_high_matrix = np.zeros_like(signal_matrix)
    next_low_matrix = np.zeros_like(signal_matrix)
    idx_for_returns = common_time_index[:signal_matrix.shape[0]]
    for i, sym in enumerate(stock_symbols):
        ret_fwd_values = aligned_dfs[sym]["ret_fwd"].loc[idx_for_returns].values
        ret_fwd_matrix[:, i] = np.nan_to_num(ret_fwd_values, nan=0.0, posinf=0.0, neginf=0.0)
        # For simple intrabar stop-loss modeling (lag==1), we need entry close
        # at t and next bar's high/low.
        close_t_vals = aligned_dfs[sym]["close"].loc[idx_for_returns].values
        close_t_matrix[:, i] = np.nan_to_num(close_t_vals, nan=0.0, posinf=0.0, neginf=0.0)
        # Shift high/low by -1 so index t corresponds to the next bar extremes
        try:
            next_high_vals = aligned_dfs[sym]["high"].shift(-1).loc[idx_for_returns].values
            next_low_vals = aligned_dfs[sym]["low"].shift(-1).loc[idx_for_returns].values
        except Exception:
            next_high_vals = np.zeros_like(close_t_vals)
            next_low_vals = np.zeros_like(close_t_vals)
        next_high_matrix[:, i] = np.nan_to_num(next_high_vals, nan=0.0, posinf=0.0, neginf=0.0)
        next_low_matrix[:, i] = np.nan_to_num(next_low_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply optional intrabar stop-loss (per-asset). Supported for lag==1.
    # If enabled and lag!=1, the stop is ignored to avoid lookahead bias.
    ret_used_matrix = ret_fwd_matrix.copy()
    # We'll compute per-bar extra stop costs after exposure scaling.
    stop_mask_matrix = np.zeros_like(ret_fwd_matrix, dtype=bool)
    stop_hits_total = 0
    stop_hit_bars = 0
    if stop_loss_pct and stop_loss_pct > 0.0 and lag == 1:
        s = float(stop_loss_pct)
        # For each time t and asset i, determine if stop is hit during next bar.
        # Long: next_low <= entry*(1-s) → realized return capped at -s
        # Short: next_high >= entry*(1+s) → realized return capped at +s
        entry_price = close_t_matrix
        long_mask = (actual_positions > 0)
        short_mask = (actual_positions < 0)
        long_stop = (next_low_matrix <= entry_price * (1.0 - s)) & long_mask
        short_stop = (next_high_matrix >= entry_price * (1.0 + s)) & short_mask
        # Cap underlying return used
        stop_mask = (long_stop | short_stop)
        stop_mask_matrix = stop_mask
        # Cap underlying return used
        ret_used_matrix = np.where(long_stop, -s, ret_used_matrix)
        ret_used_matrix = np.where(short_stop, +s, ret_used_matrix)
        # Stats (notional and extra costs computed later after scaling)
        stop_hits_total = int(np.sum(stop_mask))
        stop_hit_bars = int(np.sum(np.any(stop_mask, axis=1)))

    # Base (unit-gross) portfolio returns before costs
    base_returns = np.sum(actual_positions * ret_used_matrix, axis=1)

    # Exposure scaling via volatility targeting and drawdown limiter
    T = base_returns.shape[0]
    exposure_mult = np.ones(T, dtype=float)
    fee_rate = (fee_bps * 1e-4)
    eps = 1e-9
    # Forward compute scaled positions, costs, and returns
    scaled_positions = np.zeros_like(actual_positions)
    daily_portfolio_returns_net = np.zeros(T, dtype=float)
    abs_pos_diff_sum = np.zeros(T, dtype=float)
    equity = 1.0
    peak = 1.0
    per_bar_stop_hits = np.zeros(T, dtype=float)
    for t in range(T):
        # Volatility targeting multiplier based on base returns history
        mult = 1.0
        if volatility_target and volatility_target > 0.0:
            start = max(0, t - int(max(1, volatility_lookback)))
            if t - start >= 2:
                realized = np.std(base_returns[start:t], ddof=0)
                if realized > eps:
                    mult = float(volatility_target) / realized
            # Clamp leverage
            mult = float(np.clip(mult, min_leverage, max_leverage))
        # Drawdown limiter (based on net equity up to t-1)
        if dd_limit and dd_limit > 0.0:
            dd = (equity - peak) / (peak + eps)
            if dd < -abs(dd_limit):
                mult *= float(dd_reduction)
        exposure_mult[t] = mult
        # Apply multiplier to positions
        scaled_positions[t, :] = mult * actual_positions[t, :]
        # Transaction costs (position change costs)
        prev = scaled_positions[t - 1, :] if t > 0 else np.zeros(n_stocks)
        pos_diff_t = scaled_positions[t, :] - prev
        abs_pos_diff_sum[t] = np.sum(np.abs(pos_diff_t))
        # Extra stop costs this bar using scaled exposure (intrabar exit)
        extra_stop_cost_t = 0.0
        if stop_loss_pct and stop_loss_pct > 0.0 and lag == 1:
            per_bar_stop_notional = np.sum(np.abs(scaled_positions[t, :]) * stop_mask_matrix[t, :])
            extra_stop_cost_t = fee_rate * per_bar_stop_notional
            per_bar_stop_hits[t] = float(np.sum(stop_mask_matrix[t, :]))
        # Net return this bar
        gross_ret_t = np.dot(scaled_positions[t, :], ret_used_matrix[t, :])
        transaction_costs_t = fee_rate * abs_pos_diff_sum[t] + extra_stop_cost_t
        ret_net_t = gross_ret_t - transaction_costs_t
        daily_portfolio_returns_net[t] = ret_net_t
        # Update equity/peak for next bar drawdown logic
        equity *= (1.0 + ret_net_t)
        if equity > peak:
            peak = equity

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
    peak_curve = np.maximum.accumulate(equity_curve)
    drawdown_series = (equity_curve - peak_curve) / (peak_curve + 1e-9)
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
        "Bars": len(daily_portfolio_returns_net),
        "StopHits": stop_hits_total,
        "StopBars": stop_hit_bars,
        "VolTarget": float(volatility_target),
        "DDLimit": float(dd_limit),
        # Time series for diagnostics
        "EquityCurve": equity_curve,
        "ExposureMult": exposure_mult,
        "Drawdown": drawdown_series,
        "StopHitsPerBar": per_bar_stop_hits,
        "RetNet": daily_portfolio_returns_net,
    }
