import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
from ut_bot_adaptation import UTBot # Import the UTBot class

# Set up logging
log_dir = "strategy_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/ut_bot_strategy_{timestamp}.log"

# Function to log messages
def log_message(message):
    """Log message to file and print to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_filename, 'a') as f:
        f.write(log_msg + '\n')

log_message("=== UT BOT TRADING STRATEGY BACKTEST ===")

# Load the Renko data for backtesting
log_message("Loading 7-day Renko data...")
renko_df = pd.read_csv('/Users/georgehove/quanta/stpRNG_7days_renko_0_1.csv')
renko_df['datetime'] = pd.to_datetime(renko_df['datetime'])
log_message(f"Renko data shape: {renko_df.shape}")

# Initialize UTBot with atr_period=1 and sensitivity=1
ut_bot = UTBot(atr_period=1, sensitivity=1)
log_message("Running UTBot to generate signals...")
signals_df = ut_bot.run(renko_df)

# Merge signals with Renko data
renko_with_signals = pd.concat([renko_df, signals_df], axis=1)
log_message("Signals merged with Renko data.")

# Dynamic risk management function (copied from xgboost_strategy_direct.py)
def calculate_dynamic_risk(equity, equity_history, win_streak, loss_streak, recent_outcomes=None):
    """Calculate dynamic risk percentage based on multiple factors"""
    base_risk = 0.04  # Base risk percentage

    # 1. Streak-based adjustment
    if win_streak >= 5:
        streak_factor = 1.3
    elif win_streak >= 3:
        streak_factor = 1.15
    elif loss_streak >= 3:
        streak_factor = 0.7
    elif loss_streak >= 1:
        streak_factor = 0.85
    else:
        streak_factor = 1.0

    # 2. Equity milestone adjustment
    if equity < 1000:
        equity_factor = 1.0
    elif equity < 10000:
        equity_factor = 0.95
    elif equity < 100000:
        equity_factor = 0.9
    elif equity < 1000000:
        equity_factor = 0.85
    else:
        equity_factor = 0.8

    # 3. Recent performance adjustment
    if recent_outcomes and len(recent_outcomes) >= 20:
        win_rate = sum(1 for t in recent_outcomes[-20:] if t > 0) / 20
        if win_rate > 0.9:
            performance_factor = 1.2
        elif win_rate > 0.8:
            performance_factor = 1.1
        elif win_rate < 0.5:
            performance_factor = 0.8
        else:
            performance_factor = 1.0
    else:
        performance_factor = 1.0

    # 4. Drawdown protection
    if len(equity_history) > 1:
        peak = max(equity_history)
        drawdown = (peak - equity) / peak
        if drawdown > 0.2:
            drawdown_factor = 0.6
        elif drawdown > 0.1:
            drawdown_factor = 0.8
        else:
            drawdown_factor = 1.0
    else:
        drawdown_factor = 1.0

    # Calculate final risk percentage
    risk = base_risk * streak_factor * equity_factor * performance_factor * drawdown_factor

    # Ensure risk stays within reasonable bounds
    return max(0.01, min(risk, 0.06))  # Cap between 1% and 6%

# Position sizing function (copied from xgboost_strategy_direct.py)
def calculate_position_size(risk_amount, price_risk, max_per_position, max_total_vol, current_open_vol):
    """Calculate optimal position allocation respecting volume constraints"""
    # Calculate ideal lot size
    ideal_lot_size = risk_amount / (price_risk * 10) # Multiplier 10 from original script

    # Available volume within total limit
    available_vol = max_total_vol - current_open_vol

    # Check if we need multiple positions
    if ideal_lot_size <= max_per_position:
        # Single position is sufficient
        lot_size = min(ideal_lot_size, max_per_position, available_vol)
        num_positions = 1 if lot_size > 0 else 0
    else:
        # Need multiple positions
        max_possible_vol = min(ideal_lot_size, available_vol)
        if max_possible_vol <= 0:
            return 0, 0

        num_positions = math.ceil(max_possible_vol / max_per_position)
        lot_size = min(max_per_position, max_possible_vol / num_positions)

    return num_positions, lot_size

# Strategy parameters (copied from xgboost_strategy_direct.py, adjusted for UT Bot)
BRICK_SIZE = 0.1  # Changed from 0.05 to match the Renko data
SPREAD = 0.0
COMMISSION_RATE = 0.15  # 15% commission on profits for zero spread account
MIN_VOL = 0.10
MAX_VOL_PER_POS = 50.0
MAX_TOTAL_VOL = 200.0
# CONFIDENCE_THRESHOLD is not directly applicable for UT Bot signals, as they are direct buy/sell.

def apply_commission(profit):
    """Apply 15% commission on profits only. Losses are not charged commission."""
    if profit > 0:
        commission = profit * COMMISSION_RATE
        net_profit = profit - commission
        return net_profit, commission
    else:
        return profit, 0.0  # No commission on losses

# Account state
equity = 10.0  # Starting with $10 as requested
open_volume = 0
equity_history = [equity]
win_streak = 0
loss_streak = 0
trade_outcomes = []
win_count = 0
trade_count = 0
total_commission = 0.0  # Track total commission paid

# Trading results
trades = []

# Backtest loop
# Start after enough data for UTBot's internal calculations (ATR period is 1, so 1 data point is enough)
# The original Pine Script uses src[1] and xATRTrailingStop[1], implying at least 2 bars for initial calculation.
# Let's start from index 1 to ensure prev_close and prev_xATRTrailingStop are available.
i = 1
while i < len(renko_with_signals) - 10: # Leave some room at the end for trade management
    current_row = renko_with_signals.iloc[i]

    # Check for UT Bot buy/sell signals
    if current_row['buy'] or current_row['sell']:

        # Calculate dynamic risk
        risk_percentage = calculate_dynamic_risk(
            equity,
            equity_history,
            win_streak,
            loss_streak,
            trade_outcomes
        )

        # Calculate risk amount and price risk
        # Price risk is based on SL bricks, which is 2 bricks for initial trade
        price_risk = 2 * BRICK_SIZE + SPREAD
        risk_amount = equity * risk_percentage

        # Calculate position size
        num_positions, lot_size = calculate_position_size(
            risk_amount,
            price_risk,
            MAX_VOL_PER_POS,
            MAX_TOTAL_VOL,
            open_volume
        )

        if num_positions > 0 and lot_size >= MIN_VOL:
            # Execute trade
            entry_time = current_row['datetime']
            entry_price = current_row['close']
            position_type = "LONG" if current_row['buy'] else "SHORT"
            position_lot_size = lot_size
            open_volume += position_lot_size
            trade_count += 1

            log_message(f"Trade #{trade_count} executed - {position_type} at {entry_time}, price: {entry_price}, volume: {position_lot_size:.2f}")

            # Trade management
            tp_bricks = 5  # Take profit at 5 bricks
            sl_bricks = 2  # Stop loss at 2 bricks

            # Initialize trade outcome variables
            profit = 0
            trade_commission = 0.0
            outcome = None
            exit_price = entry_price
            exit_time = entry_time

            # Simulate trade
            # Iterate through subsequent bars to find exit condition
            for j in range(i + 1, min(i + 20, len(renko_with_signals))): # Look up to 20 bars ahead
                move = renko_with_signals.iloc[j]['direction']
                current_price_at_exit_check = renko_with_signals.iloc[j]['close']

                if position_type == "LONG":
                    if move == 'up':
                        tp_bricks -= 1
                        if tp_bricks == 0:
                            gross_profit = (5 * BRICK_SIZE - SPREAD) * 10 * position_lot_size
                            profit, trade_commission = apply_commission(gross_profit)
                            outcome = 'LONG_TP'
                            exit_price = current_price_at_exit_check
                            exit_time = renko_with_signals.iloc[j]['datetime']
                            log_message(f"LONG_TP hit. Gross: ${gross_profit:.2f}, Commission: ${trade_commission:.2f}, Net: ${profit:.2f}")
                            break
                    elif move == 'down':
                        sl_bricks -= 1
                        if sl_bricks == 0:
                            outcome = 'LONG_SL'
                            log_message("LONG_SL hit, preparing for short reversal")

                            # Short reversal
                            reversal_tp_bricks = 2  # Take profit at 2 bricks down
                            reversal_sl_bricks = 5  # Stop loss at 5 bricks up

                            for k in range(j + 1, min(j + 20, len(renko_with_signals))): # Look up to 20 bars ahead for reversal
                                reversal_move = renko_with_signals.iloc[k]['direction']
                                reversal_price_at_exit_check = renko_with_signals.iloc[k]['close']

                                if reversal_move == 'down':
                                    reversal_tp_bricks -= 1
                                    if reversal_tp_bricks == 0:
                                        gross_profit = (2 * BRICK_SIZE - SPREAD) * 10 * position_lot_size
                                        profit, trade_commission = apply_commission(gross_profit)
                                        outcome = 'SHORT_TP_AFTER_LONG_SL'
                                        exit_price = reversal_price_at_exit_check
                                        exit_time = renko_with_signals.iloc[k]['datetime']
                                        log_message(f"SHORT_TP hit after LONG_SL. Gross: ${gross_profit:.2f}, Commission: ${trade_commission:.2f}, Net: ${profit:.2f}")
                                        break
                                elif reversal_move == 'up':
                                    reversal_sl_bricks -= 1
                                    if reversal_sl_bricks == 0:
                                        profit = -(5 * BRICK_SIZE + SPREAD) * 10 * position_lot_size
                                        outcome = 'SHORT_SL_AFTER_LONG_SL'
                                        exit_price = reversal_price_at_exit_check
                                        exit_time = renko_with_signals.iloc[k]['datetime']
                                        log_message(f"SHORT_SL hit after LONG_SL. Loss: ${profit:.2f}")
                                        break
                            
                            if outcome == 'LONG_SL': # If no exit in the short reversal, original SL is the loss
                                profit = -(2 * BRICK_SIZE + SPREAD) * 10 * position_lot_size
                                exit_price = current_price_at_exit_check
                                exit_time = renko_with_signals.iloc[j]['datetime']
                            break

                elif position_type == "SHORT":
                    if move == 'down':
                        tp_bricks -= 1
                        if tp_bricks == 0:
                            gross_profit = (5 * BRICK_SIZE - SPREAD) * 10 * position_lot_size
                            profit, trade_commission = apply_commission(gross_profit)
                            outcome = 'SHORT_TP'
                            exit_price = current_price_at_exit_check
                            exit_time = renko_with_signals.iloc[j]['datetime']
                            log_message(f"SHORT_TP hit. Gross: ${gross_profit:.2f}, Commission: ${trade_commission:.2f}, Net: ${profit:.2f}")
                            break
                    elif move == 'up':
                        sl_bricks -= 1
                        if sl_bricks == 0:
                            outcome = 'SHORT_SL'
                            log_message("SHORT_SL hit, preparing for long reversal")

                            # Long reversal
                            reversal_tp_bricks = 2  # Take profit at 2 bricks up
                            reversal_sl_bricks = 5  # Stop loss at 5 bricks down

                            for k in range(j + 1, min(j + 20, len(renko_with_signals))): # Look up to 20 bars ahead for reversal
                                reversal_move = renko_with_signals.iloc[k]['direction']
                                reversal_price_at_exit_check = renko_with_signals.iloc[k]['close']

                                if reversal_move == 'up':
                                    reversal_tp_bricks -= 1
                                    if reversal_tp_bricks == 0:
                                        gross_profit = (2 * BRICK_SIZE - SPREAD) * 10 * position_lot_size
                                        profit, trade_commission = apply_commission(gross_profit)
                                        outcome = 'LONG_TP_AFTER_SHORT_SL'
                                        exit_price = reversal_price_at_exit_check
                                        exit_time = renko_with_signals.iloc[k]['datetime']
                                        log_message(f"LONG_TP hit after SHORT_SL. Gross: ${gross_profit:.2f}, Commission: ${trade_commission:.2f}, Net: ${profit:.2f}")
                                        break
                                elif reversal_move == 'down':
                                    reversal_sl_bricks -= 1
                                    if reversal_sl_bricks == 0:
                                        profit = -(5 * BRICK_SIZE + SPREAD) * 10 * position_lot_size
                                        outcome = 'LONG_SL_AFTER_SHORT_SL'
                                        exit_price = reversal_price_at_exit_check
                                        exit_time = renko_with_signals.iloc[k]['datetime']
                                        log_message(f"LONG_SL hit after SHORT_SL. Loss: ${profit:.2f}")
                                        break
                            
                            if outcome == 'SHORT_SL': # If no exit in the long reversal, original SL is the loss
                                profit = -(2 * BRICK_SIZE + SPREAD) * 10 * position_lot_size
                                exit_price = current_price_at_exit_check
                                exit_time = renko_with_signals.iloc[j]['datetime']
                            break

            # Time-based exit if no other exit condition met
            if profit == 0:
                # Exit at the close of the 10th bar after entry, or the last available bar if less than 10
                exit_bar_index = min(i + 10, len(renko_with_signals) - 1)
                exit_price = renko_with_signals.iloc[exit_bar_index]['close']
                exit_time = renko_with_signals.iloc[exit_bar_index]['datetime']

                if position_type == "LONG":
                    gross_profit = (exit_price - entry_price) * 10 * position_lot_size
                else:  # SHORT
                    gross_profit = (entry_price - exit_price) * 10 * position_lot_size
                
                profit, trade_commission = apply_commission(gross_profit)
                outcome = 'TIME_EXIT'
                
                if trade_commission > 0:
                    log_message(f"Time exit taken. Gross: ${gross_profit:.2f}, Commission: ${trade_commission:.2f}, Net: ${profit:.2f}")
                else:
                    log_message(f"Time exit taken. Loss: ${profit:.2f} (no commission on losses)")

            # Update account state
            previous_equity = equity
            equity += profit
            total_commission += trade_commission
            equity_history.append(equity)
            trade_outcomes.append(profit)

            # Safety check: Stop trading if balance goes negative
            if equity < 0:
                log_message(f"CRITICAL: Balance went negative (${equity:.2f}). Stopping trading immediately!")
                log_message(f"Trade #{trade_count} caused the negative balance.")
                break

            # Update win/loss streaks
            if profit > 0:
                win_streak += 1
                loss_streak = 0
                win_count += 1
            else:
                win_streak = 0
                loss_streak += 1

            # Log trade completion
            log_message(f"Trade #{trade_count} completed:")
            log_message(f"Entry Time: {entry_time}")
            log_message(f"Exit Time: {exit_time}")
            log_message(f"Outcome: {outcome}")
            log_message(f"Profit: ${profit:.2f}")
            log_message(f"Equity: ${previous_equity:.2f} -> ${equity:.2f}")

            # Record trade
            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'position_type': position_type,
                'volume': round(position_lot_size, 2),
                'exit_time': exit_time,
                'exit_price': exit_price,
                'outcome': outcome,
                'profit': round(profit, 2),
                'commission': round(trade_commission, 2),
                'balance': round(equity, 2),
                'risk_percentage': round(risk_percentage, 4)
            })

            # Update open volume
            open_volume -= position_lot_size

            # Move to the next bar after the exit
            # Ensure we don't go out of bounds
            i = renko_with_signals.index.get_loc(renko_with_signals[renko_with_signals['datetime'] == exit_time].index[0]) + 1
        else:
            i += 1 # No trade executed, move to next bar
    else:
        i += 1 # No signal, move to next bar

# Save trading results
results_df = pd.DataFrame(trades)
if not results_df.empty:
    results_df.to_csv(f"{log_dir}/ut_bot_strategy_results_{timestamp}.csv", index=False)
    log_message(f"Saved strategy results to {log_dir}/ut_bot_strategy_results_{timestamp}.csv")
else:
    log_message("No trades executed during the backtest.")

# Calculate performance metrics
win_rate = (results_df['profit'] > 0).mean() * 100 if len(results_df) > 0 else 0
profit_factor = results_df[results_df['profit'] > 0]['profit'].sum() / abs(results_df[results_df['profit'] < 0]['profit'].sum() + 1e-6) if len(results_df) > 0 else 0
win_percentage = (win_count / trade_count * 100) if trade_count > 0 else 0

# Final report
log_message("\n=== FINAL STRATEGY RESULTS ===")
log_message(f"Trades Executed: {trade_count}")
log_message(f"Final Balance: ${equity:.2f}")
log_message(f"Total Commission Paid: ${total_commission:.2f}")
log_message(f"Commission Rate: {COMMISSION_RATE*100:.1f}% on profits")
log_message(f"Win Rate: {win_rate:.2f}%")
log_message(f"Win Count: {win_count} of {trade_count} trades ({win_percentage:.2f}%)")
log_message(f"Profit Factor: {profit_factor:.2f}")

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_history)
plt.title('Equity Curve')
plt.xlabel('Trade Number')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.savefig(f"{log_dir}/ut_bot_equity_curve_{timestamp}.png")
log_message(f"Saved equity curve to {log_dir}/ut_bot_equity_curve_{timestamp}.png")

log_message("UT Bot strategy backtest completed successfully!")