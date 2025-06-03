import pandas as pd
import numpy as np

class UTBot:
    def __init__(self, atr_period=1, sensitivity=1, use_heikin_ashi=False):
        self.a = sensitivity
        self.c = atr_period
        self.h = use_heikin_ashi
        self.xATRTrailingStop_history = []
        self.pos_history = []
        self.true_range_history = []
        self.atr_history = []

    def _calculate_true_range(self, high, low, close, prev_close):
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        range1 = high - low
        range2 = abs(high - prev_close) if prev_close is not None else 0
        range3 = abs(low - prev_close) if prev_close is not None else 0
        return max(range1, range2, range3)

    def _calculate_atr(self, true_range, current_index):
        if current_index < self.c - 1:
            return np.nan # Not enough data for initial ATR
        elif current_index == self.c - 1:
            # Initial ATR is SMA of first 'c' true ranges
            return np.mean(self.true_range_history[:self.c])
        else:
            # Wilder's smoothing
            prev_atr = self.atr_history[-1]
            return (prev_atr * (self.c - 1) + true_range) / self.c

    def _nz(self, value, default_value):
        return default_value if pd.isna(value) else value

    def _crossover(self, series1_prev, series1_current, series2_prev, series2_current):
        return series1_prev < series2_prev and series1_current > series2_current

    def run(self, df):
        # Ensure DataFrame has 'open', 'high', 'low', 'close'
        # For simplicity, assuming 'close' is the primary source for now.
        # If Heikin Ashi is truly needed, it would require calculating HA candles first.
        # Given h=False, src will always be close.

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        signals = {
            'xATRTrailingStop': [],
            'pos': [],
            'buy': [],
            'sell': [],
            'barbuy': [],
            'barsell': [],
            'xATR': [],
            'ema_src': [] # ema(src, 1) is just src
        }

        for i in range(len(df)):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            prev_close = closes[i-1] if i > 0 else None

            # Calculate True Range
            tr = self._calculate_true_range(current_high, current_low, current_close, prev_close)
            self.true_range_history.append(tr)

            # Calculate ATR
            xATR = self._calculate_atr(tr, i)
            self.atr_history.append(xATR)
            signals['xATR'].append(xATR)

            nLoss = self.a * xATR

            src = current_close # Since h is False

            prev_xATRTrailingStop = self._nz(self.xATRTrailingStop_history[-1] if i > 0 else np.nan, 0)
            src_prev = closes[i-1] if i > 0 else np.nan

            current_xATRTrailingStop = np.nan
            if not pd.isna(src) and not pd.isna(nLoss):
                if src > prev_xATRTrailingStop and src_prev > prev_xATRTrailingStop:
                    current_xATRTrailingStop = max(prev_xATRTrailingStop, src - nLoss)
                elif src < prev_xATRTrailingStop and src_prev < prev_xATRTrailingStop:
                    current_xATRTrailingStop = min(prev_xATRTrailingStop, src + nLoss)
                elif src > prev_xATRTrailingStop:
                    current_xATRTrailingStop = src - nLoss
                else: # src < prev_xATRTrailingStop
                    current_xATRTrailingStop = src + nLoss
            self.xATRTrailingStop_history.append(current_xATRTrailingStop)
            signals['xATRTrailingStop'].append(current_xATRTrailingStop)

            prev_pos = self._nz(self.pos_history[-1] if i > 0 else np.nan, 0)
            current_pos = prev_pos
            if not pd.isna(src) and not pd.isna(src_prev) and not pd.isna(prev_xATRTrailingStop):
                if src_prev < prev_xATRTrailingStop and src > prev_xATRTrailingStop:
                    current_pos = 1
                elif src_prev > prev_xATRTrailingStop and src < prev_xATRTrailingStop:
                    current_pos = -1
            self.pos_history.append(current_pos)
            signals['pos'].append(current_pos)

            ema_src = src # ema(src, 1) is just src
            signals['ema_src'].append(ema_src)

            # Crossover logic
            above = False
            below = False
            if i > 0 and not pd.isna(ema_src) and not pd.isna(current_xATRTrailingStop) and \
               not pd.isna(signals['ema_src'][i-1]) and not pd.isna(signals['xATRTrailingStop'][i-1]):
                above = self._crossover(signals['ema_src'][i-1], ema_src, signals['xATRTrailingStop'][i-1], current_xATRTrailingStop)
                below = self._crossover(signals['xATRTrailingStop'][i-1], current_xATRTrailingStop, signals['ema_src'][i-1], ema_src)

            # Buy/Sell signals
            buy = src > current_xATRTrailingStop and above
            sell = src < current_xATRTrailingStop and below
            signals['buy'].append(buy)
            signals['sell'].append(sell)

            barbuy = src > current_xATRTrailingStop
            barsell = src < current_xATRTrailingStop
            signals['barbuy'].append(barbuy)
            signals['barsell'].append(barsell)

        return pd.DataFrame(signals, index=df.index)

# Example Usage:
# if __name__ == "__main__":
#     # Create dummy data for demonstration
#     data = {
#         'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
#         'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
#         'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
#         'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5]
#     }
#     df = pd.DataFrame(data)

#     # Initialize UTBot with atr_period=1 and sensitivity=1 as requested
#     ut_bot = UTBot(atr_period=1, sensitivity=1)
#     results_df = ut_bot.run(df)

#     print("UT Bot Adaptation Results:")
#     print(results_df)