from src.algotradebase import AlgoTradeBase
import pandas as pd
import numpy as np


class SMA(AlgoTradeBase):
    """
    algo = SMA({
        'SHORT_WINDOW': days,
        'LONG_WINDOW': days,
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def sma_strategy(short_window, long_window):
        sma1 = short_window
        sma2 = long_window
        sma_signal = []
        signal = 0
        for i in range(len(short_window)):
            if sma1[i] > sma2[i]:
                if signal != 1:
                    signal = 1
                    sma_signal.append(signal)
                else:
                    sma_signal.append(0)
            elif sma2[i] > sma1[i]:
                if signal != -1:
                    signal = -1
                    sma_signal.append(-1)
                else:
                    sma_signal.append(0)
            else:
                sma_signal.append(0)
        return sma_signal

    def strategy(self, df_data, strategy_parameters):
        SHORT_WINDOW = strategy_parameters["SHORT_WINDOW"]
        LONG_WINDOW = strategy_parameters["LONG_WINDOW"]
        short_window = self.sma(df_data["close"], SHORT_WINDOW).to_list()
        long_window = self.sma(df_data["close"], LONG_WINDOW).to_list()
        return self.sma_strategy(short_window, long_window), df_data["close"].to_list()


class SuperTrend(AlgoTradeBase):
    """
    algo = SuperTrend({
        'LOOKBACK_STRATEGY_PARAM': lookback_days,
        'MULTIPLIER': multiplier
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def get_supertrend(high, low, close, lookback, multiplier):
        # ATR
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
        atr = tr.ewm(lookback).mean()
        # H/L AVG AND BASIC UPPER & LOWER BAND
        hl_avg = (high + low) / 2
        upper_band = (hl_avg + multiplier * atr).dropna()
        lower_band = (hl_avg - multiplier * atr).dropna()
        # FINAL UPPER BAND
        final_bands = pd.DataFrame(columns=["upper", "lower"])
        final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
        final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 0] = 0
            else:
                if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (close[i - 1] > final_bands.iloc[i - 1, 0]):
                    final_bands.iloc[i, 0] = upper_band[i]
                else:
                    final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]
        # FINAL LOWER BAND
        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 1] = 0
            else:
                if (lower_band[i] > final_bands.iloc[i - 1, 1]) | (close[i - 1] < final_bands.iloc[i - 1, 1]):
                    final_bands.iloc[i, 1] = lower_band[i]
                else:
                    final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]
        # SUPERTREND
        supertrend = pd.DataFrame(columns=[f"supertrend_{lookback}"])
        supertrend.iloc[:, 0] = [x for x in final_bands["upper"] - final_bands["upper"]]
        for i in range(len(supertrend)):
            if i == 0:
                supertrend.iloc[i, 0] = 0
            elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] < final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] > final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] > final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] < final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        supertrend = supertrend.set_index(upper_band.index)
        supertrend = supertrend.dropna()[1:]
        # ST UPTREND/DOWNTREND
        upt = []
        dt = []
        start_pos = len(close) - len(supertrend)
        close = close.iloc[start_pos:].reset_index(drop=True)
        for i in range(len(supertrend)):
            if close[i] > supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close[i] < supertrend.iloc[i, 0]:
                upt.append(np.nan)
                dt.append(supertrend.iloc[i, 0])
            else:
                upt.append(np.nan)
                dt.append(np.nan)

        st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
        upt.index, dt.index = supertrend.index, supertrend.index
        return st, upt, dt

    @staticmethod
    def st_strategy(prices, st):
        st_signal = [0]
        signal = 0
        for i in range(1, len(st)):
            if st[i - 1] > prices[i - 1] and st[i] < prices[i]:
                if signal != 1:
                    signal = 1
                    st_signal.append(signal)
                else:
                    st_signal.append(0)
            elif st[i - 1] < prices[i - 1] and st[i] > prices[i]:
                if signal != -1:
                    signal = -1
                    st_signal.append(signal)
                else:
                    st_signal.append(0)
            else:
                st_signal.append(0)
        return st_signal

    def strategy(self, df_data, strategy_parameters):
        LOOKBACK_STRATEGY_PARAM = strategy_parameters.get("LOOKBACK_STRATEGY_PARAM", 10)
        MULTIPLIER = strategy_parameters.get("MULTIPLIER", 3)
        df_data["st"], df_data["s_upt"], df_data["st_dt"] = self.get_supertrend(
            df_data["high"], df_data["low"], df_data["close"], LOOKBACK_STRATEGY_PARAM, MULTIPLIER
        )
        df_data = df_data[1:].reset_index(drop=True)
        return self.st_strategy(df_data["close"], df_data["st"]), df_data["close"].to_list()


# EOF
