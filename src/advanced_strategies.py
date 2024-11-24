from src.algotradebase import AlgoTradeBase
import pandas as pd


class WilliansppRMACD(AlgoTradeBase):
    """
    algo = WilliansppRMACD({
        'LOOKBACK_STRATEGY_PARAM': lookback_days,
        slow, fast, smooth = 26, 12, 9 (optional)
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def implement_wr_macd(wr, macd, macd_signal):
        wr_macd_signal = [0]
        signal = 0
        for i in range(1, len(wr)):
            if wr[i - 1] > -50 and wr[i] < -50 and macd[i] > macd_signal[i]:
                if signal != 1:
                    signal = 1
                    wr_macd_signal.append(signal)
                else:
                    wr_macd_signal.append(0)
            elif wr[i - 1] < -50 and wr[i] > -50 and macd[i] < macd_signal[i]:
                if signal != -1:
                    signal = -1
                    wr_macd_signal.append(signal)
                else:
                    wr_macd_signal.append(0)
            else:
                wr_macd_signal.append(0)
        return wr_macd_signal

    @staticmethod
    def get_wr(high, low, close, lookback):
        highh = high.rolling(lookback).max()
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr

    def strategy(self, df_data, strategy_parameters):
        LOOKBACK_STRATEGY_PARAM = strategy_parameters["LOOKBACK_STRATEGY_PARAM"]
        slow = strategy_parameters.get("SLOW", 26)
        fast = strategy_parameters.get("FAST", 12)
        smooth = strategy_parameters.get("SMOOTH", 9)
        df_data["wr"] = self.get_wr(df_data["high"], df_data["low"], df_data["close"], LOOKBACK_STRATEGY_PARAM)
        macd, macd_signal, hist = self.get_macd(df_data["close"], slow, fast, smooth)
        df_data["macd"] = macd
        df_data["macd_signal"] = macd_signal
        df_data["macd_hist"] = hist
        df_data = df_data.fillna(0)
        close = df_data["close"].to_list()
        wr_signal = self.implement_wr_macd(
            df_data["wr"].to_list(), df_data["macd"].to_list(), df_data["macd_signal"].to_list()
        )
        return wr_signal, close


class ADXRSI(AlgoTradeBase):
    """
    algo = ADXRSI({
        'LOOKBACK_STRATEGY_PARAM': lookback_days,
        'RSI': False|True|None # default: True
        'ADX': False|True|None # default: True
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def get_adx(high, low, close, lookback):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
        atr = tr.rolling(lookback).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha=1 / lookback).mean()
        return plus_di, minus_di, adx_smooth

    @staticmethod
    def get_rsi(close, lookback):
        ret = close.diff()
        up = []
        down = []
        for i in range(len(ret)):
            if ret[i] < 0:
                up.append(0)
                down.append(ret[i])
            else:
                up.append(ret[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com=lookback - 1, adjust=False).mean()
        down_ewm = down_series.ewm(com=lookback - 1, adjust=False).mean()
        rs = up_ewm / down_ewm
        rsi = 100 - (100 / (1 + rs))
        rsi_df = pd.DataFrame(rsi).rename(columns={0: "rsi"}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df[3:]

    @staticmethod
    def adx_strategy(pdi, ndi, adx):
        adx_signal = [0]
        signal = 0
        for i in range(len(adx)):
            if adx[i - 1] < 25 and adx[i] > 25 and pdi[i] > ndi[i]:
                if signal != 1:
                    signal = 1
                    adx_signal.append(signal)
                else:
                    adx_signal.append(0)
            elif adx[i - 1] < 25 and adx[i] > 25 and ndi[i] > pdi[i]:
                if signal != -1:
                    signal = -1
                    adx_signal.append(signal)
                else:
                    adx_signal.append(0)
            else:
                adx_signal.append(0)
        return adx_signal

    @staticmethod
    def rsi_strategy(rsi):
        rsi_signal = [0]
        signal = 0
        for i in range(len(rsi)):
            if rsi[i - 1] > 30 and rsi[i] < 30:
                if signal != 1:
                    signal = 1
                    rsi_signal.append(signal)
                else:
                    rsi_signal.append(0)
            elif rsi[i - 1] < 70 and rsi[i] > 70:
                if signal != -1:
                    signal = -1
                    rsi_signal.append(signal)
                else:
                    rsi_signal.append(0)
            else:
                rsi_signal.append(0)
        return rsi_signal

    @staticmethod
    def adx_rsi_strategy(adx, pdi, ndi, rsi):
        adx_rsi_signal = []
        signal = 0
        for i in range(len(adx)):
            if adx[i] > 35 and pdi[i] < ndi[i] and rsi[i] < 50:
                if signal != 1:
                    signal = 1
                    adx_rsi_signal.append(signal)
                else:
                    adx_rsi_signal.append(0)
            elif adx[i] > 35 and pdi[i] > ndi[i] and rsi[i] > 50:
                if signal != -1:
                    signal = -1
                    adx_rsi_signal.append(signal)
                else:
                    adx_rsi_signal.append(0)
            else:
                adx_rsi_signal.append(0)
        return adx_rsi_signal

    def strategy(self, df_data, strategy_parameters):
        LOOKBACK_STRATEGY_PARAM = strategy_parameters["LOOKBACK_STRATEGY_PARAM"]
        RSI = strategy_parameters.get("RSI", True)
        ADX = strategy_parameters.get("ADX", True)
        if RSI and ADX:
            plus_di, minus_di, adx_smooth = self.get_adx(
                df_data["high"], df_data["low"], df_data["close"], LOOKBACK_STRATEGY_PARAM
            )
            df_data["plus_di"] = pd.DataFrame(plus_di).rename(columns={0: "plus_di"})
            df_data["minus_di"] = pd.DataFrame(minus_di).rename(columns={0: "minus_di"})
            df_data["rsi"] = self.get_rsi(df_data["close"], LOOKBACK_STRATEGY_PARAM)
            df_data["adx"] = pd.DataFrame(adx_smooth).rename(columns={0: "adx"})
            df_data = self.fillna_with(df_data)
            out_signal = self.adx_rsi_strategy(
                df_data["adx"].to_list(),
                df_data["plus_di"].to_list(),
                df_data["minus_di"].to_list(),
                df_data["rsi"].to_list(),
            )
        elif ADX:
            plus_di, minus_di, adx_smooth = self.get_adx(
                df_data["high"], df_data["low"], df_data["close"], LOOKBACK_STRATEGY_PARAM
            )
            df_data["plus_di"] = pd.DataFrame(plus_di).rename(columns={0: "plus_di"})
            df_data["minus_di"] = pd.DataFrame(minus_di).rename(columns={0: "minus_di"})
            df_data["adx"] = pd.DataFrame(adx_smooth).rename(columns={0: "adx"})
            df_data = self.fillna_with(df_data)
            out_signal = self.adx_strategy(
                df_data["plus_di"].to_list(), df_data["minus_di"].to_list(), df_data["adx"].to_list()
            )
        elif RSI:
            df_data["rsi"] = self.get_rsi(df_data["close"], LOOKBACK_STRATEGY_PARAM)
            df_data = self.fillna_with(df_data)
            out_signal = self.rsi_strategy(df_data["rsi"].to_list())
        close = df_data["close"].to_list()
        return out_signal, close


# EOF
