from src.algotradebase import AlgoTradeBase
import numpy as np


##########################################################################################
#
# Willians %R
#
##########################################################################################


class WilliansppR(AlgoTradeBase):
    """
    algo = WilliansppR({'LOOKBACK_STRATEGY_PARAM': lookback_days})
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def get_wr(high, low, close, lookback):
        highh = high.rolling(lookback).max()
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr

    @staticmethod
    def implement_wr_strategy(wr):
        wr_signal = [0]
        signal = 0
        for i in range(1, len(wr)):
            if wr[i - 1] > -80 and wr[i] < -80:
                if signal != 1:
                    signal = 1
                    wr_signal.append(signal)
                else:
                    wr_signal.append(0)
            elif wr[i - 1] < -20 and wr[i] > -20:
                if signal != -1:
                    signal = -1
                    wr_signal.append(signal)
                else:
                    wr_signal.append(0)
            else:
                wr_signal.append(0)
        return wr_signal

    def strategy(self, df_data, strategy_parameters):
        LOOKBACK_STRATEGY_PARAM = strategy_parameters["LOOKBACK_STRATEGY_PARAM"]
        df_data["wr"] = self.get_wr(df_data["high"], df_data["low"], df_data["close"], LOOKBACK_STRATEGY_PARAM)
        df_data = self.fillna_with(df_data)
        close = df_data["close"].to_list()
        wr = df_data["wr"].to_list()
        wr_signal = self.implement_wr_strategy(wr)
        return wr_signal, close


##########################################################################################
#
# Awesome Oscillator
#
##########################################################################################


class AwesomeOscillator(AlgoTradeBase):

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    def get_ao(self, price, short_period, long_period):
        median = price.rolling(2).median()
        short = self.sma(median, short_period)
        long = self.sma(median, long_period)
        ao = short - long
        return ao

    @staticmethod
    def implement_ao_crossover(ao):
        ao_signal = [0]
        signal = 0
        for i in range(1, len(ao)):
            if ao[i] > 0 and ao[i - 1] < 0:
                if signal != 1:
                    signal = 1
                    ao_signal.append(signal)
                else:
                    ao_signal.append(0)
            elif ao[i] < 0 and ao[i - 1] > 0:
                if signal != -1:
                    signal = -1
                    ao_signal.append(signal)
                else:
                    ao_signal.append(0)
            else:
                ao_signal.append(0)
        return ao_signal

    def strategy(self, df_data, strategy_parameters):
        short_period = strategy_parameters["short_period"]
        long_period = strategy_parameters["long_period"]

        df_data["ao"] = self.get_ao(df_data["close"], short_period, long_period)
        df_data = self.fillna_with(df_data)
        ao_signal = self.implement_ao_crossover(df_data["ao"].to_list())
        return ao_signal, df_data["close"].to_list()


##########################################################################################
#
# Commodity Channel Index
#
##########################################################################################


class CommodityChannelIndex(AlgoTradeBase):

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    def get_cci(self, df, n):
        df_tmp = type(df)()
        df_tmp["pt"] = (df["high"] + df["low"] + df["close"]) / 3
        df_tmp["sma_pt"] = self.sma(df_tmp["pt"], n)
        df_tmp["mad_pt"] = df["close"].rolling(n).apply(self.mad)
        return (df_tmp["pt"] - df_tmp["sma_pt"]) / (0.015 * df_tmp["mad_pt"])

    @staticmethod
    def implement_cci_strategy(prices, cci):
        cci_signal = [0]
        signal = 0
        lower_band = -150
        upper_band = 150
        for i in range(1, len(prices)):
            if cci[i - 1] > lower_band and cci[i] < lower_band:
                if signal != 1:
                    signal = 1
                    cci_signal.append(signal)
                else:
                    cci_signal.append(0)
            elif cci[i - 1] < upper_band and cci[i] > upper_band:
                if signal != -1:
                    signal = -1
                    cci_signal.append(signal)
                else:
                    cci_signal.append(0)
            else:
                cci_signal.append(0)
        return cci_signal

    def strategy(self, df, strategy_parameters):
        LOOKBACK_STRATEGY_PARAM = strategy_parameters["LOOKBACK_STRATEGY_PARAM"]
        df_data = df.copy()
        df_data["cci"] = self.get_cci(df_data, LOOKBACK_STRATEGY_PARAM)
        df_data = self.fillna_with(df_data)
        cci_signal = self.implement_cci_strategy(df_data["close"].to_list(), df_data["cci"].to_list())
        return cci_signal, df_data["close"].to_list()


##########################################################################################
#
# Coppock Curve
#
##########################################################################################


class CoppockCurve(AlgoTradeBase):
    """
    algo = CoppockCurve({
        'shortROC': days,
        'longROC': days,
        'lookbackWMA': shortROC + (shortROC + longROC) // 2,
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def wma(data, lookback):
        weights = np.arange(1, lookback + 1)
        val = data.rolling(lookback)
        wma = val.apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        return wma

    @staticmethod
    def get_roc(close, n):
        difference = close.diff(n)
        nprev_values = close.shift(n)
        roc = (difference / nprev_values) * 100
        return roc

    def get_cc(self, data, roc1_n, roc2_n, wma_lookback):
        longROC = self.get_roc(data, roc2_n)
        shortROC = self.get_roc(data, roc1_n)
        ROC = longROC + shortROC
        cc = self.wma(ROC, wma_lookback)
        return cc

    @staticmethod
    def implement_cc_strategy(prices, cc):
        signal = 1
        cc_signal = [1]
        for i in range(1, 4):
            cc_signal.append(0)
        for i in range(4, len(prices)):
            if cc[i - 4] < 0 and cc[i - 3] < 0 and cc[i - 2] < 0 and cc[i - 1] < 0 and cc[i] > 0:
                if signal != 1:
                    signal = 1
                    cc_signal.append(signal)
                else:
                    cc_signal.append(0)
            elif cc[i - 4] > 0 and cc[i - 3] > 0 and cc[i - 2] > 0 and cc[i - 1] > 0 and cc[i] < 0:
                if signal != -1:
                    signal = -1
                    cc_signal.append(signal)
                else:
                    cc_signal.append(0)
            else:
                cc_signal.append(0)

        return cc_signal

    def strategy(self, df, strategy_parameters):
        shortROC = strategy_parameters["shortROC"]
        longROC = strategy_parameters["longROC"]
        lookbackWMA = strategy_parameters["lookbackWMA"]
        lookbackWMA = lookbackWMA if lookbackWMA is not None else shortROC + (shortROC + longROC) // 2

        df_data = df.copy()
        df_data["cc"] = self.get_cc(df_data["close"], shortROC, longROC, 10)
        df_data = self.fillna_with(df_data)
        return self.implement_cc_strategy(df_data["close"], df_data["cc"]), df_data["close"].to_list()


##########################################################################################
#
# MACD
#
##########################################################################################


class MACD(AlgoTradeBase):
    """
    algo = MACD({
        slow, fast, smooth = 26, 12, 9 (optional)
    })
    """

    def __init__(self, strategy_parameters):
        super().__init__(strategy_parameters)

    @staticmethod
    def implement_macd_strategy(macd, input_signal):
        macd_signal = []
        signal = 0
        for i in range(len(macd)):
            if macd[i] > input_signal[i]:
                if signal != 1:
                    signal = 1
                    macd_signal.append(signal)
                else:
                    macd_signal.append(0)
            elif macd[i] < input_signal[i]:
                if signal != -1:
                    signal = -1
                    macd_signal.append(signal)
                else:
                    macd_signal.append(0)
            else:
                macd_signal.append(0)
        return macd_signal

    def strategy(self, df_data, strategy_parameters):
        slow = strategy_parameters.get("SLOW", 26)
        fast = strategy_parameters.get("FAST", 12)
        smooth = strategy_parameters.get("SMOOTH", 9)
        macd, macd_signal, hist = self.get_macd(df_data["close"], slow, fast, smooth)
        df_data["macd"] = macd
        df_data["macd_signal"] = macd_signal
        df_data["macd_hist"] = hist
        df_data = df_data.fillna(0)
        close = df_data["close"].to_list()
        wr_signal = self.implement_macd_strategy(df_data["macd"].to_list(), df_data["macd_signal"].to_list())
        return wr_signal, close


# EOF
