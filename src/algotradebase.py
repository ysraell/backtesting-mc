from abc import ABC, abstractmethod
from typing import List, Any, Dict
from numbers import Number
import pandas as pd

Position = int  # 1 or 0
StrategyIndex = int  # -1, 0 or 1
Bugedt = float  # Bugedt >= 0
ClosePrice = float  # close price >= 0.0
Positions = List[Position]
StrategySignal = List[StrategyIndex]
CloseSignal = List[ClosePrice]


class AlgoTradeBase(ABC):
    def __init__(self, strategy_parameters: Dict[str, Any]):
        self.strategy_parameters = strategy_parameters
        self.position_hist: Positions = []
        self.close_hist = []
        self.M = 0
        self.M_diffs = []

    @staticmethod
    def fillna_with(df, val: Number = None):
        for col in df.columns:
            try:
                if val is None:
                    df[col].fillna(df[col].mean())
                else:
                    df[col].fillna(val)
            except TypeError:
                pass
        return df

    @staticmethod
    def sma(price, period):
        return price.rolling(period).mean()

    @staticmethod
    def mad(series):
        """
        usage: df[col].rolling(period).apply(mad)
        """
        return (series - series.mean()).abs().mean()

    @staticmethod
    def get_macd(price, slow, fast, smooth):
        exp1 = price.ewm(span=fast, adjust=False).mean()
        exp2 = price.ewm(span=slow, adjust=False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns={"close": "macd"})
        signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={"macd": "signal"})
        hist = pd.DataFrame(macd["macd"] - signal["signal"]).rename(columns={0: "hist"})
        return macd, signal, hist

    @abstractmethod
    def strategy(self, df_input, strategy_parameters) -> tuple[StrategySignal, CloseSignal]:
        raise NotImplementedError

    def apply_strategy(self, df_input, M_initial: Bugedt = None):
        """
        external input: M: initial budget.
        internal input: self.position_hist, self.close_hist)
        internal output:
            M_diffs: performance by operation, diff between buy and sell.
            M: final bugdet.
        """
        df_data = df_input.copy()
        strategy_signal, self.close_hist = self.strategy(df_data, self.strategy_parameters)
        position = []
        for i in range(len(strategy_signal)):
            if strategy_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)
        for i in range(len(self.close_hist)):
            if strategy_signal[i] == 1:
                position[i] = 1
            elif strategy_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]
        self.position_hist = position
        M = self.M if M_initial is None else M_initial
        Invested = 0
        buy_price = -1
        M_diffs = []
        M_before = 0
        for pos, val in zip(self.position_hist, self.close_hist):
            if pos == 1:
                if buy_price < 0:
                    Invested = M / val
                    M_before = M
                    M = 0
                    buy_price = val
            elif pos == 0:
                if buy_price >= 0:
                    M = val * Invested
                    M_diffs.append(M - M_before)
                    M_before = 0
                    Invested = 0
                    buy_price = -1
        if buy_price >= 0:
            M = val * Invested
            M_diffs.append(M - M_before)
            M_before = 0
            Invested = 0
            buy_price = -1
        self.M = M
        self.M_diffs = M_diffs

    def metrics(self):
        return self.M, self.M_diffs


# EOF
