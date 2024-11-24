import pandas as pd
from glob import glob
from functools import cache
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

company_code_list = sorted(["amzn", "ibm", "iwm", "msft", "nvda", "qqq", "spx", "tsla"])


@cache
def load_data(intraday_data, code):
    csv_files = {
        y: next((csv for csv in glob(f"{intraday_data}/*.csv") if code in csv and y in csv), None)
        for y in ["2022", "2023", "2024"]
    }

    dfs = []
    for year, file in csv_files.items():
        if file:
            df = pd.read_csv(file)
            df["year"] = year
            df["company_code"] = code
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["date"])
    return df.drop(columns=["date"])


@cache
def get_data_history(comp_code, intraday_data: str, semester: str = "year", groupby: bool = True):
    df = load_data(intraday_data, comp_code)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_date = df["datetime"].max().date()
    one_year_ago = (max_date - pd.DateOffset(years=1)).date()
    six_months_ago = (max_date - pd.DateOffset(months=6)).date()

    df["date"] = df["datetime"].dt.date
    groupby_cols = ["company_code", "date"]
    values_cols = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if groupby:
        df = df.groupby(groupby_cols).agg(values_cols)
        df.reset_index(inplace=True)

    if semester == "1st":
        past_cutoff, future_cutoff = one_year_ago, six_months_ago
    elif semester == "2nd":
        past_cutoff, future_cutoff = six_months_ago, max_date
    else:  # year
        past_cutoff, future_cutoff = one_year_ago, max_date

    df_out_past = df[df["date"] < past_cutoff].reset_index()
    df_out_future = df[(df["date"] >= past_cutoff) & (df["date"] < future_cutoff)].reset_index()

    return df_out_past, df_out_future


@cache
def get_data_history_ForwardMC(comp_code, intraday_data, period, N_MC_Sims):
    _, df_future = get_data_history(comp_code, intraday_data, period)
    return [df_future] * N_MC_Sims


@cache
def get_data_history_BackMC(comp_code, intraday_data, period, N_MC_Sims):
    _, df_past = get_data_history(comp_code, intraday_data, period, groupby=False)
    # Decompose the time series into trend, seasonal, and residual components
    # using the seasonal_decompose function from the statsmodels library.
    close = df_past.set_index("datetime")["close"]
    decomposition = seasonal_decompose(close, model="additive", period=4)
    # Extract the trend, seasonal, and residual components from the decomposition.
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # Calculate the mean of the residual component.
    residual_mean = residual.mean()
    # Calculate the standard deviation of the residual component.
    residual_std = residual.std()
    # Generate random samples from a normal distribution with the same mean and standard deviation as the residual component.
    random_samples = np.random.normal(residual_mean, residual_std, size=(N_MC_Sims, len(residual)))
    # Add the random samples to the residual component to generate the Monte Carlo simulations.
    clean_series = trend + seasonal

    df_past["close"] = clean_series.to_numpy()
    df_past["close"].fillna(close.reset_index(drop=True))
    df_past["close"] = df_past["close"].fillna(close.reset_index(drop=True))
    for cols in ["high", "low", "open"]:
        df_past[cols] = df_past["close"]

    future_prev_list = []
    for noisy in random_samples:
        df = df_past.copy()
        df["date"] = df["datetime"].dt.date
        df["close"] = df["close"].to_numpy() + noisy
        groupby_cols = ["company_code", "date"]
        values_cols = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df = df.groupby(groupby_cols).agg(values_cols)
        df.reset_index(inplace=True)
        future_prev_list.append(df)

    return future_prev_list


# EOF
