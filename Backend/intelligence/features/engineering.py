import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parents[2]
DATA_PATH = BASE_DIR / "btc_usd_historical.csv"

FEATURE_COLS = [
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4",
    "close_lag_5", "close_lag_6", "close_lag_7",
    "sma_7",  "rolling_std_7",
    "sma_14", "rolling_std_14",
    "sma_21", "rolling_std_21",
    "ema_7",  "ema_21",
    "return_1d", "return_3d", "return_7d",
    "volatility_7d", "volatility_21d",
    "high_low_ratio", "close_open_ratio",
    "day_of_week",
    "rsi_14",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "atr_14",
    "stoch_k", "stoch_d",
    "obv",
    "volume_sma_7", "volume_ratio",
]


def load_data() -> pd.DataFrame:
    """Load and sort the historical BTC CSV."""
    logger.debug("Loading BTC data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()
    logger.info("Loaded %d rows  (%s -> %s)", len(df), df.index[0].date(), df.index[-1].date())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  
    import ta

    logger.debug("Engineering features for %d rows", len(df))
    df_feat = df.copy()
    close = df_feat["Close"]
    high = df_feat["High"]
    low = df_feat["Low"]
    volume = df_feat["Volume"]

    # Lag features
    for lag in range(1, 8):
        df_feat[f"close_lag_{lag}"] = close.shift(lag)

    # Rolling statistics (interleaved per window, matching notebook column order)
    for window in [7, 14, 21]:
        df_feat[f"sma_{window}"] = close.rolling(window).mean()
        df_feat[f"rolling_std_{window}"] = close.rolling(window).std()

    # EMAs
    for span in [7, 21]:
        df_feat[f"ema_{span}"] = close.ewm(span=span).mean()

    # Returns & Volatility
    df_feat["return_1d"] = close.pct_change(1)
    df_feat["return_3d"] = close.pct_change(3)
    df_feat["return_7d"] = close.pct_change(7)
    df_feat["volatility_7d"] = df_feat["return_1d"].rolling(7).std()
    df_feat["volatility_21d"] = df_feat["return_1d"].rolling(21).std()

    # Price ratios
    df_feat["high_low_ratio"] = high / low
    df_feat["close_open_ratio"] = close / df_feat["Open"]
    df_feat["day_of_week"] = df_feat.index.dayofweek

    # RSI (ta library)
    df_feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD (ta library) — column name: macd_histogram (matches notebook)
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df_feat["macd"] = macd_ind.macd()
    df_feat["macd_signal"] = macd_ind.macd_signal()
    df_feat["macd_histogram"] = macd_ind.macd_diff()   # notebook name

    # Bollinger Bands (ta library)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df_feat["bb_upper"] = bb.bollinger_hband()
    df_feat["bb_lower"] = bb.bollinger_lband()
    df_feat["bb_width"] = bb.bollinger_wband()
    df_feat["bb_pct"] = bb.bollinger_pband()

    # ATR (ta library)
    df_feat["atr_14"] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    # Stochastic Oscillator (ta library)
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3
    )
    df_feat["stoch_k"] = stoch.stoch()
    df_feat["stoch_d"] = stoch.stoch_signal()

    # OBV (ta library)
    df_feat["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Volume indicators
    df_feat["volume_sma_7"] = volume.rolling(7).mean()
    df_feat["volume_ratio"] = volume / df_feat["volume_sma_7"]

    nan_counts = df_feat[FEATURE_COLS].isna().sum()
    total_nans = int(nan_counts.sum())
    if total_nans:
        logger.debug("Feature NaNs after engineering: %d total (expected from window warm-up)", total_nans)

    return df_feat


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) after feature engineering, dropping NaN rows."""
    df_feat = engineer_features(df)
    df_feat["target"] = df_feat["Close"].shift(-1)
    df_feat = df_feat.dropna(subset=FEATURE_COLS + ["target"])
    return df_feat[FEATURE_COLS], df_feat["target"]
