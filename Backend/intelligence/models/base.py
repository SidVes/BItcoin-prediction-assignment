"""Abstract base class for all BTC price predictors."""
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEST_DAYS = 60
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


# Shared utilities

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAPE, and Directional Accuracy."""
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) > 0 else 0.0

    return {
        "rmse": round(rmse, 2),
        "mape": round(mape, 4),
        "dir_accuracy": round(dir_acc, 2),
    }


def load_metadata() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def save_metadata(model_name: str, last_data_date: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    meta[model_name] = {"last_data_date": last_data_date}
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def needs_training(model_name: str, artifact_path: Path, last_data_date: str) -> bool:
    """Return True if artifact is missing or training data is stale."""
    if not artifact_path.exists():
        return True
    trained_date = load_metadata().get(model_name, {}).get("last_data_date")
    return trained_date != last_data_date


# Abstract base
class BasePredictor(ABC):
    """Common interface for ARIMA, XGBoost, LSTM, Prophet, PatchTST."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train on df (handles train/test split internally). Saves artifact."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate on last TEST_DAYS rows. Returns {rmse, mape, dir_accuracy}."""

    @abstractmethod
    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict next-day BTC close price."""

    def run_pipeline(self, df: pd.DataFrame, force_retrain: bool = False) -> Dict[str, Any]:
        """Train (if needed / forced) -> evaluate -> predict. Returns full ModelResult dict."""
        last_data_date = str(df.index[-1].date())
        retrained = self._maybe_train(df, last_data_date, force_retrain)

        t0 = time.perf_counter()
        metrics = self.evaluate(df)
        logger.info("[%s] Evaluation done in %.1fs — RMSE=%.2f, MAPE=%.4f%%, DirAcc=%.1f%%",
                    self.name, time.perf_counter() - t0,
                    metrics["rmse"], metrics["mape"], metrics["dir_accuracy"])

        if metrics["mape"] > 10:
            logger.warning("[%s] High MAPE=%.4f%% — model may be poorly calibrated.", self.name, metrics["mape"])

        prediction = self.predict_next(df)
        logger.info("[%s] Next-day prediction: $%.2f (%s, %+.2f%%)",
                    self.name, prediction["predicted_price"],
                    prediction["direction"], prediction["pct_change"])

        return {
            **prediction,
            **metrics,
            "trained": retrained,
        }

    def _maybe_train(self, df: pd.DataFrame, last_data_date: str, force_retrain: bool = False) -> bool:
        """Train and save when forced or when cache is stale. Returns True if training happened."""
        artifact = self._artifact_path()
        if force_retrain or needs_training(self.name, artifact, last_data_date):
            reason = "force_retrain" if force_retrain else f"stale/missing (last_data_date={last_data_date})"
            logger.info("[%s] Training — reason: %s", self.name, reason)
            t0 = time.perf_counter()
            self.train(df)
            logger.info("[%s] Training complete in %.1fs", self.name, time.perf_counter() - t0)
            save_metadata(self.name, last_data_date)
            logger.debug("[%s] Metadata saved (last_data_date=%s)", self.name, last_data_date)
            return True
        logger.info("[%s] Cache up-to-date — loading artifact.", self.name)
        self._load()
        return False

    @abstractmethod
    def _artifact_path(self) -> Path:
        """Primary artifact file used to check freshness."""

    @abstractmethod
    def _load(self) -> None:
        """Load model from artifact (called when training is skipped)."""

    # Helpers

    def _build_result(
        self,
        predicted_price: float,
        current_price: float,
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        direction = "UP" if predicted_price > current_price else "DOWN"
        pct_change = (predicted_price - current_price) / current_price * 100
        return {
            "model": self.name,
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "direction": direction,
            "pct_change": round(pct_change, 4),
            "details": details or {},
        }
