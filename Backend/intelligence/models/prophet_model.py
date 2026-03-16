"""Prophet predictor — train, evaluate, predict via LangGraph pipeline."""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BasePredictor, ARTIFACTS_DIR, TEST_DAYS, compute_metrics

logger = logging.getLogger(__name__)

HP_JSON_PATH = Path(__file__).parents[2] / "experiment" / "best_hyperparameters.json"
ARTIFACT_PATH = ARTIFACTS_DIR / "prophet_model.pkl"


def _load_params() -> dict:
    with open(HP_JSON_PATH) as f:
        return json.load(f)["Prophet"]


class ProphetPredictor(BasePredictor):

    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "Prophet"

    def _artifact_path(self) -> Path:
        return ARTIFACT_PATH

    # Train

    def train(self, df: pd.DataFrame) -> None:
        from prophet import Prophet

        params = _load_params()
        train_df = df.iloc[:-TEST_DAYS][["Close"]].reset_index().rename(
            columns={"Date": "ds", "Close": "y"}
        )

        logger.info("[%s] Training on %d rows …", self.name, len(train_df))
        self._model = Prophet(**params, daily_seasonality=False)
        self._model.fit(train_df)

        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("[%s] Artifact saved -> %s", self.name, ARTIFACT_PATH)

    # Load

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.debug("[%s] Loading artifact from %s", self.name, ARTIFACT_PATH)
        with open(ARTIFACT_PATH, "rb") as f:
            self._model = pickle.load(f)
        logger.info("[%s] Artifact loaded.", self.name)

    # Evaluate

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        test_dates = df.index[-TEST_DAYS:]
        future = pd.DataFrame({"ds": test_dates})
        forecast = self._model.predict(future)

        y_pred = forecast["yhat"].values
        y_true = df["Close"].iloc[-TEST_DAYS:].values

        return compute_metrics(y_true, y_pred)

    # Predict

    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        next_date = df.index[-1] + pd.Timedelta(days=1)
        forecast = self._model.predict(pd.DataFrame({"ds": [next_date]}))

        predicted_price = float(forecast["yhat"].iloc[0])
        current_price = float(df["Close"].iloc[-1])

        return self._build_result(
            predicted_price, current_price,
            details={
                "yhat_lower": round(float(forecast["yhat_lower"].iloc[0]), 2),
                "yhat_upper": round(float(forecast["yhat_upper"].iloc[0]), 2),
                "trend":      round(float(forecast["trend"].iloc[0]), 2),
            },
        )
