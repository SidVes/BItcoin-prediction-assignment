"""ARIMA predictor — train, evaluate, predict via LangGraph pipeline."""
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
ARTIFACT_PATH = ARTIFACTS_DIR / "arima_model.pkl"


def _load_order() -> tuple:
    with open(HP_JSON_PATH) as f:
        return tuple(json.load(f)["ARIMA"]["order"])


class ARIMAPredictor(BasePredictor):

    def __init__(self):
        self._model = None
        self._close_series = None
        self._order = _load_order()

    @property
    def name(self) -> str:
        p, d, q = self._order
        return f"ARIMA({p},{d},{q})"

    def _artifact_path(self) -> Path:
        return ARTIFACT_PATH


    def train(self, df: pd.DataFrame) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        # Fit on all data except last TEST_DAYS (test holdout)
        train_close = df["Close"].iloc[:-TEST_DAYS]
        self._close_series = train_close
        logger.info("[%s] Fitting on %d rows …", self.name, len(train_close))
        self._model = ARIMA(train_close, order=self._order).fit()

        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "wb") as f:
            pickle.dump({"model_result": self._model, "close_series": train_close}, f)
        logger.info("[%s] Artifact saved -> %s  (AIC=%.2f)", self.name, ARTIFACT_PATH, self._model.aic)



    def _load(self) -> None:
        if self._model is not None:
            return
        logger.debug("[%s] Loading artifact from %s", self.name, ARTIFACT_PATH)
        with open(ARTIFACT_PATH, "rb") as f:
            data = pickle.load(f)
        self._model = data["model_result"]
        self._close_series = data["close_series"]
        logger.info("[%s] Artifact loaded (AIC=%.2f)", self.name, self._model.aic)

    

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Rolling 1-step-ahead forecast on the last TEST_DAYS."""
        from statsmodels.tsa.arima.model import ARIMA

        close = df["Close"]
        train_close = close.iloc[:-TEST_DAYS]
        test_close = close.iloc[-TEST_DAYS:]

        history = list(train_close.values)
        preds = []
        for actual in test_close:
            yhat = ARIMA(history, order=self._order).fit().forecast(steps=1)[0]
            preds.append(float(yhat))
            history.append(actual)

        return compute_metrics(test_close.values, np.array(preds))


    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        predicted_price = float(self._model.forecast(steps=1)[0])
        current_price = float(df["Close"].iloc[-1])
        return self._build_result(
            predicted_price, current_price,
            details={"order": list(self._order), "aic": round(self._model.aic, 2)},
        )
