"""XGBoost predictor — train, evaluate, predict via LangGraph pipeline."""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BasePredictor, ARTIFACTS_DIR, TEST_DAYS, compute_metrics
from ..features.engineering import engineer_features, FEATURE_COLS

logger = logging.getLogger(__name__)

HP_JSON_PATH = Path(__file__).parents[2] / "experiment" / "best_hyperparameters.json"
ARTIFACT_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"


def _load_params() -> dict:
    with open(HP_JSON_PATH) as f:
        params = json.load(f)["XGBoost"].copy()
    params.update({"random_state": 42, "n_jobs": -1})
    return params


class XGBoostPredictor(BasePredictor):

    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "XGBoost"

    def _artifact_path(self) -> Path:
        return ARTIFACT_PATH

    def train(self, df: pd.DataFrame) -> None:
        import xgboost as xgb

        params = _load_params()
        featured = engineer_features(df)
        featured["target"] = featured["Close"].shift(-1)
        featured = featured.dropna(subset=FEATURE_COLS + ["target"])

        # Time-aware split
        X_train = featured[FEATURE_COLS].iloc[:-TEST_DAYS]
        y_train = featured["target"].iloc[:-TEST_DAYS]

        logger.info("[%s] Training on %d rows …", self.name, len(X_train))
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X_train, y_train, verbose=False)

        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("[%s] Artifact saved -> %s", self.name, ARTIFACT_PATH)

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.debug("[%s] Loading artifact from %s", self.name, ARTIFACT_PATH)
        with open(ARTIFACT_PATH, "rb") as f:
            self._model = pickle.load(f)
        logger.info("[%s] Artifact loaded.", self.name)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        featured = engineer_features(df).dropna(subset=FEATURE_COLS)
        X_test   = featured[FEATURE_COLS].iloc[-TEST_DAYS:]
        y_test   = featured["Close"].iloc[-TEST_DAYS:].shift(-1).dropna()
        X_test   = X_test.iloc[: len(y_test)]

        preds = self._model.predict(X_test)
        return compute_metrics(y_test.values, preds)

    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        featured = engineer_features(df).dropna(subset=FEATURE_COLS)
        last_row = featured[FEATURE_COLS].iloc[[-1]]

        predicted_price = float(self._model.predict(last_row)[0])
        current_price = float(df["Close"].iloc[-1])

        importances = dict(zip(FEATURE_COLS, self._model.feature_importances_.tolist()))
        top3 = sorted(importances, key=importances.get, reverse=True)[:3]
        logger.debug("[%s] Top-3 features: %s", self.name, top3)

        return self._build_result(
            predicted_price, current_price,
            details={"top_features": top3},
        )
