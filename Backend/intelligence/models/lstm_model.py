"""LSTM predictor — train, evaluate, predict via LangGraph pipeline."""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BasePredictor, ARTIFACTS_DIR, TEST_DAYS, compute_metrics
from ..features.engineering import engineer_features, FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_PATH = ARTIFACTS_DIR / "lstm_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "lstm_scalers.pkl"
SEQ_LEN = 30

# Best hyperparameters (KerasTuner — 20 Bayesian trials)
HP = {
    "lstm_units_1":  64,
    "bidirectional": True,
    "batchnorm_1": True,
    "dropout_1": 0.1,
    "lstm_units_2": 32,
    "batchnorm_2": False,
    "dropout_2": 0.1,
    "third_lstm": False,
    "lstm_units_3": 32,
    "dense_units": 16,
    "dropout_dense": 0.2,
    "learning_rate": 0.001,
    "loss": "huber",
}


def _build_model(n_features: int):
    from tensorflow import keras

    inp = keras.Input(shape=(SEQ_LEN, n_features))
    lstm1 = keras.layers.LSTM(HP["lstm_units_1"], return_sequences=True)
    x = keras.layers.Bidirectional(lstm1)(inp) if HP["bidirectional"] else lstm1(inp)
    if HP["batchnorm_1"]:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(HP["dropout_1"])(x)
    x = keras.layers.LSTM(HP["lstm_units_2"])(x)
    if HP["batchnorm_2"]:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(HP["dropout_2"])(x)
    x = keras.layers.Dense(HP["dense_units"], activation="relu")(x)
    x = keras.layers.Dropout(HP["dropout_dense"])(x)
    out = keras.layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HP["learning_rate"]),
        loss=HP["loss"],
    )
    return model


def _make_sequences(X: np.ndarray, y: np.ndarray):
    Xs, ys = [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i : i + SEQ_LEN])
        ys.append(y[i + SEQ_LEN])
    return np.array(Xs), np.array(ys)


class LSTMPredictor(BasePredictor):

    def __init__(self):
        self._model = None
        self._feat_scaler = None
        self._tgt_scaler = None

    @property
    def name(self) -> str:
        return "LSTM (Bidirectional)"

    def _artifact_path(self) -> Path:
        return MODEL_PATH

    # Train

    def train(self, df: pd.DataFrame) -> None:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler

        gpus = tf.config.list_physical_devices("GPU")
        logger.info("[%s] Training device: %s", self.name, "GPU" if gpus else "CPU")

        featured = engineer_features(df)
        featured["target"] = featured["Close"].shift(-1)
        featured = featured.dropna(subset=FEATURE_COLS + ["target"])

        X_all = featured[FEATURE_COLS].values
        y_all = featured["target"].values.reshape(-1, 1)

        # Time-aware split — scalers fit on train only
        split = len(X_all) - TEST_DAYS
        X_train_raw, X_test_raw = X_all[:split], X_all[split:]
        y_train_raw, y_test_raw = y_all[:split], y_all[split:]

        self._feat_scaler = MinMaxScaler().fit(X_train_raw)
        self._tgt_scaler  = MinMaxScaler().fit(y_train_raw)

        X_train_sc = self._feat_scaler.transform(X_train_raw)
        y_train_sc = self._tgt_scaler.transform(y_train_raw).flatten()

        X_seq, y_seq = _make_sequences(X_train_sc, y_train_sc)

        logger.info("[%s] Training on %d sequences (features=%d, seq_len=%d) …",
                    self.name, len(X_seq), len(FEATURE_COLS), SEQ_LEN)
        self._model = _build_model(n_features=len(FEATURE_COLS))
        self._model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump({"feat": self._feat_scaler, "tgt": self._tgt_scaler}, f)
        logger.info("[%s] Artifact saved -> %s", self.name, MODEL_PATH)

    # Load

    def _load(self) -> None:
        if self._model is not None:
            return
        import tensorflow as tf

        logger.debug("[%s] Loading model from %s", self.name, MODEL_PATH)
        self._model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scalers = pickle.load(f)
        self._feat_scaler = scalers["feat"]
        self._tgt_scaler  = scalers["tgt"]
        logger.info("[%s] Model and scalers loaded.", self.name)

    # Evaluate

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        featured = engineer_features(df).dropna(subset=FEATURE_COLS)

        X_all = featured[FEATURE_COLS].values
        y_all = featured["Close"].shift(-1).dropna().values

        # Use test portion (with a SEQ_LEN buffer from the train end)
        split = len(X_all) - TEST_DAYS
        buffer_X = self._feat_scaler.transform(X_all[split - SEQ_LEN : split])
        test_X = self._feat_scaler.transform(X_all[split : split + TEST_DAYS])
        X_combined = np.vstack([buffer_X, test_X])

        y_test = y_all[split : split + TEST_DAYS]

        # Build sequences over test window
        Xs = np.array([X_combined[i : i + SEQ_LEN] for i in range(len(test_X))])
        preds_sc = self._model.predict(Xs, verbose=0)
        preds = self._tgt_scaler.inverse_transform(preds_sc).flatten()
        actual_len = min(len(preds), len(y_test))

        return compute_metrics(y_test[:actual_len], preds[:actual_len])

    # Predict

    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        featured = engineer_features(df).dropna(subset=FEATURE_COLS)
        last_seq = self._feat_scaler.transform(featured[FEATURE_COLS].values[-SEQ_LEN:])
        pred_scaled = self._model.predict(last_seq[np.newaxis], verbose=0)[0, 0]
        predicted_price = float(self._tgt_scaler.inverse_transform([[pred_scaled]])[0, 0])
        current_price = float(df["Close"].iloc[-1])

        return self._build_result(
            predicted_price, current_price,
            details={"seq_len": SEQ_LEN, "n_features": len(FEATURE_COLS)},
        )
