"""
PatchTST predictor — loads fine-tuned checkpoint, evaluates, predicts.

PatchTST is NOT retrained by the pipeline (fine-tuning requires the full
tsfm/HuggingFace training loop from the experiment notebook).
The model's internal scaling (config: scaling="std") handles normalisation,
so raw Close prices are passed directly — no external scaler needed.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BasePredictor, ARTIFACTS_DIR, TEST_DAYS, compute_metrics, save_metadata

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = (
    Path(__file__).parents[2]
    / "experiment"
    / "patchtst_btc_finetune"
    / "checkpoint-288"
)
CONTEXT_LEN = 516


class PatchTSTPredictor(BasePredictor):
    """Fine-tuned PatchTST — load-only (training happens in experiment notebook)."""

    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "PatchTST (Transformer)"

    def _artifact_path(self) -> Path:
        # Use the checkpoint config as the sentinel for freshness checks.
        return CHECKPOINT_PATH / "config.json"

    # Train (not supported — checkpoint is already fine-tuned)

    def train(self, df: pd.DataFrame) -> None:
        raise NotImplementedError(
            "PatchTST fine-tuning must be run in experiment/experiments.ipynb. "
            "The checkpoint at experiment/patchtst_btc_finetune/checkpoint-256/ is used."
        )

    # Load

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import PatchTSTForPrediction

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"PatchTST checkpoint not found at {CHECKPOINT_PATH}. "
                "Run experiment/experiments.ipynb to fine-tune the model."
            )
        logger.info("[%s] Loading checkpoint from %s …", self.name, CHECKPOINT_PATH)
        self._model = PatchTSTForPrediction.from_pretrained(str(CHECKPOINT_PATH))
        self._model.eval()
        logger.info("[%s] Checkpoint loaded (context_len=%d).", self.name, CONTEXT_LEN)

    # Pipeline override — never retrain, always load

    def _maybe_train(self, df: pd.DataFrame, last_data_date: str, force_retrain: bool = False) -> bool:
        """PatchTST skips training entirely — just load the checkpoint."""
        self._load()
        # Keep metadata in sync so the pipeline doesn't try to retrain later.
        save_metadata(self.name, last_data_date)
        return False   # never newly trained in this pipeline

    # Evaluate

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Rolling forward pass on the 60-day test window."""
        import torch

        close = df["Close"].values.astype(np.float32)
        preds = []
        y_true = close[-TEST_DAYS:]
        logger.debug("[%s] Starting rolling evaluation (%d passes) …", self.name, TEST_DAYS)

        for i in range(TEST_DAYS):
            # Context ends just before test day i
            ctx_end = len(close) - TEST_DAYS + i
            ctx_start = ctx_end - CONTEXT_LEN
            if ctx_start < 0:
                preds.append(float("nan"))
                continue
            ctx = torch.tensor(close[ctx_start:ctx_end]).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                out = self._model.generate(past_values=ctx)
            # sequences: (batch, n_samples, pred_len, channels)
            pred = float(out.sequences[0, :, 0, 0].mean().cpu())
            preds.append(pred)

        preds = np.array(preds)
        valid = ~np.isnan(preds)
        return compute_metrics(y_true[valid], preds[valid])

    # Predict

    def predict_next(self, df: pd.DataFrame) -> Dict[str, Any]:
        import torch

        close = df["Close"].values.astype(np.float32)
        context = torch.tensor(close[-CONTEXT_LEN:]).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            out = self._model.generate(past_values=context)

        samples = out.sequences[0, :, 0, 0].cpu().numpy()
        predicted_price = float(samples.mean())
        current_price = float(df["Close"].iloc[-1])

        return self._build_result(
            predicted_price, current_price,
            details={
                "ci_90_lower":  round(float(np.percentile(samples, 5)), 2),
                "ci_90_upper":  round(float(np.percentile(samples, 95)), 2),
                "context_days": CONTEXT_LEN,
            },
        )
