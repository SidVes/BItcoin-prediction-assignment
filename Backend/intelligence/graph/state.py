"""LangGraph state for the full fetch -> train -> evaluate -> predict pipeline."""
import operator
from typing import Annotated, Any, Dict, List, TypedDict

import pandas as pd


class ModelResult(TypedDict):
    # Identity
    model: str
    # Prediction
    predicted_price: float
    current_price: float
    direction: str           # "UP" | "DOWN"
    pct_change: float
    # Evaluation
    rmse: float
    mape: float
    dir_accuracy: float
    # Pipeline meta
    trained: bool
    details: Dict[str, Any]  # model-specific extras


class PipelineState(TypedDict):
    # User intent — set by router_node
    user_query: str
    intent: str   # "train" | "predict"
    force_retrain: bool  # True forces all models to retrain regardless of cache

    # Set by fetch_data_node — stored as JSON string to keep state JSON-serializable
    df: str
    current_price: float
    last_date: str

    # Parallel-safe accumulator — each model node appends one ModelResult
    model_results: Annotated[List[ModelResult], operator.add]

    # Written by synthesize_node
    synthesis: str
