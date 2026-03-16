import logging
import os
from functools import lru_cache
from typing import Any, Dict

import pandas as pd

from ..features.engineering import DATA_PATH
from ..models.arima_model import ARIMAPredictor
from ..models.xgboost_model import XGBoostPredictor
from ..models.lstm_model import LSTMPredictor
from ..models.prophet_model import ProphetPredictor
from ..models.patchtst_model import PatchTSTPredictor
from .state import PipelineState

logger = logging.getLogger(__name__)


# Model singletons

@lru_cache(maxsize=1)
def _get_models() -> Dict[str, Any]:
    return {
        "arima": ARIMAPredictor(),
        "xgboost": XGBoostPredictor(),
        "lstm": LSTMPredictor(),
        "prophet": ProphetPredictor(),
        "patchtst": PatchTSTPredictor(),
    }


# Helpers

def _df_from_state(state: PipelineState) -> pd.DataFrame:
    df = pd.read_json(state["df"], orient="split")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# Node: intent router

# Keywords that signal the user wants to trigger (re)training
_TRAIN_KEYWORDS = {
    "train", "retrain", "re-train", "fit", "rebuild",
    "update model", "update models", "refresh model", "refresh models",
}


def _is_btc_query(query: str) -> bool:
    """
    Use LLM to determine whether the query is specifically about Bitcoin (BTC).
    Returns True only for BTC-related questions; ETH, other cryptos, or unrelated
    topics return False.
    """
    system = (
        "You are a strict query classifier. "
        "Reply with exactly one word: YES if the user's question is specifically about "
        "Bitcoin (BTC/BTC-USD/Bitcoin price/Bitcoin prediction/Bitcoin market). "
        "Reply NO for everything else — including other cryptocurrencies like ETH, SOL, "
        "ADA, XRP, general crypto questions, or completely unrelated topics."
    )
    response = _call_llm(system, query).strip().upper()
    logger.debug("BTC guardrail LLM response: %r", response)
    return response.startswith("YES")


def router_node(state: PipelineState) -> Dict[str, Any]:
    """
    Classify user query into 'train' | 'predict' | 'off_topic' intent.

    Guardrail: LLM decides if query is specifically about Bitcoin.
    Non-BTC queries (including other cryptos) are marked off_topic and skip
    all data-fetch and model nodes entirely.
    """
    query = state["user_query"]

    # Check train intent first — retrain commands are implicitly BTC-scoped
    if any(kw in query.lower() for kw in _TRAIN_KEYWORDS):
        logger.info("Router -> intent=train (force_retrain=True)")
        return {"intent": "train", "force_retrain": True}

    if not _is_btc_query(query):
        logger.info("Router -> intent=off_topic (LLM guardrail rejected query)")
        return {"intent": "off_topic", "force_retrain": False}

    logger.info("Router -> intent=predict")
    return {"intent": "predict", "force_retrain": False}


def _safe_run(model_key: str, df: pd.DataFrame, force_retrain: bool) -> Dict[str, Any]:
    """Run full pipeline for one model; return error dict on failure."""
    predictor = _get_models()[model_key]
    try:
        return predictor.run_pipeline(df, force_retrain=force_retrain)
    except Exception as exc:
        logger.error("[%s] pipeline failed: %s", predictor.name, exc, exc_info=True)
        return {
            "model": predictor.name,
            "predicted_price": None,
            "current_price": float(df["Close"].iloc[-1]),
            "direction": "N/A",
            "pct_change": None,
            "rmse": None,
            "mape": None,
            "dir_accuracy": None,
            "trained": False,
            "details": {"error": str(exc)},
        }


# Node: fetch latest BTC data

def fetch_data_node(_state: PipelineState) -> Dict[str, Any]:
    """
    Smart incremental data fetch:
      - CSV exists and is current (last row = today or yesterday) -> load CSV, no download
      - CSV exists but stale -> fetch only the missing days and append
      - CSV missing -> full 3-year download
    """
    import yfinance as yf
    from datetime import date, datetime, timedelta

    today = date.today()

    
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()
        last_date = df.index[-1].date()

        if last_date >= today - timedelta(days=1):
            logger.info("Data is current (last=%s) — skipping download.", last_date)
            return {
                "df": df.reset_index().to_json(orient="split", date_format="iso"),
                "current_price": float(df["Close"].iloc[-1]),
                "last_date": str(last_date),
                "model_results": [],
            }

        fetch_start = last_date + timedelta(days=1)
        logger.info("Fetching missing days %s -> %s …", fetch_start, today)
        new = yf.download(
            "BTC-USD", start=fetch_start, end=today + timedelta(days=1),
            interval="1d", progress=False,
        )
        if not new.empty:
            if isinstance(new.columns, pd.MultiIndex):
                new.columns = new.columns.get_level_values(0)
            new.index.name = "Date"
            df = pd.concat([df, new[~new.index.isin(df.index)]]).sort_index()
            df = df.ffill().bfill()
            df.to_csv(DATA_PATH)
            logger.info("Appended %d new rows -> %s (last=%s)", len(new), DATA_PATH, df.index[-1].date())
        else:
            logger.info("No new rows returned by yfinance — using existing CSV.")

    else:
        # No CSV — full 3-year download
        logger.info("CSV not found — performing full 3-year download …")
        start = datetime.now() - timedelta(days=3 * 365 + 60)
        df = yf.download("BTC-USD", start=start, end=datetime.now(),
                         interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df = df.ffill().bfill()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH)
        logger.info("Saved %d rows -> %s", len(df), DATA_PATH)

    return {
        "df": df.reset_index().to_json(orient="split", date_format="iso"),
        "current_price": float(df["Close"].iloc[-1]),
        "last_date": str(df.index[-1].date()),
        "model_results": [],
    }


# Model nodes (each runs in its own parallel branch)

def _run_model_node(key: str, state: PipelineState) -> Dict[str, Any]:
    logger.info("Node [%s] started", key)
    result = _safe_run(key, _df_from_state(state), state["force_retrain"])
    status = "OK" if result.get("predicted_price") is not None else "ERROR"
    logger.info("Node [%s] finished — %s", key, status)
    return {"model_results": [result]}


def arima_node(state: PipelineState) -> Dict[str, Any]:
    return _run_model_node("arima", state)


def xgboost_node(state: PipelineState) -> Dict[str, Any]:
    return _run_model_node("xgboost", state)


def lstm_node(state: PipelineState) -> Dict[str, Any]:
    return _run_model_node("lstm", state)


def prophet_node(state: PipelineState) -> Dict[str, Any]:
    return _run_model_node("prophet", state)


def patchtst_node(state: PipelineState) -> Dict[str, Any]:
    return _run_model_node("patchtst", state)


# Node: LLM synthesis

def synthesize_node(state: PipelineState) -> Dict[str, Any]:
    # Guardrail: off-topic queries never reach models — reply immediately
    if state.get("intent") == "off_topic":
        logger.info("Synthesize — off_topic query, returning guardrail response.")
        return {
            "synthesis": (
                "I'm specialised in Bitcoin (BTC/USDT) price analysis and predictions. "
                "Your question doesn't appear to be related to BTC. "
                "Please ask me something about Bitcoin prices, trends, or forecasts and I'll be happy to help!"
            )
        }

    results = state["model_results"]
    user_query = state["user_query"]
    current_price = state["current_price"]
    last_date = state["last_date"]

    pred_lines = []
    for r in results:
        if r.get("predicted_price") is None:
            pred_lines.append(f"  • {r['model']}: ERROR — {r['details'].get('error', '?')}")
            continue
        rmse_str = f"${r['rmse']:,.0f}" if r.get("rmse") else "N/A"
        pred_lines.append(
            f"  • {r['model']}: ${r['predicted_price']:,.2f} "
            f"({r['direction']}, {r['pct_change']:+.2f}%) "
            f"| RMSE {rmse_str}, MAPE {r.get('mape','N/A')}%, "
            f"Dir-Acc {r.get('dir_accuracy','N/A')}%"
        )

    directions = [r["direction"] for r in results if r.get("direction") not in (None, "N/A")]
    consensus = "UP" if directions.count("UP") >= directions.count("DOWN") else "DOWN"

    system_prompt = (
        "You are an expert quantitative analyst specialising in cryptocurrency markets. "
        "You have just run a full ML pipeline — live data fetch, model training, "
        "evaluation, and next-day prediction — across five independently trained "
        "time-series models for BTC/USDT. Answer the user's question concisely, "
        "referencing both model predictions and their evaluation metrics. "
        "Always include a disclaimer that past performance does not guarantee future results."
    )

    user_prompt = (
        f"User question: {user_query}\n\n"
        f"##BTC Market Context\n"
        f"Last data date : {last_date}\n"
        f"Current price  : ${current_price:,.2f}\n\n"
        f"##Model Results (next-day prediction + 60-day test evaluation)\n"
        f"{chr(10).join(pred_lines)}\n\n"
        f"Consensus direction: {consensus} "
        f"({directions.count('UP')}/{len(directions)} models bullish)\n\n"
        "Provide a comprehensive answer interpreting the collective output, "
        "noting which models performed best on the test set and what that implies."
    )

    logger.info("Synthesize — consensus=%s (%d/%d bullish), calling LLM …",
                consensus, directions.count("UP"), len(directions))
    synthesis = _call_llm(system_prompt, user_prompt)
    logger.info("LLM synthesis received (%d chars).", len(synthesis))
    return {"synthesis": synthesis}



# LLM helpers
def _call_llm(system: str, user: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    logger.debug("Calling LLM model=%s", model_name)
    llm = ChatOpenAI(model=model_name, max_tokens=512)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    logger.debug("LLM response tokens: ~%d", len(response.content.split()))
    return response.content
