"""
Observability helpers.

  - LangSmith tracing  : enabled when LANGCHAIN_API_KEY is set in env
  - In-memory metrics  : lightweight counters/timings exposed via /metrics
"""
import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


# LangSmith setup

def setup_langsmith() -> bool:
    """
    Enable LangSmith tracing when LANGCHAIN_API_KEY is present.
    Returns True if tracing was activated.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        logger.info("LangSmith tracing disabled (LANGCHAIN_API_KEY not set).")
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "btc-prediction"))
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    project = os.environ["LANGCHAIN_PROJECT"]
    logger.info("LangSmith tracing enabled (project=%r).", project)
    return True


# In-memory metrics

@dataclass
class _PipelineRun:
    query: str
    intent: str
    latency_s: float
    model_ok: int
    model_fail: int
    timestamp: float = field(default_factory=time.time)


class MetricsStore:
    """Thread-safe-enough in-memory store for pipeline run stats."""

    def __init__(self, max_history: int = 200):
        self._runs: List[_PipelineRun] = []
        self._max_history = max_history
        self._model_errors: Dict[str, int] = defaultdict(int)

    def record(
        self,
        query: str,
        intent: str,
        latency_s: float,
        model_results: list,
    ) -> None:
        ok = sum(1 for r in model_results if r.get("predicted_price") is not None)
        fail = len(model_results) - ok

        for r in model_results:
            if r.get("predicted_price") is None:
                self._model_errors[r["model"]] += 1

        run = _PipelineRun(
            query=query, intent=intent,
            latency_s=latency_s, model_ok=ok, model_fail=fail,
        )
        self._runs.append(run)
        if len(self._runs) > self._max_history:
            self._runs.pop(0)

    def summary(self) -> dict:
        if not self._runs:
            return {"total_requests": 0}

        latencies = [r.latency_s for r in self._runs]
        total = len(self._runs)
        intents = defaultdict(int)
        for r in self._runs:
            intents[r.intent] += 1

        recent = self._runs[-10:]

        return {
            "total_requests": total,
            "latency": {
                "mean_s": round(sum(latencies) / total, 2),
                "min_s": round(min(latencies), 2),
                "max_s": round(max(latencies), 2),
                "p95_s": round(sorted(latencies)[int(total * 0.95)], 2),
            },
            "intent_counts": dict(intents),
            "model_error_counts": dict(self._model_errors),
            "recent_requests": [
                {
                    "query": r.query[:80],
                    "intent": r.intent,
                    "latency_s": round(r.latency_s, 2),
                    "models_ok": r.model_ok,
                    "models_failed": r.model_fail,
                }
                for r in reversed(recent)
            ],
        }


# Singleton
metrics = MetricsStore()
