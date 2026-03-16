import logging
import time
from typing import Any, Dict, List

from .graph.workflow import build_workflow

logger = logging.getLogger(__name__)


class PredictionAgent:
    """
    Orchestrates the full LangGraph pipeline:
    """

    def __init__(self):
        logger.info("Compiling LangGraph workflow …")
        self._graph = build_workflow()
        logger.info("Workflow ready.")

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the full pipeline for a user question.
        """
        initial_state = {
            "user_query": user_query,
            "intent": "",
            "force_retrain": False,
            "df": "",
            "current_price": 0.0,
            "last_date": "",
            "model_results": [],
            "synthesis": "",
        }

        logger.info("Running pipeline for query: %r", user_query)
        t0 = time.perf_counter()
        final = self._graph.invoke(initial_state)
        elapsed = time.perf_counter() - t0
        logger.info("Pipeline complete in %.1fs  (intent=%s, force_retrain=%s)",
                    elapsed, final.get("intent"), final.get("force_retrain"))

        return {
            "synthesis": final["synthesis"],
            "model_results": final.get("model_results", []),
            "current_price": final.get("current_price", 0.0),
            "last_date": final.get("last_date", ""),
        }


    @staticmethod
    def format_table(model_results: List[Dict[str, Any]]) -> str:
        """Return a markdown table of predictions and evaluation metrics."""
        header = (
            "| Model | Prediction | Direction | Δ% | RMSE | MAPE | Dir-Acc | Status |\n"
            "|-------|-----------|-----------|-----|------|------|---------|--------|\n"
        )
        rows = []
        for r in model_results:
            if r.get("predicted_price") is None:
                rows.append(f"| {r['model']} | ERROR | — | — | — | — | — | — |")
                continue
            arrow  = "▲" if r["direction"] == "UP" else "▼"
            status = "trained" if r.get("trained") else "cached"
            rows.append(
                f"| {r['model']} "
                f"| ${r['predicted_price']:,.0f} "
                f"| {arrow} {r['direction']} "
                f"| {r['pct_change']:+.2f}% "
                f"| ${r.get('rmse', 0):,.0f} "
                f"| {r.get('mape', 0):.2f}% "
                f"| {r.get('dir_accuracy', 0):.1f}% "
                f"| {status} |"
            )
        return header + "\n".join(rows)
