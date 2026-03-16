from langgraph.graph import END, START, StateGraph
from .state import PipelineState
from .nodes import (
    router_node,
    fetch_data_node,
    arima_node,
    xgboost_node,
    lstm_node,
    prophet_node,
    patchtst_node,
    synthesize_node,
)


def build_workflow():
    """Compile and return the executable LangGraph workflow."""
    builder = StateGraph(PipelineState)

    builder.add_node("router", router_node)
    builder.add_node("fetch_data", fetch_data_node)
    builder.add_node("arima", arima_node)
    builder.add_node("xgboost", xgboost_node)
    builder.add_node("lstm", lstm_node)
    builder.add_node("prophet", prophet_node)
    builder.add_node("patchtst", patchtst_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "router")

    # Guardrail: off-topic queries skip data fetch and all models entirely
    builder.add_conditional_edges(
        "router",
        lambda state: "synthesize" if state.get("intent") == "off_topic" else "fetch_data",
        {"synthesize": "synthesize", "fetch_data": "fetch_data"},
    )

    for model in ("arima", "xgboost", "lstm", "prophet", "patchtst"):
        builder.add_edge("fetch_data", model)

    for model in ("arima", "xgboost", "lstm", "prophet", "patchtst"):
        builder.add_edge(model, "synthesize")

    builder.add_edge("synthesize", END)

    return builder.compile()
