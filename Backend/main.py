import logging
import time
from pathlib import Path
from typing import Any, Dict

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from logging_config import setup_logging
from observability import setup_langsmith, metrics

setup_logging(log_file=Path(__file__).parent / "logs" / "app.log")
setup_langsmith()

logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Prediction Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("-> %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info("← %s %s  status=%d  %.2fs",
                request.method, request.url.path, response.status_code, elapsed)
    return response


# Lazy-load the agent so startup is fast
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        logger.info("Initialising PredictionAgent …")
        from intelligence import PredictionAgent
        _agent = PredictionAgent()
        logger.info("PredictionAgent ready.")
    return _agent


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    synthesis: str
    model_results: list
    current_price: float
    last_date: str
    table: str
    run_id: str
    latency_s: float


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metrics")
def get_metrics():
    return metrics.summary()


MAX_QUERY_TOKENS = 134  # 1 100 total - 110 (system) - 344 (user_prompt static + pred lines) - 512 (response)

def _ensure_english(text: str) -> str:
    """Detect the language of the query via LLM.

    - If English: return the original text unchanged.
    - If non-English: translate it to English in the same LLM call and return
      the translated version so the pipeline always receives English input.
    - On any LLM error: log a warning and return the original text unchanged.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import os

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            max_tokens=256,
            temperature=0,
        )
        response = llm.invoke([
            SystemMessage(content=(
                "You are a language detector and translator. "
                "If the user message is already in English, reply with exactly: ENGLISH: <original text>. "
                "If it is in any other language, translate it to English and reply with: TRANSLATED: <english translation>. "
                "Output nothing else."
            )),
            HumanMessage(content=text),
        ])
        reply = response.content.strip()
        if reply.upper().startswith("TRANSLATED:"):
            translated = reply[len("TRANSLATED:"):].strip()
            logger.info("Query translated to English: %r -> %r", text, translated)
            return translated
        # ENGLISH: prefix — return original
        return text
    except Exception:
        logger.warning("Language detection/translation via LLM failed — using original query.", exc_info=True)
        return text


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (OpenAI/Anthropic rule of thumb)."""
    return max(1, len(text) // 4)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> Dict[str, Any]:
    import uuid
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    query = _ensure_english(req.query)

    query_tokens = _estimate_tokens(query)
    if query_tokens > MAX_QUERY_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Your message is too long. Please keep your question short and concise — "
                f"try to stay under {MAX_QUERY_TOKENS * 4} characters."
            ),
        )

    run_id = str(uuid.uuid4())
    logger.info("Chat query: %r  run_id=%s", query, run_id)
    t0 = time.perf_counter()
    try:
        agent = get_agent()
        result = agent.run(query)
        latency = time.perf_counter() - t0

        n_ok = sum(1 for r in result["model_results"] if r.get("predicted_price") is not None)
        n_fail = len(result["model_results"]) - n_ok
        logger.info(
            "Pipeline done — run_id=%s price=%.2f date=%s models_ok=%d models_failed=%d latency=%.2fs",
            run_id, result["current_price"], result["last_date"], n_ok, n_fail, latency,
        )
        if n_fail:
            logger.warning("%d model(s) returned errors — check logs above.", n_fail)

        metrics.record(
            query=query,
            intent=result.get("intent", "predict"),
            latency_s=latency,
            model_results=result["model_results"],
        )

        table = agent.format_table(result["model_results"]) if result["model_results"] else ""

        return {
            **result,
            "table": table,
            "run_id": run_id,
            "latency_s": round(latency, 3),
        }
    except Exception as exc:
        logger.exception("Pipeline error for query %r run_id=%s: %s", req.query, run_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


app.include_router(router)             # /chat, /health, /metrics  (nginx production)
app.include_router(router, prefix="/api")  # /api/chat, /api/health  (dev proxy)
