"""Centralized logging configuration — call setup_logging() at process startup."""
import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that are very chatty at INFO
_QUIET = [
    "prophet", "cmdstanpy", "transformers", "tensorflow", "tf2onnx",
    "yfinance", "urllib3", "httpx", "httpcore", "absl",
    "langchain", "langgraph", "openai",
]


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """
    Configure the root logger with a console handler (stdout) and an
    optional rotating file handler.  Silences noisy third-party loggers.
    """
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    for lib in _QUIET:
        logging.getLogger(lib).setLevel(logging.WARNING)
