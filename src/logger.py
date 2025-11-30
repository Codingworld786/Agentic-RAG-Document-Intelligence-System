# src/logger.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Unique log file per run
log_filename = LOG_DIR / f"agentic_rag_{datetime.now():%Y%m%d_%H%M%S}.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RotatingFileHandler(
            log_filename,
            maxBytes=10_000_000,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        ),
        logging.StreamHandler()  # Still shows in console!
    ]
)

# Create dedicated loggers
logger = logging.getLogger("AgenticRAG")
router_logger = logging.getLogger("Router")
retrieval_logger = logging.getLogger("Retrieval")
answer_logger = logging.getLogger("Answer")

# Optional: Color in console (beautiful!)
try:
    from colorlog import ColoredFormatter
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':     'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
except ImportError:
    pass  # colorlog not installed → still works without colors

# Export for use
__all__ = ["logger", "router_logger", "retrieval_logger", "answer_logger"]