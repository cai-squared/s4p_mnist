import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Install pretty tracebacks globally
install_rich_traceback(show_locals=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(name: str = "s4p_mnist", level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging to stdout (rich) and a rotating file.

    Args:
        name: Logger name (typically the module name).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.propagate = False
        return logger  # already configured, avoid duplicate handlers

    # --- Handler 1: Rich colored console output ---
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    rich_handler.setLevel(level)

    # --- Handler 2: Rotating file under logs/ ---
    file_handler = RotatingFileHandler(
        LOGS_DIR / "s4p_mnist.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Typically pass __name__ from the calling module.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
