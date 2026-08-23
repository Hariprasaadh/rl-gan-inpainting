import os
import sys
import logging
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


def setup_logger(name: str = "rl_gan", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close and remove all handlers from a logger."""
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


class MetricLogger:
    """Helper for writing scalars to TensorBoard and keeping running averages."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metrics: dict = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, tag_dict: dict, step: int) -> None:
        for k, v in tag_dict.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()
