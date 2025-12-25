"""Logging configuration for the application."""

import logging

logger = logging.getLogger("tensorflow_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

if (
    not logger.handlers
):  # Avoid logging being propagated to the root logger (prevents duplicate logs)
    logger.addHandler(console_handler)

logger.propagate = False
