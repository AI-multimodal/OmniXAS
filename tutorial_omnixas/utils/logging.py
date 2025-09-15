import io
import logging
import sys
from pathlib import Path

import colorlog


class StreamToLogger(io.StringIO):
    """Custom stream to redirect print statements to a logger."""

    def __init__(self, logger, level):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.strip()
        if message:  # Avoid empty log messages
            self.logger.log(self.level, message)

    def flush(self):
        pass  # No-op for compatibility


def setup_logger(
    app_name: str = "omnixas",
    log_level: str = "INFO",
    capture_prints: bool = False,
    file_logging: bool = False,
    capture_warnings: bool = False,
) -> None:
    # Get the root logger
    root = logging.getLogger()

    if root.hasHandlers():
        return  # Avoid duplicate handlers

    # Configure console handler with color formatting
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-2s%(reset)s %(blue)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Add file handler if requested
    if file_logging:
        log_dir = Path.home() / ".local" / "share" / "logs" / app_name
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=10_000_000, backupCount=5
        )
        file_formatter = logging.Formatter("%(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    # Set log level
    root.setLevel(log_level)

    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.addHandler(handler)

    # Configure print capture if requested
    if capture_prints:
        sys.stdout = StreamToLogger(root, logging.INFO)
        sys.stderr = StreamToLogger(root, logging.ERROR)
        root.warn("Logging has been configured to capture print statements.")
