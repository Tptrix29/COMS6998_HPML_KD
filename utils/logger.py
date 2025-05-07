import os
import sys
import time
import logging


class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[1;91m' # Bold Red
    }
    RESET_CODE = '\033[0m'

    def format(self, record):
        log_color = self.COLOR_CODES.get(record.levelname, '')
        message = super().format(record)
        return f"{log_color}{message}{self.RESET_CODE}"


class Logger:
    def __init__(self, name: str, level: int = logging.DEBUG, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent double logging if root logger is used elsewhere

        # Avoid adding multiple handlers in interactive environments
        if not self.logger.handlers:
            # Console handler with colors
            console_handler = logging.StreamHandler(sys.stdout)
            console_format = ColoredFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

            # Optional file handler (no color)
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)

    # Expose logger methods directly
    def debug(self, msg, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self.logger.critical(msg, *args, **kwargs)

