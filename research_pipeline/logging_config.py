from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(processName)-14s | %(name)-16s | %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)
    # Quiet noisy third-party loggers unless we're in DEBUG
    if level > logging.DEBUG:
        logging.getLogger("redis").setLevel(logging.WARNING)
