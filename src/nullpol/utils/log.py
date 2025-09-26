from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("nullpol")


def setup_logger(outdir=".", label=None, log_level="INFO"):
    """Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError as exc:
            raise ValueError(f"log_level {log_level} not understood") from exc
    else:
        level = int(log_level)

    # Use the global logger instance
    logger.propagate = False
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M")
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        if label:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            log_file = f"{outdir}/{label}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)
