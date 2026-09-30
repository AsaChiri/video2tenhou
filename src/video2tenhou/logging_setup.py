# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Command logging with progress on stderr and machine-readable results on stdout."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import TextIO

RESULT = logging.getLogger("video2tenhou.result")


@contextmanager
def _command_logger(name: str, stream: TextIO) -> Iterator[None]:
    logger = logging.getLogger(name)
    handlers, level, propagate = logger.handlers, logger.level, logger.propagate
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        handler.close()


def command_logging[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Configure command streams temporarily, preserving an embedding app's logging."""

    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        with (
            _command_logger("video2tenhou", sys.stderr),
            _command_logger(RESULT.name, sys.stdout),
        ):
            return function(*args, **kwargs)

    return run
