# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe real call results while preserving the wrapped callable's interface."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps


def record_results[**Parameters, Result](
    function: Callable[Parameters, Result], results: list[Result]
) -> Callable[Parameters, Result]:
    """Append successful results without changing arguments or exceptions."""

    @wraps(function)
    def recorded(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        result = function(*args, **kwargs)
        results.append(result)
        return result

    return recorded
