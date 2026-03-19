from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def debug_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.debug("%s %s", event, json.dumps(_normalize(fields), sort_keys=True, ensure_ascii=False))


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if hasattr(value, "to_dict"):
        return _normalize(value.to_dict())
    if hasattr(value, "__dict__"):
        return _normalize({key: item for key, item in vars(value).items() if not key.startswith("_")})
    return str(value)
