from __future__ import annotations

import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object so json.dump(..., allow_nan=False) succeeds.

    - float('nan')/inf/-inf -> None
    - dict/list/tuple -> sanitized recursively
    - everything else -> unchanged
    """

    # Fast-path for common primitives.
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            # JSON keys must be strings; preserve original but coerce to str.
            out[str(k)] = sanitize_for_json(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]

    # Best-effort: try to convert other numeric types (e.g. numpy scalar)
    try:
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            return sanitize_for_json(obj.item())
    except Exception:
        pass

    return obj
