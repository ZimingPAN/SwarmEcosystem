from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def dump(obj: Any, file_name: str | bytes | Path, *args: Any, **kwargs: Any) -> None:
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name: str | bytes | Path, *args: Any, **kwargs: Any) -> Any:
    path = Path(file_name)
    with path.open("rb") as f:
        return pickle.load(f)
