"""Auto-refreshing price lookup backed by LiteLLM's public price database.

LiteLLM maintains a community price table covering hundreds of models across
providers. We fetch it lazily on the first price lookup, cache it to disk for
~7 days, and fall back to the built-in `pricing._BUILTIN_PRICES` table if
LiteLLM does not know the model or the network is unavailable.

Set `COSTSCOPE_OFFLINE=1` to disable the fetch entirely (e.g. airgapped CI,
reproducible audits where stable prices matter more than fresh ones).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_CACHE_DIR = Path.home() / ".cache" / "costscope"
_CACHE_PATH = _CACHE_DIR / "litellm_prices.json"
_TTL_SECONDS = 7 * 24 * 3600  # one week
_TIMEOUT_SECONDS = 5
_OFFLINE_ENV = "COSTSCOPE_OFFLINE"

_memo: Optional[dict] = None  # in-process cache so we read the JSON once per run


def offline() -> bool:
    """True if the user opted out of network fetches via env var."""
    return os.environ.get(_OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes")


def lookup(model: str) -> Optional[tuple[float, float, float]]:
    """Per-1M-token (input, output, image_output) prices for `model`, or None.

    Returns None when LiteLLM does not know the model, the fetch fails, or the
    user has set COSTSCOPE_OFFLINE. Callers should fall back to their own table.
    """
    data = _data()
    if not data:
        return None
    entry = data.get(model)
    if entry is None:
        lowered = model.lower()
        for key, val in data.items():
            if key.lower() == lowered:
                entry = val
                break
    if not isinstance(entry, dict):
        return None
    in_cost = entry.get("input_cost_per_token")
    out_cost = entry.get("output_cost_per_token")
    if in_cost is None or out_cost is None:
        return None
    # LiteLLM stores per-token; costscope stores per-1M.
    return (float(in_cost) * 1_000_000, float(out_cost) * 1_000_000, 0.0)


def force_refresh() -> bool:
    """Re-fetch the LiteLLM table now, ignoring TTL. True on success."""
    global _memo
    fresh = _fetch()
    if fresh is None:
        return False
    _save(fresh)
    _memo = fresh
    return True


def _data() -> Optional[dict]:
    global _memo
    if _memo is not None:
        return _memo
    if offline():
        return None
    if _CACHE_PATH.exists() and not _is_stale(_CACHE_PATH):
        cached = _load_cached()
        if cached is not None:
            _memo = cached
            return _memo
    fresh = _fetch()
    if fresh is not None:
        _save(fresh)
        _memo = fresh
        return _memo
    # Network failed; serve stale cache if we have one.
    _memo = _load_cached()
    return _memo


def _is_stale(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) > _TTL_SECONDS
    except OSError:
        return True


def _fetch() -> Optional[dict]:
    try:
        with urllib.request.urlopen(_URL, timeout=_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _save(data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
    except OSError:
        pass


def _load_cached() -> Optional[dict]:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
