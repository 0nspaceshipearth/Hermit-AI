"""Runtime profiles and observed performance metrics for adaptive generation."""

import json
import os
import threading
import time
from typing import Any, Dict, Optional

_PROFILE_LOCK = threading.Lock()


def _project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def _profile_path() -> str:
    data_dir = os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "model_profiles.json")


def _default_payload() -> Dict[str, Any]:
    return {"models": {}}


def load_profiles() -> Dict[str, Any]:
    path = _profile_path()
    if not os.path.exists(path):
        return _default_payload()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data
    except Exception:
        pass
    return _default_payload()


def save_profiles(payload: Dict[str, Any]) -> None:
    path = _profile_path()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def record_generation_metrics(
    model: str,
    lane: str,
    prompt_chars: int,
    output_chars: int,
    duration_seconds: float,
    first_token_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not model:
        return

    with _PROFILE_LOCK:
        payload = load_profiles()
        model_entry = payload["models"].setdefault(model, {})
        lane_entry = model_entry.setdefault("lanes", {}).setdefault(lane, {
            "samples": 0,
            "avg_duration_seconds": 0.0,
            "avg_first_token_seconds": None,
            "avg_prompt_chars": 0.0,
            "avg_output_chars": 0.0,
            "last_updated": None,
            "last_metadata": {},
        })

        samples = int(lane_entry.get("samples", 0)) + 1

        def _avg(previous: Optional[float], value: float) -> float:
            base = float(previous or 0.0)
            return ((base * (samples - 1)) + value) / samples

        lane_entry["samples"] = samples
        lane_entry["avg_duration_seconds"] = _avg(lane_entry.get("avg_duration_seconds"), duration_seconds)
        lane_entry["avg_prompt_chars"] = _avg(lane_entry.get("avg_prompt_chars"), float(prompt_chars))
        lane_entry["avg_output_chars"] = _avg(lane_entry.get("avg_output_chars"), float(output_chars))
        lane_entry["last_updated"] = int(time.time())

        if first_token_seconds is not None:
            prev_first = lane_entry.get("avg_first_token_seconds")
            prev_first = float(prev_first) if prev_first is not None else 0.0
            lane_entry["avg_first_token_seconds"] = ((prev_first * (samples - 1)) + first_token_seconds) / samples

        if metadata:
            lane_entry["last_metadata"] = metadata

        save_profiles(payload)


def get_model_lane_stats(model: str, lane: str) -> Optional[Dict[str, Any]]:
    payload = load_profiles()
    return payload.get("models", {}).get(model, {}).get("lanes", {}).get(lane)
