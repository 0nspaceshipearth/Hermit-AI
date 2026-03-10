"""Wave-mode teleport contracts and guarded execution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import re
import time
import uuid

from chatbot import config


@dataclass
class TeleportEnvelope:
    contract_version: str
    intent: str
    source_mode: str
    target_mode: str
    objective: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_reason: str = ""
    fallback_reason: str = ""
    context_hash: str = ""
    created_at: float = field(default_factory=time.time)

    def validate(self) -> Dict[str, Any]:
        issues = []
        if self.contract_version != "teleport.v1":
            issues.append("unsupported contract version")
        if not self.intent:
            issues.append("missing intent")
        if not self.objective:
            issues.append("missing objective")
        if not self.source_mode:
            issues.append("missing source mode")
        if not self.target_mode:
            issues.append("missing target mode")
        return {"ok": not issues, "issues": issues}


@dataclass
class TeleportResult:
    ok: bool
    status: str
    message: str
    artifact: Dict[str, Any] = field(default_factory=dict)
    envelope: Optional[TeleportEnvelope] = None


def wave_mode_enabled() -> bool:
    return str(getattr(config, "RUNTIME_MODE", "classic") or "classic").strip().lower() == "wave"


def parse_python_desktop_write_intent(user_input: str) -> Optional[Dict[str, Any]]:
    text = (user_input or "").strip()
    low = text.lower()
    if not text:
        return None
    if "desktop" not in low:
        return None
    if "python" not in low and ".py" not in low and "script" not in low:
        return None
    if not any(term in low for term in ["write", "create", "save", "put", "paste"]):
        return None

    filename = "desktop_script.py"
    m = re.search(r"(?:as|named)\s+([a-zA-Z0-9_.-]+\.py)\b", text, flags=re.IGNORECASE)
    if m:
        filename = m.group(1)
    elif "clock" in low:
        filename = "clock_app.py"

    desktop_dir = os.path.expanduser("~/Desktop")
    return {
        "objective": text,
        "filename": filename,
        "target_path": os.path.join(desktop_dir, filename),
        "requires_confirmation": True,
    }


def build_desktop_write_envelope(user_input: str) -> Optional[TeleportEnvelope]:
    parsed = parse_python_desktop_write_intent(user_input)
    if not parsed:
        return None
    envelope = TeleportEnvelope(
        contract_version="teleport.v1",
        intent="desktop_python_write",
        source_mode="wave",
        target_mode="chamber",
        objective=parsed["objective"],
        constraints={
            "target_path": parsed["target_path"],
            "filename": parsed["filename"],
            "requires_confirmation": parsed["requires_confirmation"],
            "allow_prose_fallback": False,
        },
        policy_reason="desktop write crosses chamber boundary and needs explicit teleport contract",
    )
    return envelope


def execute_desktop_write_contract(envelope: TeleportEnvelope, code: str) -> TeleportResult:
    validation = envelope.validate()
    if not validation["ok"]:
        return TeleportResult(False, "contract_error", "; ".join(validation["issues"]), envelope=envelope)

    target_path = str(envelope.constraints.get("target_path", "")).strip()
    if not target_path:
        return TeleportResult(False, "contract_error", "missing target path", envelope=envelope)

    parent = os.path.dirname(target_path)
    if parent and not os.path.isdir(parent):
        return TeleportResult(False, "boundary_error", f"target directory does not exist: {parent}", envelope=envelope)

    try:
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write((code or "").rstrip() + "\n")
        return TeleportResult(
            True,
            "completed",
            f"teleport write completed: {target_path}",
            artifact={"path": target_path, "kind": "python_script"},
            envelope=envelope,
        )
    except Exception as e:
        return TeleportResult(False, "execution_error", str(e), envelope=envelope)
