"""Shared runtime helpers for Hermit frontends.

This module centralizes Wave-mode chamber execution utilities so CLI/GUI/TUI
frontends do not grow separate agent brains.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chatbot import config
from chatbot.models import Message

from chatbot.chat import load_runtime_checkpoint, save_runtime_checkpoint
from chatbot.teleport import execute_file_write, execute_teleport_contract, detect_command, TeleportEnvelope


@dataclass
class RuntimeEvent:
    kind: str
    text: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeTurnResult:
    handled: bool
    path: str
    assistant_reply: str = ""
    display_text: str = ""
    messages: List[dict] = field(default_factory=list)
    events: List[RuntimeEvent] = field(default_factory=list)
    intent: Any = None


def _compact_artifact_payload(payload: Optional[dict], max_text: int = 800) -> dict:
    if not isinstance(payload, dict):
        return payload or {}

    compact = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > max_text:
            compact[key] = value[: max_text - 3] + "..."
        else:
            compact[key] = value
    return compact


def record_chamber_artifact(envelope: TeleportEnvelope, status: str, note: str, payload: Optional[dict] = None) -> None:
    """Persist a compact teleport artifact into the runtime checkpoint."""
    checkpoint = load_runtime_checkpoint()
    artifacts = list(checkpoint.get("artifacts", []) or [])
    artifacts.append({
        "mode": envelope.intent,
        "step": "teleport",
        "query": envelope.objective,
        "status": status,
        "note": note,
        "payload": _compact_artifact_payload(payload),
    })
    checkpoint["artifacts"] = artifacts[-8:]
    checkpoint["objective"] = checkpoint.get("objective") or "shell_chamber_execution"
    checkpoint["next_step"] = "summarize_artifact"
    checkpoint["source"] = {
        **(checkpoint.get("source", {}) or {}),
        "routing_mode": "shell_chamber",
        "routing_reason": envelope.intent,
    }
    open_loops = list(checkpoint.get("open_loops", []) or [])
    open_loops.append(note)
    checkpoint["open_loops"] = open_loops[-5:]
    save_runtime_checkpoint(checkpoint)


def execute_teleport_for_workspace(envelope: TeleportEnvelope, workspace: str) -> Optional[str]:
    """Execute a teleport envelope for a frontend workspace.

    Returns a formatted result string, or None when content generation is still
    required (file/script creation).
    """
    intent = envelope.intent
    envelope.constraints["workspace"] = workspace

    if intent == "shell_command":
        if not envelope.constraints.get("command"):
            cmd_guess = detect_command(envelope.objective)
            envelope.constraints["command"] = cmd_guess or envelope.objective
        result = execute_teleport_contract(envelope)
    elif intent in ("file_write", "script_create"):
        return None
    else:
        result = execute_teleport_contract(envelope)

    if result.ok:
        artifact = result.artifact or {}
        record_chamber_artifact(
            envelope,
            "ok",
            f"command completed (exit={artifact.get('exit_code', 0)})",
            payload=artifact,
        )
        output = result.message or "(no output)"
        return f"✅ Command executed successfully (exit={artifact.get('exit_code', 0)})\n{output}"

    record_chamber_artifact(
        envelope,
        "error",
        f"execution failed: {result.status}",
        payload=result.artifact or {"message": result.message},
    )
    return f"❌ Execution failed: {result.status} — {result.message}"


def extract_agent_command(response: str) -> Optional[str]:
    match = re.search(r'\[HERMIT_CMD\](.+?)\[/HERMIT_CMD\]', response or "", re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_file_content(response: str, language: str = None, require_marker: bool = False) -> Tuple[Optional[str], bool]:
    """Extract file content from model output."""
    response = response or ""

    file_match = re.search(r'\[HERMIT_FILE\](.+?)\[/HERMIT_FILE\]', response, re.DOTALL)
    if file_match:
        return file_match.group(1).strip(), True

    fence_pattern = r'```(?:' + (re.escape(language) if language else r'\w*') + r')?\s*\n(.+?)\n```'
    fence_match = re.search(fence_pattern, response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip(), True

    any_fence = re.search(r'```\w*\n(.+?)\n```', response, re.DOTALL)
    if any_fence:
        return any_fence.group(1).strip(), True

    if require_marker:
        return None, False

    lines = response.strip().split('\n')
    if len(lines) > 1 or any(c in response for c in '{}[]()=;:'):
        return response.strip(), False

    return None, False


def execute_file_write_from_response(envelope: TeleportEnvelope, response: str, workspace: str) -> Optional[str]:
    language = envelope.constraints.get("language", "")
    content, _found = extract_file_content(response, language, require_marker=True)
    if not content:
        return None

    target_path = envelope.constraints.get("target_path")
    if not target_path:
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "html": ".html",
            "css": ".css",
            "json": ".json",
            "markdown": ".md",
        }
        ext = ext_map.get(language, ".txt")
        target_path = os.path.join(workspace, f"generated_file{ext}")
        envelope.constraints["target_path"] = target_path

    envelope.constraints["workspace"] = workspace
    result = execute_teleport_contract(envelope, content) if envelope.intent == "script_create" else execute_file_write(envelope, content)
    if result.ok:
        artifact = result.artifact or {}
        record_chamber_artifact(
            envelope,
            "ok",
            f"file written: {artifact.get('path', target_path)}",
            payload=artifact,
        )
        kind_label = "script" if envelope.intent == "script_create" else "file"
        executable = "\nExecutable: yes" if artifact.get("executable") else ""
        return (
            f"✅ {kind_label.capitalize()} written successfully\n"
            f"Path: {artifact.get('path', target_path)}\n"
            f"Size: {artifact.get('size', len(content))} chars\n"
            f"Type: {artifact.get('kind', 'file')}"
            f"{executable}"
        )

    record_chamber_artifact(
        envelope,
        "error",
        f"file write failed: {result.status}",
        payload=result.artifact or {"message": result.message},
    )
    return f"❌ File write failed: {result.status} — {result.message}"


def file_generation_contract(envelope: TeleportEnvelope) -> str:
    language = envelope.constraints.get("language", "")
    target_path = envelope.constraints.get("target_path", "")
    lang_hint = f" ({language})" if language else ""
    path_hint = f"Target path: {target_path}\n" if target_path else ""
    return (
        "\n\nFILE CREATION CONTRACT:\n"
        f"- You are generating file contents{lang_hint}.\n"
        f"- {path_hint}"
        "- Return ONLY the file body wrapped in [HERMIT_FILE]...[/HERMIT_FILE].\n"
        "- Do not include explanations, prefaces, or markdown outside the tags.\n"
        "- If the task is impossible, return [HERMIT_FILE]# unable to generate requested file[/HERMIT_FILE].\n"
    )


def handle_turn(
    system_prompt: str,
    history: List[Message],
    workspace: str,
    execute_teleport: Callable[[TeleportEnvelope, str], Optional[str]],
    build_messages_fn: Callable[..., List[dict]],
    build_messages_with_intent_fn: Callable[..., Tuple[List[dict], Any]],
    generate_text_fn: Callable[[List[dict]], str],
    execute_file_write_fn: Optional[Callable[[TeleportEnvelope, str, str], Optional[str]]] = None,
) -> RuntimeTurnResult:
    """Shared runtime turn handler for CLI/GUI/TUI frontends.

    This is intentionally modest for the first pass: it centralizes the major
    Wave/classic branching so frontends stop reinventing the same logic.
    """
    execute_file_write_fn = execute_file_write_fn or execute_file_write_from_response

    messages, intent = build_messages_with_intent_fn(system_prompt, history)
    result = RuntimeTurnResult(handled=False, path="normal", messages=messages, intent=intent)

    if (
        str(getattr(config, "RUNTIME_MODE", "classic") or "classic").lower() == "wave"
        and getattr(intent, "shell_intent", None)
        and getattr(intent, "teleport_envelope", None)
        and not intent.teleport_envelope.constraints.get("refused")
    ):
        envelope = intent.teleport_envelope

        if envelope.intent == "shell_command":
            teleport_result = execute_teleport(envelope, workspace)
            if teleport_result is not None:
                observation_msg = (
                    f"[SHELL CHAMBER OBSERVATION]\n"
                    f"Intent: {envelope.intent}\n"
                    f"Command: {envelope.constraints.get('command', envelope.objective)}\n"
                    f"Working directory: {workspace}\n"
                    f"Result:\n{teleport_result}\n"
                    f"[/SHELL CHAMBER OBSERVATION]\n\n"
                    f"Summarize the result for the user. If you need to run another command, "
                    f"output it inside [HERMIT_CMD]command here[/HERMIT_CMD] tags."
                )
                turn_history = list(history) + [
                    Message(role="assistant", content=f"[Executed: {envelope.constraints.get('command', '')}]"),
                    Message(role="user", content=observation_msg),
                ]
                follow_messages = build_messages_fn(system_prompt, turn_history)
                assistant_reply = generate_text_fn(follow_messages)
                result.handled = True
                result.path = "wave_shell"
                result.assistant_reply = assistant_reply
                result.display_text = teleport_result
                result.messages = follow_messages
                result.events.append(RuntimeEvent(kind="teleport", text=teleport_result, data={"intent": envelope.intent}))
                return result

        if envelope.intent in ("file_write", "script_create"):
            gen_messages = [dict(m) for m in messages]
            if gen_messages:
                gen_messages[0]["content"] = gen_messages[0].get("content", "") + file_generation_contract(envelope)
            assistant_reply = generate_text_fn(gen_messages)
            write_result = execute_file_write_fn(envelope, assistant_reply, workspace)
            result.handled = True
            result.path = "wave_file"
            result.assistant_reply = assistant_reply
            result.display_text = write_result or ""
            result.messages = gen_messages
            if write_result:
                result.events.append(RuntimeEvent(kind="artifact", text=write_result, data={"intent": envelope.intent}))
            return result

    # normal / classic path
    assistant_reply = generate_text_fn(messages)
    result.handled = True
    result.assistant_reply = assistant_reply
    result.display_text = assistant_reply
    return result
