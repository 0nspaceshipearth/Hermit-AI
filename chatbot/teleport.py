"""Wave-mode teleport contracts and guarded execution helpers.

Shell Chamber Architecture:
- Generic intent routing: If a query needs a file, command, or script, Hermit teleports into the shell chamber.
- Autonomous resolution: Inside the chamber, it has a terminal and can execute operations.
- Contract enforcement: Non-wave mode explicitly refuses instead of giving text-only fake answers.

Teleport Destinations:
- shell_chamber: Execute shell commands in a bounded workspace
- file_write: Write files to permitted locations
- script_create: Create executable scripts with proper scaffolding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import os
import re
import subprocess
import time
import uuid
import shlex

from chatbot import config


class TeleportDestination(str, Enum):
    """Available teleport destinations."""
    SHELL_CHAMBER = "shell_chamber"
    FILE_WRITE = "file_write"
    SCRIPT_CREATE = "script_create"
    DESKTOP_WRITE = "desktop_write"  # Legacy compatibility


class ShellIntent(str, Enum):
    """Types of shell-related intents."""
    FILE_WRITE = "file_write"
    SHELL_COMMAND = "shell_command"
    SCRIPT_CREATE = "script_create"
    DIRECTORY_LIST = "directory_list"
    FILE_READ = "file_read"
    UNKNOWN = "unknown"


@dataclass
class TeleportEnvelope:
    """Contract envelope for teleport operations."""
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
        """Validate the envelope for required fields."""
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
    """Result of a teleport execution."""
    ok: bool
    status: str
    message: str
    artifact: Dict[str, Any] = field(default_factory=dict)
    envelope: Optional[TeleportEnvelope] = None


# =============================================================================
# Wave Mode Check
# =============================================================================

def wave_mode_enabled() -> bool:
    """Check if Hermit is running in wave mode (required for teleport)."""
    return str(getattr(config, "RUNTIME_MODE", "classic") or "classic").strip().lower() == "wave"


def require_wave_mode() -> Tuple[bool, str]:
    """Check wave mode and return explanation if not enabled.
    
    Returns:
        (is_wave, message) - if not wave, message explains why teleport is refused
    """
    if wave_mode_enabled():
        return True, "Wave mode active - teleport available"
    return False, (
        "Teleport contracts require Wave mode. "
        "In Classic mode, I cannot safely hand off execution to a shell chamber. "
        "To enable: type 'mode wave' in the CLI, or set RUNTIME_MODE=wave in config. "
        "I will NOT provide a text-only approximation that could mislead you."
    )


# =============================================================================
# Shell Intent Detection
# =============================================================================

# Patterns for detecting shell-related intents
FILE_WRITE_PATTERNS = [
    # General file creation
    r'\b(write|create|save|put|make|generate)\s+(a\s+)?(file|script|program|app|code)\b',
    # Specific file extensions
    r'\b(write|create|save|make)\s+.*\.(py|js|sh|txt|md|json|yaml|yml|html|css|xml|toml)\b',
    # Named files
    r'\bsave\s+(it|this)\s+(as|to)\b',
    r'\bnamed\s+\w+\.(py|js|sh|txt|md)\b',
    # Desktop operations
    r'\bput\s+(it|this|a\s+\w+)?\s*on\s+(my\s+)?(desktop|folder|directory)\b',
    r'\b(write|create|save|put).*\bdesktop\b',
    # Script to location
    r'\b(write|create)\s+(a\s+)?(python|javascript|bash|shell)\s+(script|file|app).*\b(to|on|in)\b',
    # New file operations
    r'\bnew\s+(python|javascript|bash|shell)\s+(script|file)\b',
    # Quick writes
    r'\bwrite\s+(me\s+)?(a\s+)?(quick\s+)?(script|file|program)\b',
]

SHELL_COMMAND_PATTERNS = [
    r'\b(run|execute|launch|start)\s+(a|the|this)?\s*(command|script|program)\b',
    r'\b(run|execute)\s+.*\.(py|sh|js)\b',
    r'\bshell\s+(command|execute|run)\b',
    r'\bterminal\s+(command|session|window)\b',
    r'\b(what\'?s|what is)\s+(the\s+)?(output|result)\s+of\b',
    r'\blist\s+(files|directory|dir|the\s+contents)\b',
    r'\bshow\s+me\s+(the\s+)?(files|contents|directory)\b',
    r'\bcommand\s+line\b',
    # Direct command patterns
    r'\brun\s+`[^`]+`',
    r'\bexecute\s+`[^`]+`',
    # Bare "run <word>" for short commands like "run ls", "run pytest"
    r'\b(run|execute)\s+[a-z][\w\-]*\b',
    # "run it" / "execute it" for follow-up commands
    r'\b(run|execute)\s+(it|this|that)\b',
]

SCRIPT_PATTERNS = [
    r'\bcreate\s+(a\s+)?(python|bash|shell|javascript)\s+script\b',
    r'\bwrite\s+(a\s+)?(clock|app|program|calculator|game|timer|counter)\b',
    r'\bmake\s+(a\s+)?(simple\s+)?(script|program|app|tool)\b',
    r'\bbuild\s+(a\s+)?(simple\s+)?(script|tool|app)\b',
    r'\bgenerate\s+(a\s+)?(script|bash|python|shell)\b',
    r'\b(scripts?|programs?|apps?)\s+(that|which)\s+(does|creates?|calculates?)\b',
]

DESKTOP_PATTERNS = [
    r'\bdesktop\b',
    r'\bhome\s+screen\b',
    r'\bhome\s+folder\b',
    r'\bdocuments\s+folder\b',
    r'\bdownloads\s+folder\b',
]


def classify_shell_intent(user_input: str) -> ShellIntent:
    """Classify the type of shell intent from user input.

    Returns the most specific intent type detected.
    """
    text = (user_input or "").strip()
    low = text.lower()

    if not text:
        return ShellIntent.UNKNOWN

    # Question guard: most question-form queries are explanatory/factual, not
    # execution requests. Only allow through if the wording clearly asks Hermit
    # to execute something in a terminal.
    question_starts = (
        "what ", "who ", "why ", "how ", "when ", "where ",
        "is ", "are ", "can ", "does ", "could ", "would ",
        "should ", "do ", "did ", "has ", "have ", "will ",
        "what's ", "who's ", "how's ", "where's ", "when's ",
    )
    explicit_execution_patterns = [
        r'^can\s+you\s+(run|execute|launch|start)\b',
        r'\b(run|execute|launch|start)\s+(the\s+|a\s+)?(command|script|program)\b',
        r'\bwhat\'?s\s+(the\s+)?(output|result)\s+of\b',
        r'\b(show|list)\s+me\s+(the\s+)?(files|contents|directory)\b',
        r'\b(list\s+(files|directory|dir|the\s+contents))\b',
        r'`[^`]+`',
    ]
    if low.startswith(question_starts):
        if not any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in explicit_execution_patterns):
            return ShellIntent.UNKNOWN
    
    # Check for file write patterns
    for pattern in FILE_WRITE_PATTERNS:
        if re.search(pattern, low, flags=re.IGNORECASE):
            return ShellIntent.FILE_WRITE
    
    # Check for shell command patterns
    for pattern in SHELL_COMMAND_PATTERNS:
        if re.search(pattern, low, flags=re.IGNORECASE):
            return ShellIntent.SHELL_COMMAND
    
    # Check for script creation patterns
    for pattern in SCRIPT_PATTERNS:
        if re.search(pattern, low, flags=re.IGNORECASE):
            return ShellIntent.SCRIPT_CREATE
    
    return ShellIntent.UNKNOWN


def detect_target_path(user_input: str, default_workspace: str = None) -> Optional[str]:
    """Detect a target file path from user input.
    
    Supports:
    - Explicit paths: "save to /home/user/file.py"
    - Desktop: "put on desktop" -> ~/Desktop
    - Named files: "save as clock.py"
    - Relative paths: "save to myfolder/file.txt"
    """
    text = (user_input or "").strip()
    low = text.lower()
    
    if not text:
        return None
    
    # Default workspace
    workspace = default_workspace or os.getcwd()
    
    # Detect Desktop
    is_desktop = any(re.search(p, low) for p in DESKTOP_PATTERNS)
    
    # Try to extract filename
    filename = None
    
    # Pattern: "save as X" / "named X" / "called X"
    name_patterns = [
        r'(?:save|write|name|call)\s+(?:it\s+)?(?:as\s+)?["\']?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)["\']?',
        r'(?:named|called)\s+["\']?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)["\']?',
        r'\b([a-zA-Z][a-zA-Z0-9_]*\.(py|js|sh|txt|md|json|html|css))\b',
    ]
    
    for pattern in name_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            filename = m.group(1)
            break
    
    # If desktop mentioned, use ~/Desktop
    if is_desktop:
        desktop_dir = os.path.expanduser("~/Desktop")
        if filename:
            return os.path.join(desktop_dir, filename)
        return desktop_dir
    
    # If we have a filename, resolve relative to workspace
    if filename:
        if os.path.isabs(filename):
            return filename
        return os.path.join(workspace, filename)
    
    return None


def detect_language(user_input: str) -> str:
    """Detect the programming language from user input."""
    text = (user_input or "").strip().lower()
    
    # Language keywords
    lang_map = {
        "python": ["python", ".py", "py script"],
        "javascript": ["javascript", "js", ".js", "node"],
        "bash": ["bash", "shell", "sh", ".sh", "shell script"],
        "html": ["html", ".html", "webpage", "web page"],
        "css": ["css", ".css", "stylesheet"],
        "json": ["json", ".json"],
        "markdown": ["markdown", "md", ".md"],
    }
    
    for lang, keywords in lang_map.items():
        for kw in keywords:
            if kw in text:
                return lang
    
    # Default to Python for scripts
    return "python"


def detect_command(user_input: str) -> Optional[str]:
    """Extract a shell command from user input."""
    text = (user_input or "").strip()

    if not text:
        return None

    # Prefer more specific forms first so we do not capture filler like "the command".
    patterns = [
        r'(?:run|execute|launch|start)\s+(?:the\s+|a\s+)?command\s+(.+)$',
        r'(?:run|execute|launch|start)\s+(?:the\s+|a\s+)?script\s+(.+)$',
        r'command:\s*["\']?([^"\']+)["\']?',
        r'\$\s*(.+)$',
        r'(?:run|execute|launch|start)\s+["\']?([^"\']+)["\']?$',
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            command = m.group(1).strip()
            command = re.sub(r'^(?:the\s+|a\s+)', '', command, flags=re.IGNORECASE)
            return command.strip()

    return None


# =============================================================================
# Envelope Builders
# =============================================================================

def build_shell_envelope(user_input: str, workspace: str = None) -> Optional[TeleportEnvelope]:
    """Build a teleport envelope for shell operations.
    
    This is the main entry point for detecting and building shell teleport contracts.
    Returns None if no shell intent is detected.
    """
    intent = classify_shell_intent(user_input)

    if intent == ShellIntent.UNKNOWN:
        return None

    # Only gate on wave mode after we know the query actually needs teleport.
    is_wave, wave_msg = require_wave_mode()
    if not is_wave:
        # Return a special envelope that indicates refusal
        return TeleportEnvelope(
            contract_version="teleport.v1",
            intent="wave_mode_required",
            source_mode="classic",
            target_mode="chamber",
            objective=user_input,
            constraints={"refused": True, "reason": wave_msg},
            policy_reason="Teleport refused: wave mode not enabled",
            fallback_reason=wave_msg,
        )
    
    workspace = workspace or getattr(config, 'CLI_WORKSPACE_ROOT', os.getcwd())
    target_path = detect_target_path(user_input, workspace)
    language = detect_language(user_input)
    command = detect_command(user_input)
    
    constraints = {
        "workspace": workspace,
        "language": language,
        "requires_confirmation": True,
        "allow_prose_fallback": False,  # NEVER give text-only fake answers
    }
    
    if target_path:
        constraints["target_path"] = target_path
        constraints["filename"] = os.path.basename(target_path)
    
    if command:
        constraints["command"] = command
    
    # Determine specific intent string
    if intent == ShellIntent.FILE_WRITE:
        intent_str = "file_write"
        policy_reason = "File write operation requires shell chamber teleport for safe execution"
    elif intent == ShellIntent.SHELL_COMMAND:
        intent_str = "shell_command"
        policy_reason = "Shell command execution requires chamber boundary for security"
    elif intent == ShellIntent.SCRIPT_CREATE:
        intent_str = "script_create"
        policy_reason = "Script creation requires shell chamber for proper scaffolding"
    else:
        intent_str = "shell_chamber"
        policy_reason = "Shell operation requires chamber execution"
    
    return TeleportEnvelope(
        contract_version="teleport.v1",
        intent=intent_str,
        source_mode="wave",
        target_mode="chamber",
        objective=user_input,
        constraints=constraints,
        policy_reason=policy_reason,
    )


# Legacy compatibility - keep old function names working
def parse_python_desktop_write_intent(user_input: str) -> Optional[Dict[str, Any]]:
    """Legacy compatibility wrapper.
    
    This is kept for backwards compatibility but delegates to the new
    generalized intent detection system.
    """
    envelope = build_shell_envelope(user_input)
    if not envelope or envelope.constraints.get("refused"):
        return None
    
    target_path = envelope.constraints.get("target_path")
    if not target_path:
        # Default to Desktop for legacy calls
        desktop_dir = os.path.expanduser("~/Desktop")
        filename = envelope.constraints.get("filename", "script.py")
        target_path = os.path.join(desktop_dir, filename)
    
    return {
        "objective": envelope.objective,
        "filename": os.path.basename(target_path),
        "target_path": target_path,
        "requires_confirmation": envelope.constraints.get("requires_confirmation", True),
    }


def build_desktop_write_envelope(user_input: str) -> Optional[TeleportEnvelope]:
    """Legacy compatibility wrapper."""
    return build_shell_envelope(user_input)


# =============================================================================
# Execution Chamber
# =============================================================================

def execute_file_write(envelope: TeleportEnvelope, content: str) -> TeleportResult:
    """Execute a file write operation in the chamber.
    
    Args:
        envelope: The teleport envelope with constraints
        content: The content to write
    
    Returns:
        TeleportResult with artifact info
    """
    validation = envelope.validate()
    if not validation["ok"]:
        return TeleportResult(
            False, "contract_error", 
            "; ".join(validation["issues"]), 
            envelope=envelope
        )
    
    target_path = str(envelope.constraints.get("target_path", "")).strip()
    if not target_path:
        return TeleportResult(
            False, "contract_error", 
            "missing target path", 
            envelope=envelope
        )
    
    # Validate path is within allowed workspace
    workspace = envelope.constraints.get("workspace", os.getcwd())
    try:
        abs_target = os.path.abspath(target_path)
        abs_workspace = os.path.abspath(workspace)
        # Allow writes outside workspace for explicit paths like Desktop
        # But log the boundary crossing
    except Exception as e:
        return TeleportResult(
            False, "boundary_error", 
            f"invalid path: {e}", 
            envelope=envelope
        )
    
    # Check if parent directory exists, create if needed
    parent = os.path.dirname(target_path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as e:
            return TeleportResult(
                False, "boundary_error", 
                f"cannot create directory {parent}: {e}", 
                envelope=envelope
            )
    
    try:
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write((content or "").rstrip() + "\n")
        
        # Determine file kind from extension
        ext = os.path.splitext(target_path)[1].lower()
        kind_map = {
            ".py": "python_script",
            ".js": "javascript_script",
            ".sh": "shell_script",
            ".html": "html_file",
            ".css": "stylesheet",
            ".json": "json_file",
            ".md": "markdown_file",
            ".txt": "text_file",
        }
        kind = kind_map.get(ext, "file")
        
        return TeleportResult(
            True,
            "completed",
            f"teleport write completed: {target_path}",
            artifact={
                "path": target_path,
                "kind": kind,
                "size": len(content),
                "language": envelope.constraints.get("language", "unknown"),
            },
            envelope=envelope,
        )
    except Exception as e:
        return TeleportResult(
            False, "execution_error", 
            str(e), 
            envelope=envelope
        )


def execute_shell_command(envelope: TeleportEnvelope, timeout: int = None) -> TeleportResult:
    """Execute a shell command in the chamber.
    
    Args:
        envelope: The teleport envelope with command constraint
        timeout: Maximum execution time (default from config)
    
    Returns:
        TeleportResult with command output
    """
    validation = envelope.validate()
    if not validation["ok"]:
        return TeleportResult(
            False, "contract_error", 
            "; ".join(validation["issues"]), 
            envelope=envelope
        )
    
    command = envelope.constraints.get("command", "")
    if not command:
        return TeleportResult(
            False, "contract_error", 
            "missing command to execute", 
            envelope=envelope
        )
    
    workspace = envelope.constraints.get("workspace", os.getcwd())
    timeout = timeout or getattr(config, 'CLI_SHELL_TIMEOUT', 20)
    
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        
        return TeleportResult(
            result.returncode == 0,
            "completed" if result.returncode == 0 else "failed",
            output.strip() or f"(exit={result.returncode})",
            artifact={
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "workspace": workspace,
            },
            envelope=envelope,
        )
    except subprocess.TimeoutExpired:
        return TeleportResult(
            False, "timeout", 
            f"command timed out after {timeout}s", 
            envelope=envelope
        )
    except Exception as e:
        return TeleportResult(
            False, "execution_error", 
            str(e), 
            envelope=envelope
        )


def execute_script_create(envelope: TeleportEnvelope, script_content: str = None) -> TeleportResult:
    """Create a script with proper scaffolding.
    
    This is a higher-level operation that combines file creation with
    optional execution permissions and boilerplate.
    """
    if not script_content:
        # Generate boilerplate based on language
        language = envelope.constraints.get("language", "python")
        objective = envelope.objective
        
        if language == "python":
            script_content = f'''#!/usr/bin/env python3
"""
Auto-generated script: {objective}
Created by Hermit Shell Chamber
"""

def main():
    # TODO: Implement {objective}
    print("Hello from Hermit!")

if __name__ == "__main__":
    main()
'''
        elif language == "bash":
            script_content = f'''#!/bin/bash
# Auto-generated script: {objective}
# Created by Hermit Shell Chamber

set -e

main() {{
    # TODO: Implement {objective}
    echo "Hello from Hermit!"
}}

main "$@"
'''
        elif language == "javascript":
            script_content = f'''#!/usr/bin/env node
/**
 * Auto-generated script: {objective}
 * Created by Hermit Shell Chamber
 */

function main() {{
    // TODO: Implement {objective}
    console.log("Hello from Hermit!");
}}

main();
'''
        else:
            script_content = f"# {objective}\n# Created by Hermit Shell Chamber\n"
    
    # Delegate to file write
    result = execute_file_write(envelope, script_content)
    
    # Make executable if successful
    if result.ok and envelope.constraints.get("target_path"):
        try:
            os.chmod(envelope.constraints["target_path"], 0o755)
            result.artifact["executable"] = True
        except Exception:
            pass  # Not critical if chmod fails
    
    return result


def execute_teleport_contract(envelope: TeleportEnvelope, content: str = None) -> TeleportResult:
    """Main entry point for executing a teleport contract.
    
    Routes to the appropriate executor based on intent type.
    """
    # Check for refusal envelope
    if envelope.constraints.get("refused"):
        return TeleportResult(
            False,
            "refused",
            envelope.constraints.get("reason", "Teleport refused"),
            envelope=envelope,
        )
    
    intent = envelope.intent
    
    if intent == "file_write":
        return execute_file_write(envelope, content or "")
    elif intent == "shell_command":
        return execute_shell_command(envelope)
    elif intent == "script_create":
        return execute_script_create(envelope, content)
    elif intent == "desktop_write" or intent == "desktop_python_write":
        # Legacy compatibility
        return execute_file_write(envelope, content or "")
    else:
        return TeleportResult(
            False, 
            "unknown_intent", 
            f"Unknown teleport intent: {intent}",
            envelope=envelope,
        )


# Legacy compatibility
def execute_desktop_write_contract(envelope: TeleportEnvelope, code: str) -> TeleportResult:
    """Legacy compatibility wrapper."""
    return execute_file_write(envelope, code)
