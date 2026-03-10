
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import cmd
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional
import libzim

from chatbot.rag import RAGSystem, TextProcessor
from chatbot import config
from chatbot.chat import build_messages, build_messages_with_intent, stream_chat, clear_runtime_memory, load_runtime_checkpoint, save_runtime_checkpoint
from chatbot.model_manager import ModelManager
from chatbot.models import Message
from chatbot.teleport import (
    execute_teleport_contract,
    execute_shell_command,
    TeleportEnvelope,
    TeleportResult,
    wave_mode_enabled,
    detect_command,
    detect_target_path,
)


class ChatbotCLI(cmd.Cmd):
    """Command-line interface for Hermit."""

    intro = 'Welcome to Hermit CLI. Type help or ? to list commands.\n'
    prompt = '(hermit) '

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.rag = None
        self.last_results = []
        self.workspace_root = Path(getattr(config, 'CLI_WORKSPACE_ROOT', '.')).resolve()
        self.cwd = self.workspace_root

        print(f"Initializing RAG System (Model: {model_name})...")
        try:
            self.rag = RAGSystem()

            # Inject our RAG instance into the chat module so it doesn't try to reload it
            import chatbot.chat
            chatbot.chat._rag_system = self.rag

            print("RAG System Ready.")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            print("Some functionality may be limited.")

        self.history = []
        self._refresh_prompt()

    def _refresh_prompt(self) -> None:
        rel = os.path.relpath(self.cwd, self.workspace_root)
        label = "." if rel == "." else rel
        self.prompt = f'(hermit:{label}) '

    def _resolve_workspace_path(self, raw: str = "") -> Path:
        candidate = (raw or ".").strip()
        path = Path(candidate)
        if not path.is_absolute():
            path = self.cwd / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError(f"Path escapes workspace root: {resolved}")
        return resolved

    def _format_path(self, path: Path) -> str:
        rel = os.path.relpath(path, self.workspace_root)
        return "." if rel == "." else rel

    def _print_excursion(self, title: str, body: str) -> None:
        print(f"\n=== {title} ===")
        print(body.rstrip())
        print("=" * (len(title) + 8))

    def do_pwd(self, arg):
        """Show the current chamber workspace path."""
        print(self._format_path(self.cwd))

    def do_cd(self, arg):
        """Change chamber workspace: cd <path>"""
        try:
            target = self._resolve_workspace_path(arg or ".")
            if not target.exists():
                print(f"Directory not found: {target}")
                return
            if not target.is_dir():
                print(f"Not a directory: {target}")
                return
            self.cwd = target
            self._refresh_prompt()
            print(f"Entered chamber: {self._format_path(self.cwd)}")
        except Exception as e:
            print(f"cd failed: {e}")

    def do_ls(self, arg):
        """List files in the current or specified chamber path: ls [path]"""
        try:
            target = self._resolve_workspace_path(arg or ".")
            if not target.exists():
                print(f"Path not found: {target}")
                return
            if target.is_file():
                print(self._format_path(target))
                return

            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if not entries:
                print("(empty)")
                return
            for entry in entries:
                suffix = "/" if entry.is_dir() else ""
                print(f"{entry.name}{suffix}")
        except Exception as e:
            print(f"ls failed: {e}")

    def do_cat(self, arg):
        """Read a local file inside the workspace: cat <path>"""
        if not arg:
            print("Usage: cat <path>")
            return
        try:
            target = self._resolve_workspace_path(arg)
            if not target.exists() or not target.is_file():
                print(f"File not found: {target}")
                return
            data = target.read_text(encoding='utf-8', errors='replace')
            limit = int(getattr(config, 'CLI_MAX_FILE_READ_CHARS', 12000) or 12000)
            if len(data) > limit:
                data = data[:limit] + "\n...[truncated]"
            self._print_excursion(f"FILE {self._format_path(target)}", data)
        except Exception as e:
            print(f"cat failed: {e}")

    def do_write(self, arg):
        """Write a local file in one shot: write <path> ::: <content>"""
        if not getattr(config, 'CLI_WRITE_ENABLED', True):
            print("File writes are disabled in this build.")
            return
        if ":::" not in arg:
            print("Usage: write <path> ::: <content>")
            return
        raw_path, content = arg.split(":::", 1)
        try:
            target = self._resolve_workspace_path(raw_path.strip())
            text = content.lstrip()
            limit = int(getattr(config, 'CLI_MAX_FILE_WRITE_CHARS', 20000) or 20000)
            if len(text) > limit:
                print(f"Refusing write larger than {limit} chars")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
            print(f"Wrote {len(text)} chars to {self._format_path(target)}")
        except Exception as e:
            print(f"write failed: {e}")

    def do_shell(self, arg):
        """Run a bounded shell excursion in the current chamber: shell <command>"""
        if not getattr(config, 'CLI_SHELL_ENABLED', True):
            print("Shell excursions are disabled in this build.")
            return
        command = (arg or "").strip()
        if not command:
            print("Usage: shell <command>")
            return
        try:
            timeout = int(getattr(config, 'CLI_SHELL_TIMEOUT', 20) or 20)
            result = subprocess.run(
                command,
                cwd=str(self.cwd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            body = ""
            if result.stdout:
                body += result.stdout.rstrip() + "\n"
            if result.stderr:
                body += "\n[stderr]\n" + result.stderr.rstrip() + "\n"
            body += f"\n(exit={result.returncode})"
            self._print_excursion(f"SHELL {command}", body or f"(exit={result.returncode})")
        except subprocess.TimeoutExpired:
            print("shell failed: timed out")
        except Exception as e:
            print(f"shell failed: {e}")

    def do_chamber(self, arg):
        """Show chamber state, available excursions, and current residue."""
        checkpoint = load_runtime_checkpoint()
        print("\nHermit Chamber")
        print("==============")
        print(f"Workspace root: {self.workspace_root}")
        print(f"Current chamber: {self.cwd}")
        print(f"Runtime mode: {getattr(config, 'RUNTIME_MODE', 'classic')}")
        print("Excursions: ls, cd, pwd, cat, write, shell, checkpoint")
        if checkpoint.get('objective') or checkpoint.get('artifacts'):
            print("\nResidue snapshot:")
            print(f"- Objective: {checkpoint.get('objective', '')}")
            print(f"- Frontier: {', '.join(checkpoint.get('frontier', [])[:4]) or '(none)'}")
            print(f"- Risk: {checkpoint.get('risk', 'unknown')}")
            print(f"- Next step: {checkpoint.get('next_step', '')}")
            artifacts = checkpoint.get('artifacts', []) or []
            if artifacts:
                print(f"- Artifacts: {len(artifacts)}")
        print("")

    def do_checkpoint(self, arg):
        """Print the current contracted-cognition checkpoint."""
        checkpoint = load_runtime_checkpoint()
        self._print_excursion("RUNTIME CHECKPOINT", str(checkpoint))

    def do_clear(self, arg):
        """Clear in-session conversation memory."""
        self.history.clear()
        clear_runtime_memory(reset_rag=False)
        print("Session memory cleared.")

    def do_mode(self, arg):
        """Show or change runtime mode: mode [classic|wave]"""
        mode = (arg or "").strip().lower()
        if not mode:
            print(f"Current runtime mode: {getattr(config, 'RUNTIME_MODE', 'classic')}")
            return
        if mode not in {"classic", "wave"}:
            print("Usage: mode classic | mode wave")
            return
        previous = getattr(config, 'RUNTIME_MODE', 'classic')
        config.RUNTIME_MODE = mode
        self.history.clear()
        clear_runtime_memory(reset_rag=False)
        ModelManager.close_all()
        print(f"Runtime mode set to {mode} (was {previous}). Session memory cleared.")

    def do_tree(self, arg):
        """Show the current runtime model tree."""
        slots = [
            ("Final synthesis", "DEFAULT_MODEL"),
            ("Entity extraction", "ENTITY_JOINT_MODEL"),
            ("Article scoring", "SCORER_JOINT_MODEL"),
            ("Chunk filtering", "FILTER_JOINT_MODEL"),
            ("Fact extraction", "FACT_JOINT_MODEL"),
            ("Refinement", "REFINEMENT_JOINT_MODEL"),
            ("Multi-hop reasoning", "MULTI_HOP_JOINT_MODEL"),
            ("Comparison synthesis", "COMPARISON_JOINT_MODEL"),
        ]
        mode = getattr(config, 'RUNTIME_MODE', 'classic')
        print("\nHermit Runtime Tree")
        print("===================")
        print(f"Mode: {mode}")
        print("User Query")
        print("  ├─ Entity / Score / Filter")
        print("  ├─ Fact / Refine / Multi-hop / Compare")
        print("  └─ Final synthesis\n")
        if mode == 'wave':
            selected = getattr(config, 'DEFAULT_MODEL', '(unset)')
            for label, _attr in slots:
                print(f"- {label}: {selected} [wave]")
        else:
            for label, attr in slots:
                print(f"- {label}: {getattr(config, attr, '(unset)')}")
        print("")

    def do_status(self, arg):
        """Show backend, RAG, and orchestration status."""
        backend_type = "External API" if config.API_MODE else "Local (GGUF)"
        backend_detail = f"URL: {config.API_BASE_URL}" if config.API_MODE else "Engine: llama-cpp-python"

        print("\nHermit Status")
        print("=============")
        print(f"Backend: {backend_type}")
        print(backend_detail)
        print(f"Runtime mode: {getattr(config, 'RUNTIME_MODE', 'classic')}")
        print(f"Model: {self.model_name}")
        print(f"Workspace: {self._format_path(self.cwd)}")

        if not self.rag:
            print("\nRAG: Inactive")
            print("Orchestration: Unavailable")
            print("")
            return

        count_docs = len(self.rag.indexed_paths) if getattr(self.rag, 'indexed_paths', None) else 0
        count_chunks = len(self.rag.doc_chunks) if getattr(self.rag, 'doc_chunks', None) else 0

        print("\nRAG: Active")
        print(f"JIT Index: {count_docs} articles ({count_chunks} chunks)")
        print(f"Encoder: {self.rag.model_name}")
        if getattr(self.rag, 'faiss_index', None):
            print(f"Vectors: {self.rag.faiss_index.ntotal}")

        ostatus = getattr(self.rag, 'last_orchestration_status', {}) or {}
        print("\nOrchestration:")
        if not ostatus:
            print("No orchestration run yet in this session.")
        else:
            print(f"- Goal: {ostatus.get('active_goal', 'n/a')}")
            print(f"- Route: {ostatus.get('mode', 'n/a')} ({ostatus.get('routing_reason', 'n/a')})")
            print(f"- Risk: {ostatus.get('ofr_risk', 'unknown')}")
            print(f"- Steps: {ostatus.get('steps_executed', 0)} executed / {ostatus.get('steps_remaining', 0)} remaining")
            print(f"- Residue/events: {ostatus.get('residue_count', 0)}/{ostatus.get('events_count', 0)}")
            artifact_summary = ostatus.get('artifact_summary', {})
            print(f"- Excursions: {artifact_summary.get('count', 0)}")
        print("")

    def do_search(self, arg):
        """Search for articles: search <query>"""
        if not arg:
            print("Usage: search <query>")
            return

        print(f"Searching for '{arg}'...")
        if not self.rag:
            print("RAG system not available.")
            return

        try:
            results = self.rag.retrieve(arg, top_k=10)
            self.last_results = results

            if not results:
                print("No results found.")
                return

            print(f"\nFound {len(results)} results:")
            for i, res in enumerate(results, 1):
                meta = res.get('metadata', {})
                title = meta.get('title', 'Unknown')
                path = meta.get('path', 'Unknown')
                score = res.get('score', 0.0)
                print(f"{i}. {title} (Score: {score:.2f}) [Path: {path}]")
            print("\nType 'read <number>' or 'read <path>' to view an article.")

        except Exception as e:
            print(f"Search failed: {e}")

    def do_read(self, arg):
        """Read an article: read <number> or read <path>"""
        if not arg:
            print("Usage: read <result_number> or read <path>")
            return

        path_to_read = None

        if arg.isdigit() and self.last_results:
            idx = int(arg) - 1
            if 0 <= idx < len(self.last_results):
                path_to_read = self.last_results[idx]['metadata'].get('path')
                print(f"Selected result #{arg}: {path_to_read}")
            else:
                print(f"Invalid index. Valid range: 1-{len(self.last_results)}")
                return
        else:
            path_to_read = arg

        if not path_to_read:
            print("No path resolved.")
            return

        terms = []
        if self.last_results and arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(self.last_results):
                ctx = self.last_results[idx].get('search_context', {})
                terms = ctx.get('entities', [])

        self._open_zim_entry(path_to_read, highlight_terms=terms)

    def _open_zim_entry(self, path, highlight_terms=None):
        """Open ZIM entry with smart fallback logic."""
        zim_files = [f for f in os.listdir('.') if f.endswith('.zim')]
        if not zim_files:
            print("Error: No ZIM files found in current directory.")
            return

        zim_file = zim_files[0]
        try:
            zim = libzim.Archive(zim_file)
        except Exception as e:
            print(f"Error opening ZIM archive: {e}")
            return

        entry = None

        def try_find(p):
            try:
                return zim.get_entry_by_path(p)
            except Exception:
                return None

        entry = try_find(path)

        if not entry:
            try:
                entry = zim.get_entry_by_title(path)
            except Exception:
                pass

        if not entry:
            variations = []
            if ' ' in path:
                variations.append(path.replace(' ', '_'))
            if '_' in path:
                variations.append(path.replace('_', ' '))
            variations.append(path.title())
            if ' ' in path:
                variations.append(path.title().replace(' ', '_'))

            paths_to_try = [path] + variations
            for candidate in paths_to_try:
                attempts = [candidate]
                if not candidate.startswith('/'):
                    attempts.append('/' + candidate)
                if candidate.startswith('/'):
                    attempts.append(candidate[1:])

                for attempt in attempts:
                    if config.DEBUG:
                        print(f"[DEBUG] Trying: {attempt}")
                    entry = try_find(attempt)
                    if entry:
                        break
                if entry:
                    break

        if not entry or entry.is_redirect:
            print(f"Article not found: '{path}'")
            return

        item = entry.get_item()
        if item.mimetype != 'text/html':
            print(f"Cannot render non-text content ({item.mimetype})")
            return

        print(f"\n=== {entry.title} ===\n")
        try:
            content = TextProcessor.extract_renderable_text(item.content)

            if highlight_terms:
                import re
                start_hl = "\033[1;33m"
                end_hl = "\033[0m"

                for term in highlight_terms:
                    if not term or len(term) < 3:
                        continue
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    content = pattern.sub(lambda m: f"{start_hl}{m.group()}{end_hl}", content)

            print(content)
            print("\n" + "=" * 40 + "\n")
        except Exception as e:
            print(f"Error rendering content: {e}")

    def do_quit(self, arg):
        """Exit the CLI."""
        self.history.clear()
        clear_runtime_memory(reset_rag=True)
        print("Goodbye!")
        return True

    def do_exit(self, arg):
        """Exit the CLI."""
        return self.do_quit(arg)

    def _record_chamber_artifact(self, envelope, status: str, note: str, payload: Optional[dict] = None):
        """Persist a compact teleport artifact into the runtime checkpoint."""
        checkpoint = load_runtime_checkpoint()
        artifacts = list(checkpoint.get("artifacts", []) or [])
        artifacts.append({
            "mode": envelope.intent,
            "step": "teleport",
            "query": envelope.objective,
            "status": status,
            "note": note,
            "payload": payload or {},
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

    def _execute_teleport(self, envelope):
        """Execute a teleport envelope and return the formatted result string."""
        intent = envelope.intent
        workspace = str(self.cwd)
        envelope.constraints["workspace"] = workspace

        if intent == "shell_command":
            # If no command was auto-detected, try extracting from objective
            if not envelope.constraints.get("command"):
                cmd_guess = detect_command(envelope.objective)
                if cmd_guess:
                    envelope.constraints["command"] = cmd_guess
                else:
                    # The objective itself might be the command
                    envelope.constraints["command"] = envelope.objective
            result = execute_teleport_contract(envelope)
        elif intent in ("file_write", "script_create"):
            # For file writes, we need the LLM to generate the content first
            # Return None to signal "needs content generation"
            return None
        else:
            result = execute_teleport_contract(envelope)

        if result.ok:
            output = result.message or "(no output)"
            artifact = result.artifact or {}
            self._record_chamber_artifact(
                envelope,
                "ok",
                f"command completed (exit={artifact.get('exit_code', 0)})",
                payload=artifact,
            )
            return f"✅ Command executed successfully (exit={artifact.get('exit_code', 0)})\n{output}"
        else:
            self._record_chamber_artifact(
                envelope,
                "error",
                f"execution failed: {result.status}",
                payload=result.artifact or {"message": result.message},
            )
            return f"❌ Execution failed: {result.status} — {result.message}"

    def _agentic_generate(self, messages):
        """Run LLM generation and return the full response text."""
        full_response = ""
        for chunk in stream_chat(self.model_name, messages):
            print(chunk, end="", flush=True)
            full_response += chunk
        return full_response

    def _run_follow_up_commands(self, initial_response: str, max_rounds: int = 5) -> str:
        """Execute chained [HERMIT_CMD] commands until no further command is requested."""
        full_response = initial_response
        rounds = 0

        while rounds < max_rounds:
            follow_up_cmd = self._extract_agent_command(full_response)
            if not follow_up_cmd:
                break

            rounds += 1
            print(f"\n📡 Follow-up command: {follow_up_cmd}")
            follow_envelope = TeleportEnvelope(
                contract_version="teleport.v1",
                intent="shell_command",
                source_mode="wave",
                target_mode="chamber",
                objective=follow_up_cmd,
                constraints={
                    "workspace": str(self.cwd),
                    "command": follow_up_cmd,
                    "requires_confirmation": False,
                    "allow_prose_fallback": False,
                },
            )
            follow_result = self._execute_teleport(follow_envelope)
            if follow_result is None:
                break

            print(follow_result)
            observation = (
                f"[SHELL CHAMBER OBSERVATION]\n"
                f"Follow-up command: {follow_up_cmd}\n"
                f"Result:\n{follow_result}\n"
                f"[/SHELL CHAMBER OBSERVATION]\n\n"
                f"Continue summarizing. If you need another command, use [HERMIT_CMD]...[/HERMIT_CMD]."
            )
            self.history.append(Message(role="assistant", content=full_response))
            self.history.append(Message(role="user", content=observation))
            messages = build_messages(config.SYSTEM_PROMPT, self.history)

            print(f"\nHermit: ", end="", flush=True)
            full_response = self._agentic_generate(messages)
            print("\n")

        return full_response

    def _extract_agent_command(self, response):
        """Extract a [HERMIT_CMD]...[/HERMIT_CMD] block from LLM output.

        Returns the command string if found, else None.
        """
        import re
        match = re.search(r'\[HERMIT_CMD\](.+?)\[/HERMIT_CMD\]', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_file_content(self, response, language: str = None, require_marker: bool = False):
        """Extract file content from LLM output.

        Looks for:
        1. Content in [HERMIT_FILE]...[/HERMIT_FILE] tags
        2. Content in code fences (```lang ... ```)
        3. (Optional) the entire response if no markers found

        Args:
            response: Raw model response
            language: Optional expected language for fenced blocks
            require_marker: If True, refuse unmarked prose fallback

        Returns:
            (content, found_marker) - extracted content and whether a marker was found
        """
        import re

        # Try [HERMIT_FILE] tags first (explicit marker)
        file_match = re.search(r'\[HERMIT_FILE\](.+?)\[/HERMIT_FILE\]', response, re.DOTALL)
        if file_match:
            return file_match.group(1).strip(), True

        # Try code fences with language hint
        fence_pattern = r'```(?:' + (re.escape(language) if language else r'\w*') + r')?\s*\n(.+?)\n```'
        fence_match = re.search(fence_pattern, response, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip(), True

        # Try any code fence
        any_fence = re.search(r'```\w*\n(.+?)\n```', response, re.DOTALL)
        if any_fence:
            return any_fence.group(1).strip(), True

        # No markers - optionally allow best-effort fallback
        if require_marker:
            return None, False

        lines = response.strip().split('\n')
        if len(lines) > 1 or any(c in response for c in '{}[]()=;:'):
            return response.strip(), False

        return None, False

    def _execute_file_write_from_response(self, envelope, response):
        """Execute a file/script creation using LLM-generated content.

        Args:
            envelope: The teleport envelope with target_path constraint
            response: The LLM's response containing the file content

        Returns:
            Formatted result string or None if no content found
        """
        language = envelope.constraints.get("language", "")
        content, found_marker = self._extract_file_content(response, language, require_marker=True)

        if not content:
            return None

        # Ensure we have a target path
        target_path = envelope.constraints.get("target_path")
        if not target_path:
            # Generate a default filename
            import os
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
            stem = "generated_script" if envelope.intent == "script_create" else "generated_file"
            target_path = os.path.join(str(self.cwd), f"{stem}{ext}")
            envelope.constraints["target_path"] = target_path

        # Route through contract executor so script_create gets executable scaffolding behavior.
        result = execute_teleport_contract(envelope, content)

        if result.ok:
            artifact = result.artifact or {}
            kind_label = "script" if envelope.intent == "script_create" else "file"
            self._record_chamber_artifact(
                envelope,
                "ok",
                f"{kind_label} written: {artifact.get('path', target_path)}",
                payload=artifact,
            )
            executable = "\nExecutable: yes" if artifact.get("executable") else ""
            return (
                f"✅ {kind_label.capitalize()} written successfully\n"
                f"Path: {artifact.get('path', target_path)}\n"
                f"Size: {artifact.get('size', len(content))} chars\n"
                f"Type: {artifact.get('kind', 'file')}"
                f"{executable}"
            )
        else:
            self._record_chamber_artifact(
                envelope,
                "error",
                f"{envelope.intent} failed: {result.status}",
                payload=result.artifact or {"message": result.message},
            )
            return f"❌ {envelope.intent} failed: {result.status} — {result.message}"

    def _file_generation_contract(self, envelope: TeleportEnvelope) -> str:
        """Build strict output instructions for file/script creation flows."""
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

    def default(self, line):
        """Handle chat interactions — with agentic teleport in wave mode."""
        if not line:
            return

        if line.strip().startswith('#'):
            return

        self.history.append(Message(role="user", content=line))

        print(f"\nThinking...")
        try:
            messages, intent = build_messages_with_intent(config.SYSTEM_PROMPT, self.history)

            # === AGENTIC TELEPORT PATH (wave mode) ===
            if (
                wave_mode_enabled()
                and intent.shell_intent
                and intent.teleport_envelope
                and not intent.teleport_envelope.constraints.get("refused")
            ):
                envelope = intent.teleport_envelope
                teleport_result = self._execute_teleport(envelope)

                if teleport_result is not None:
                    # We executed a command — feed the real output back to the LLM
                    print(f"\n📡 Teleport [{envelope.intent}]\n")
                    print(teleport_result)

                    # Inject the excursion result as context for LLM synthesis
                    observation_msg = (
                        f"[SHELL CHAMBER OBSERVATION]\n"
                        f"Intent: {envelope.intent}\n"
                        f"Command: {envelope.constraints.get('command', envelope.objective)}\n"
                        f"Working directory: {self.cwd}\n"
                        f"Result:\n{teleport_result}\n"
                        f"[/SHELL CHAMBER OBSERVATION]\n\n"
                        f"Summarize the result for the user. If you need to run another command, "
                        f"output it inside [HERMIT_CMD]command here[/HERMIT_CMD] tags."
                    )
                    self.history.append(Message(role="assistant", content=f"[Executed: {envelope.constraints.get('command', '')}]"))
                    self.history.append(Message(role="user", content=observation_msg))
                    messages = build_messages(config.SYSTEM_PROMPT, self.history)

                    print(f"\nHermit: ", end="", flush=True)
                    full_response = self._agentic_generate(messages)
                    print("\n")

                    full_response = self._run_follow_up_commands(full_response, max_rounds=5)
                    self.history.append(Message(role="assistant", content=full_response))
                    return

                # teleport_result is None → file_write/script_create needs LLM-generated content
                # Fall through to normal generation, the system prompt already has instructions

            # === NORMAL CHAT PATH ===
            if (
                wave_mode_enabled()
                and intent.shell_intent in ("file_write", "script_create")
                and intent.teleport_envelope
                and not intent.teleport_envelope.constraints.get("refused")
                and messages
            ):
                messages = [dict(m) for m in messages]
                messages[0]["content"] = messages[0].get("content", "") + self._file_generation_contract(intent.teleport_envelope)

            print(f"Hermit: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chat(self.model_name, messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            # In wave mode, check if the LLM output contains a command to execute
            if wave_mode_enabled():
                agent_cmd = self._extract_agent_command(full_response)
                if agent_cmd:
                    print(f"\n📡 Agent command: {agent_cmd}")
                    auto_envelope = TeleportEnvelope(
                        contract_version="teleport.v1",
                        intent="shell_command",
                        source_mode="wave",
                        target_mode="chamber",
                        objective=agent_cmd,
                        constraints={
                            "workspace": str(self.cwd),
                            "command": agent_cmd,
                            "requires_confirmation": False,
                            "allow_prose_fallback": False,
                        },
                    )
                    cmd_result = self._execute_teleport(auto_envelope)
                    if cmd_result:
                        print(cmd_result)
                        self.history.append(Message(role="assistant", content=full_response))
                        self.history.append(Message(role="user", content=f"[SHELL CHAMBER OBSERVATION]\n{cmd_result}\n[/SHELL CHAMBER OBSERVATION]"))
                        messages = build_messages(config.SYSTEM_PROMPT, self.history)
                        print(f"\nHermit: ", end="", flush=True)
                        full_response = self._agentic_generate(messages)
                        print("\n")
                        full_response = self._run_follow_up_commands(full_response, max_rounds=5)
                        self.history.append(Message(role="assistant", content=full_response))
                        return

                # If this was a file_write intent with an envelope, execute the write now
                if intent.shell_intent in ("file_write", "script_create") and intent.teleport_envelope:
                    write_result = self._execute_file_write_from_response(intent.teleport_envelope, full_response)
                    if write_result:
                        print(f"\n📡 Teleport [{intent.shell_intent}]")
                        print(write_result)
                        # Add a summary to history
                        summary = f"{full_response}\n\n[System: {write_result}]"
                        self.history.append(Message(role="assistant", content=summary))
                        return
                    else:
                        print("\n❌ File write canceled: model response did not include a valid [HERMIT_FILE] block or code fence.")
                        self.history.append(Message(role="assistant", content=full_response))
                        return

            self.history.append(Message(role="assistant", content=full_response))

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"\nError: {e}")

    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        self.history.clear()
        clear_runtime_memory(reset_rag=True)
        print("")
        return True


if __name__ == '__main__':
    cli = ChatbotCLI(config.DEFAULT_MODEL)
    cli.cmdloop()
