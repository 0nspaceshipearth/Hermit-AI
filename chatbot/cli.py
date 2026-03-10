
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
from typing import Any, Dict, List, Optional
import libzim

from chatbot.rag import RAGSystem, TextProcessor
from chatbot import config
from chatbot.chat import build_messages, build_messages_with_intent, stream_chat, clear_runtime_memory, load_runtime_checkpoint
from chatbot.model_manager import ModelManager
from chatbot.models import Message
from chatbot.teleport import TeleportEnvelope, wave_mode_enabled
from chatbot.agent_runtime import (
    execute_teleport_for_workspace,
    extract_agent_command,
    execute_file_write_from_response,
    file_generation_contract,
    record_chamber_artifact,
    handle_turn,
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

    def _compact_artifact_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Trim checkpoint payloads so residue stays small and resumable."""
        if not isinstance(payload, dict):
            return {}

        max_text = 800
        max_items = 8

        def _trim(value: Any) -> Any:
            if isinstance(value, str):
                text = value.strip()
                if len(text) <= max_text:
                    return text
                return text[: max_text - 3] + "..."
            if isinstance(value, dict):
                compact: Dict[str, Any] = {}
                for idx, (k, v) in enumerate(value.items()):
                    if idx >= max_items:
                        compact["_truncated_keys"] = len(value) - max_items
                        break
                    compact[str(k)] = _trim(v)
                return compact
            if isinstance(value, list):
                if len(value) > max_items:
                    return [_trim(v) for v in value[:max_items]] + [f"... ({len(value) - max_items} more)"]
                return [_trim(v) for v in value]
            return value

        return _trim(payload)

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
            "payload": self._compact_artifact_payload(payload),
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
        """Execute a teleport envelope via shared runtime helpers."""
        return execute_teleport_for_workspace(envelope, str(self.cwd))

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
        """Compatibility shim around shared runtime command extraction."""
        return extract_agent_command(response)

    def _extract_file_content(self, response, language: str = None, require_marker: bool = False):
        """Compatibility shim around shared runtime file extraction."""
        from chatbot.agent_runtime import extract_file_content
        return extract_file_content(response, language, require_marker)

    def _execute_file_write_from_response(self, envelope, response):
        """Execute a file/script creation using shared runtime helpers."""
        return execute_file_write_from_response(envelope, response, str(self.cwd))

    def _file_generation_contract(self, envelope: TeleportEnvelope) -> str:
        """Compatibility shim around shared runtime file-generation contract."""
        return file_generation_contract(envelope)

    def default(self, line):
        """Handle chat interactions — with agentic teleport in wave mode."""
        if not line:
            return

        if line.strip().startswith('#'):
            return

        self.history.append(Message(role="user", content=line))

        print(f"\nThinking...")
        try:
            turn = handle_turn(
                system_prompt=config.SYSTEM_PROMPT,
                history=self.history,
                workspace=str(self.cwd),
                execute_teleport=execute_teleport_for_workspace,
                build_messages_fn=build_messages,
                build_messages_with_intent_fn=build_messages_with_intent,
                generate_text_fn=lambda messages: self._agentic_generate(messages),
                execute_file_write_fn=execute_file_write_from_response,
            )

            if turn.path == "wave_shell" and turn.display_text:
                print(f"\n📡 Teleport [shell_command]\n")
                print(turn.display_text)
                print("\nHermit: ", end="", flush=True)
                print(turn.assistant_reply + "\n")
                full_response = self._run_follow_up_commands(turn.assistant_reply, max_rounds=5)
                self.history.append(Message(role="assistant", content=full_response))
                return

            if turn.path == "wave_file":
                print(f"Hermit: {turn.assistant_reply}\n")
                if turn.display_text:
                    print(f"\n📡 Teleport [{getattr(turn.intent, 'shell_intent', 'file_write')}]")
                    print(turn.display_text)
                    summary = f"{turn.assistant_reply}\n\n[System: {turn.display_text}]"
                    self.history.append(Message(role="assistant", content=summary))
                else:
                    print("\n❌ File write canceled: model response did not include a valid [HERMIT_FILE] block or code fence.")
                    self.history.append(Message(role="assistant", content=turn.assistant_reply))
                return

            self.history.append(Message(role="assistant", content=turn.assistant_reply))
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
