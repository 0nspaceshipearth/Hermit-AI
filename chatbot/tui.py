#!/usr/bin/env python3
# Hermit TUI - Rich-based Terminal UI mimicking GUI feel
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rich TUI frontend for Hermit, preserving GUI look/feel."""
import asyncio
import os
import sys
from pathlib import Path
from typing import List


from rich.box import HEAVY_HEAD, MINIMAL
from rich.console import Console, ConsoleOptions, RenderResult
# from rich.containers import Columns, Horizontal, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text, TextType
from rich.traceback import install as install_rich_traceback

from chatbot import config
from chatbot.chat import build_messages
from chatbot.models import Message
from chatbot.agent_runtime import handle_turn, execute_teleport_for_workspace
from chatbot.gui_runtime import execute_file_write_from_response  # Reuse shared helpers

install_rich_traceback()

class LoadingSpinner:
    """Animated loading indicator."""
    def __init__(self, console: Console):
        self.console = console

    async def __call__(self, status: str = "Thinking..."):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(status, total=None)
            while True:  # Simulate until interrupted
                progress.update(task, description=status)
                await asyncio.sleep(0.1)

class ChatBubble:
    """Message bubble mimicking GUI style."""
    def __init__(self, role: str, content: str, theme: str = "dark"):
        self.role = role
        self.content = content
        self.theme = theme

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        styles = {
            "user": "bold cyan on default" if self.theme == "dark" else "bold blue on default",
            "ai": "green on dark_blue" if self.theme == "dark" else "white on navy",
            "system": "yellow on black" if self.theme == "dark" else "black on yellow",
        }
        style = styles.get(self.role, "white on default")
        bubble = Panel(
            Text(self.content, style="white") if self.theme == "dark" else Text(self.content),
            title=f"[{self.role.upper()}]" if self.role != "system" else "[SYSTEM]",
            border_style=style,
            box=HEAVY_HEAD if self.theme == "dark" else MINIMAL,
            padding=(1, 2),
        )
        yield bubble

class HermitTUI:
    def __init__(self, model: str = config.DEFAULT_MODEL, workspace: str = "."):
        self.console = Console()
        self.model = model
        self.workspace = Path(workspace).resolve()
        self.history: List[Message] = []
        self.theme = "dark"  # Noir-like
        self.loading = LoadingSpinner(self.console)
        self.rag = None
        print(f"Initializing RAG System (Model: {model})...")
        try:
            from chatbot.rag import RAGSystem
            self.rag = RAGSystem()
            import chatbot.chat
            chatbot.chat._rag_system = self.rag
            print("RAG System Ready.")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            print("Some functionality may be limited.")

    def run(self):
        """Main TUI loop."""
        self.console.print(Panel("Hermit TUI - Offline AI Chatbot", style="bold magenta", box=HEAVY_HEAD))
        self.console.print(f"[Model] {self.model} | [Workspace] {self.workspace}", style="dim")
        self.console.print("─" * 80)

        # loop = asyncio.get_event_loop()
        self._chat_loop()

    def _chat_loop(self):
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/]", console=self.console)
                if user_input.lower() in {"quit", "exit", ":q"}:
                    break
                if not user_input.strip():
                    continue

                self.history.append(Message(role="user", content=user_input))
                self.console.print(ChatBubble("user", user_input, self.theme))

                # Show loading
                self.console.print("[bold yellow]Thinking...[/]")
                turn = handle_turn(
                    system_prompt=config.SYSTEM_PROMPT,
                    history=self.history,
                    workspace=str(self.workspace),
                    execute_teleport=lambda env: execute_teleport_for_workspace(env, str(self.workspace)),
                    build_messages_fn=build_messages,
                    build_messages_with_intent_fn=lambda s, h, q=None: (build_messages(s, h, q), None),
                    generate_text_fn=lambda msgs: self._generate(msgs),  # Sync wrapper
                    execute_file_write_fn=lambda env, resp, ws: execute_file_write_from_response(env, resp, ws),
                )
                # loading_task.cancel()

                if turn.path == "wave_shell" and turn.display_text:
                    self.console.print(Panel(turn.display_text, title="📡 Shell Teleport", border_style="yellow"))
                if turn.assistant_reply:
                    self.console.print(ChatBubble("ai", turn.assistant_reply, self.theme))
                    self.history.append(Message(role="assistant", content=turn.assistant_reply))

            except KeyboardInterrupt:
                self.console.print("\\n[bold red]Interrupted. Goodbye![/]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {e}[/]")

    def _generate(self, messages):
        """Sync generate wrapper."""
        from chatbot.chat import full_chat
        return full_chat(self.model, messages)

    def __del__(self):
        try:
            from chatbot.model_manager import ModelManager
            ModelManager.close_all()
        except:
            pass

if __name__ == "__main__":
    # Workspace handled via init arg
    tui = HermitTUI()
    tui.run()
