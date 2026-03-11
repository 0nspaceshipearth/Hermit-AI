#!/usr/bin/env python3
"""Launcher entrypoint wiring to new Rich TUI (priority #1)."""

import os
import sys
from pathlib import Path

# Ensure chatbot/ is importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.tui import HermitTUI

if __name__ == "__main__":
    workspace = Path.cwd().resolve()
    tui = HermitTUI(workspace=str(workspace))
    tui.run()