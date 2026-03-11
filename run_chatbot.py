#!/usr/bin/env python3

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

"""Simple launcher script for chatbot."""

import sys
import os
import argparse

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add to path and import
sys.path.insert(0, script_dir)

from chatbot import ChatbotGUI
from chatbot.config import DEFAULT_MODEL, DEBUG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermit Chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--cli", action="store_true", help="Run in command-line interface mode")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Ollama model to use")
    
    args = parser.parse_args()
    
    # Validation: If model arg has spaces (e.g. user passed a query), ignore it
    if args.model and " " in args.model:
        if args.debug:
            print(f"[WARNING] Argument '{args.model}' ignored (looks like a query, not a model). Using default.", file=sys.stderr)
        args.model = DEFAULT_MODEL
    
    # Set DEBUG flag in config
    from chatbot import config
    config.DEBUG = args.debug
    
    
    if args.debug:
        print("[DEBUG] Debug mode enabled", file=sys.stderr)
        print(f"[DEBUG] Using model: {args.model}", file=sys.stderr)
        print(f"[DEBUG] Script directory: {script_dir}", file=sys.stderr)
    
    # Check for CLI mode
    if args.cli:
        from chatbot.cli import ChatbotCLI
        try:
            cli = ChatbotCLI(args.model)
            cli.cmdloop()
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"CLI Error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        app = ChatbotGUI(args.model)
        app.run()
    except KeyboardInterrupt:
        if args.debug:
            print("\n[DEBUG] Interrupted by user", file=sys.stderr)
        pass
    except RuntimeError as e:
        if args.debug:
            print(f"[DEBUG] RuntimeError: {e}", file=sys.stderr)
        # Error already handled by GUI, no need to print to terminal
        sys.exit(1)

