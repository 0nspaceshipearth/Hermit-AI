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

from chatbot import config
from chatbot.tui import HermitTUI
from chatbot.gui import ChatbotGUI
# from chatbot.config import DEFAULT_MODEL, DEBUG  # Moved to config import

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermit Chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    # Legacy flags deprecated; use TUI only
    parser.add_argument("--cli", action="store_true", help="Legacy CLI mode (deprecated)")
    parser.add_argument("--gui", action="store_true", help="Legacy GUI mode (deprecated)")
    parser.add_argument("model", nargs="?", default=config.DEFAULT_MODEL, help="Model to use")
    
    args = parser.parse_args()
    
    # Validation: If model arg has spaces (e.g. user passed a query), ignore it
    if args.model and " " in args.model:
        if args.debug:
            print(f"[WARNING] Argument '{args.model}' ignored (looks like a query, not a model). Using default.", file=sys.stderr)
        args.model = config.DEFAULT_MODEL
    
    # Set DEBUG flag in config
    from chatbot import config
    config.DEBUG = args.debug
    
    
    if args.debug:
        print("[DEBUG] Debug mode enabled", file=sys.stderr)
        print(f"[DEBUG] Using model: {args.model}", file=sys.stderr)
        print(f"[DEBUG] Script directory: {script_dir}", file=sys.stderr)
    
    # Default to new TUI; --gui for legacy GUI; --cli for legacy CLI
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
    elif args.gui:
        try:
            app = ChatbotGUI(args.model)
            app.run()
        except KeyboardInterrupt:
            pass
        except RuntimeError as e:
            if args.debug:
                print(f"[DEBUG] RuntimeError: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: New TUI
        try:
            workspace = script_dir
            tui = HermitTUI(args.model, workspace=workspace)
            tui.run()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"TUI Error: {e}", file=sys.stderr)
            sys.exit(1)

