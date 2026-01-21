
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

"""Chatbot GUI application."""

import sys
from chatbot.gui import ChatbotGUI
from chatbot.models import Message, ModelPlatform
from chatbot.config import DEFAULT_MODEL
from chatbot.state import HermitContext

__all__ = ['ChatbotGUI', 'Message', 'ModelPlatform', 'HermitContext']


def main():
    """Main entry point."""
    model = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    try:
        app = ChatbotGUI(model)
        app.run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
