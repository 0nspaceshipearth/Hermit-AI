"""GUI integration helpers for shared runtime.

These shims allow GUI to use the same handle_turn() as CLI with minimal frontend changes.
"""

import os
from typing import Callable

from chatbot.agent_runtime import (
    handle_turn,
    execute_teleport_for_workspace,
    execute_file_write_from_response,
)
from chatbot.chat import build_messages, build_messages_with_intent
from chatbot import config
