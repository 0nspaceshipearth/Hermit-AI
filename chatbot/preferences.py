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

"""Simple settings persistence for Hermit GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Keep public build settings isolated from other local Hermit variants.
HERMIT_DIR = os.environ.get("HERMIT_PUBLIC_HOME", os.path.expanduser("~/.hermit-public"))
SETTINGS_FILE = os.path.join(HERMIT_DIR, "settings.json")


def _ensure_dir() -> None:
    Path(HERMIT_DIR).mkdir(parents=True, exist_ok=True)


def _load_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def _save_settings(settings: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    try:
        os.chmod(SETTINGS_FILE, 0o600)
    except OSError:
        pass


def load_theme_name() -> Optional[str]:
    settings = _load_settings()
    value = settings.get("theme")
    return value if isinstance(value, str) else None


def save_theme_name(name: str) -> None:
    settings = _load_settings()
    settings["theme"] = name
    _save_settings(settings)
