#!/usr/bin/env bash

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

# Simple launcher for chatbot GUI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

append_ld_library_path() {
    local lib_dir="$1"
    if [ ! -d "$lib_dir" ]; then
        return 0
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$lib_dir:"*) ;;
        *) export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

export_venv_cuda_libs() {
    local python_cmd="$1"
    if [ ! -x "$python_cmd" ]; then
        return 0
    fi

    local site_packages
    site_packages="$({
        "$python_cmd" - <<'PY'
import site

candidates = []
getsitepackages = getattr(site, "getsitepackages", None)
if callable(getsitepackages):
    candidates.extend(getsitepackages())
usersite = getattr(site, "getusersitepackages", lambda: None)()
if usersite:
    candidates.append(usersite)

seen = set()
for path in candidates:
    if path and path.endswith("site-packages") and path not in seen:
        seen.add(path)
        print(path)
        break
PY
    } 2>/dev/null)"

    if [ -z "$site_packages" ] || [ ! -d "$site_packages/nvidia" ]; then
        return 0
    fi

    for lib_dir in "$site_packages"/nvidia/*/lib "$site_packages"/nvidia/*/lib64; do
        append_ld_library_path "$lib_dir"
    done
}

# Ollama check removed (Local Inference Mode)


# Launch chatbot GUI
# Check for virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"
    export_venv_cuda_libs "$PYTHON_CMD"
else
    PYTHON_CMD="python3"
fi

# Check if --debug flag is present
DEBUG_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
        break
    fi
done

# Launch chatbot GUI (suppress verbose stdout unless debug, but ALWAYS show errors)
if [ "$DEBUG_MODE" = true ]; then
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@"
else
    # Only suppress stdout, keep stderr visible for errors
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@" 2>&1 | grep -v "^DEBUG:" || true
fi
