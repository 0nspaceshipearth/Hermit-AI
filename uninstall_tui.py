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

from __future__ import annotations

import argparse
import curses
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Component:
    key: str
    title: str
    description: str
    default_selected: bool
    paths: List[Path]


def build_components(base_dir: Path) -> List[Component]:
    return [
        Component(
            key="venv",
            title="Virtual Environment (venv/)",
            description="Removes python libraries and environment.",
            default_selected=True,
            paths=[base_dir / "venv"],
        ),
        Component(
            key="models",
            title="AI Models (shared_models/)",
            description="Removes downloaded GGUF models (large files).",
            default_selected=False,
            paths=[base_dir / "shared_models"],
        ),
        Component(
            key="data",
            title="Search Indexes (data/)",
            description="Removes cached vectors and JIT indexes.",
            default_selected=True,
            paths=[base_dir / "data"],
        ),
        Component(
            key="hermit",
            title="System Command (hermit)",
            description="Checks /usr/local/bin, /usr/share/applications, and ~/.local/share.",
            default_selected=False,
            paths=[
                Path("/usr/local/bin/hermit"),
                Path("/usr/share/applications/hermit.desktop"),
                Path("/usr/share/pixmaps/hermit.png"),
                Path.home() / ".local/share/applications/hermit.desktop",
            ],
        ),
        Component(
            key="forge",
            title="System Command (forge)",
            description="Checks /usr/local/bin, /usr/share/applications, and ~/.local/share.",
            default_selected=False,
            paths=[
                Path("/usr/local/bin/forge"),
                Path("/usr/share/applications/forge.desktop"),
                Path.home() / ".local/share/applications/forge.desktop",
            ],
        ),
    ]


def get_path_size(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        if path.is_file() or path.is_symlink():
            return path.stat().st_size

        total = 0
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def selected_size(components: List[Component], selected: Dict[str, bool]) -> int:
    total = 0
    for component in components:
        if selected.get(component.key, False):
            for path in component.paths:
                total += get_path_size(path)
    return total


def render_line(stdscr: curses.window, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
    if y < 0 or width <= 0:
        return
    try:
        stdscr.addnstr(y, x, text, width, attr)
    except curses.error:
        pass


def run_checklist(components: List[Component]) -> Dict[str, bool] | None:
    checked: Dict[str, bool] = {c.key: c.default_selected for c in components}

    def _ui(stdscr: curses.window) -> Dict[str, bool] | None:
        index = 0
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()

            if max_y < 18 or max_x < 70:
                render_line(stdscr, 1, 2, "Terminal window is too small for the checklist UI.", max_x - 4)
                render_line(stdscr, 3, 2, "Resize to at least 70x18 and try again.", max_x - 4)
                render_line(stdscr, 5, 2, "Press q to cancel.", max_x - 4)
                stdscr.refresh()
                key = stdscr.getch()
                if key in (ord("q"), ord("Q"), 27):
                    return None
                continue

            render_line(stdscr, 1, 2, "Hermit Uninstaller (Terminal)", max_x - 4, curses.A_BOLD)
            render_line(stdscr, 2, 2, "Use arrow keys to move, Space to toggle, Enter to continue.", max_x - 4)

            y = 4
            for i, component in enumerate(components):
                marker = "[x]" if checked[component.key] else "[ ]"
                prefix = f"{marker} {component.title}"
                attr = curses.A_REVERSE if i == index else 0
                render_line(stdscr, y, 2, prefix, max_x - 4, attr)
                y += 1
                render_line(stdscr, y, 6, component.description, max_x - 8)
                y += 1

            space_line = f"Estimated space to free: {format_size(selected_size(components, checked))}"
            render_line(stdscr, y + 1, 2, space_line, max_x - 4, curses.A_BOLD)
            render_line(stdscr, y + 2, 2, "Safety: this script never deletes .zim files.", max_x - 4)
            render_line(stdscr, y + 4, 2, "q = cancel", max_x - 4, curses.A_DIM)

            stdscr.refresh()
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k"), ord("K")):
                index = (index - 1) % len(components)
            elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
                index = (index + 1) % len(components)
            elif key == ord(" "):
                key_name = components[index].key
                checked[key_name] = not checked[key_name]
            elif key in (curses.KEY_ENTER, 10, 13):
                return checked
            elif key in (ord("q"), ord("Q"), 27):
                return None

    return curses.wrapper(_ui)


def confirm(prompt: str, default_no: bool = True) -> bool:
    suffix = "[y/N]" if default_no else "[Y/n]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return not default_no
    return answer in ("y", "yes")


def needs_sudo(component: Component) -> bool:
    for path in component.paths:
        if str(path).startswith("/usr") and path.exists():
            return True
    return False


def run_sudo_remove(path: Path) -> None:
    rm_cmd = shutil.which("rm") or "/bin/rm"
    if path.is_dir() and not path.is_symlink():
        cmd = ["sudo", rm_cmd, "-rf", str(path)]
    else:
        cmd = ["sudo", rm_cmd, "-f", str(path)]
    subprocess.run(cmd, check=True)


def cleanup_pycache(base_dir: Path) -> None:
    for pycache in base_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)


def refresh_desktop_databases(selected_keys: List[str], may_need_sudo: bool) -> None:
    if "hermit" not in selected_keys and "forge" not in selected_keys:
        return

    updater = shutil.which("update-desktop-database")
    if not updater:
        return

    user_apps = Path.home() / ".local/share/applications"
    if user_apps.exists():
        try:
            subprocess.run(
                [updater, str(user_apps)],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass

    if may_need_sudo:
        try:
            subprocess.run(
                ["sudo", updater, "/usr/share/applications"],
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass


def uninstall(selected: List[Component], base_dir: Path) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    removed: List[str] = []

    requires_sudo = any(needs_sudo(component) for component in selected)
    if requires_sudo:
        if not shutil.which("sudo"):
            errors.append("sudo is required to remove system files, but was not found.")
            return removed, errors
        try:
            subprocess.run(["sudo", "-v"], check=True)
        except subprocess.CalledProcessError:
            errors.append("sudo authentication failed. System files were not removed.")
            return removed, errors

    for component in selected:
        for path in component.paths:
            if str(path).endswith(".zim"):
                errors.append(f"Skipped {path}: safety rule blocks deleting .zim files.")
                continue
            if not path.exists():
                continue
            try:
                if str(path).startswith("/usr"):
                    run_sudo_remove(path)
                elif path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception as exc:
                errors.append(f"Failed to remove {path}: {exc}")
        removed.append(component.key)

    if "venv" in [component.key for component in selected]:
        cleanup_pycache(base_dir)

    refresh_desktop_databases([component.key for component in selected], requires_sudo)
    return removed, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hermit terminal uninstaller (interactive checklist)."
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Skip checklist UI and use safe defaults (venv + data).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip final confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    components = build_components(base_dir)

    if args.defaults:
        selected_map = {component.key: component.default_selected for component in components}
    else:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print("Error: interactive terminal required. Use --defaults for non-interactive mode.", file=sys.stderr)
            return 1
        selected_map = run_checklist(components)
        if selected_map is None:
            print("Cancelled.")
            return 0

    selected = [component for component in components if selected_map.get(component.key, False)]
    if not selected:
        print("Nothing selected. No changes made.")
        return 0

    print("Selected for removal:")
    for component in selected:
        print(f"- {component.title}")
    estimate = format_size(selected_size(components, selected_map))
    print(f"Estimated space to free: {estimate}")
    print("Safety: .zim files are always protected.")

    if not args.yes and not confirm("Proceed with uninstall?", default_no=True):
        print("Cancelled.")
        return 0

    removed, errors = uninstall(selected, base_dir)
    if removed:
        print(f"Removed selections: {', '.join(removed)}")
    if errors:
        print("Completed with errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Uninstall completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
