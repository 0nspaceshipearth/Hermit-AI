
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

"""GUI interface for chatbot."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import subprocess
import threading
from typing import List, Tuple, Optional
from urllib.request import Request, urlopen

from chatbot.models import Message, ModelPlatform
from chatbot.chat import (
    stream_chat,
    full_chat,
    build_messages,
    set_status_callback,
    retrieve_and_display_links,
    clear_runtime_memory,
)
from chatbot import config
from chatbot.config import DEFAULT_MODEL
from chatbot.model_manager import set_download_callback
from chatbot.preferences import load_theme_name, save_theme_name


class DownloadProgressDialog:
    """A modal dialog that shows model download/loading progress."""
    
    def __init__(self, parent: tk.Tk, dark_mode: bool = True):
        self.parent = parent
        self.dark_mode = dark_mode
        self.dialog: Optional[tk.Toplevel] = None
        self.progress_var: Optional[tk.DoubleVar] = None
        self.status_label: Optional[tk.Label] = None
        self.detail_label: Optional[tk.Label] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self._pulse_job: Optional[str] = None
    
    def show(self, title: str = "Preparing Model..."):
        """Show the progress dialog."""
        if self.dialog:
            return  # Already showing

        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.dialog.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - 200
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - 75
        self.dialog.geometry(f"+{x}+{y}")
        
        # Style
        if self.dark_mode:
            bg_color = "#2A2A2A"
            fg_color = "#E0E0E0"
        else:
            bg_color = "#FFFFFF"
            fg_color = "#000000"
        
        self.dialog.configure(bg=bg_color)
        
        # Main frame
        frame = tk.Frame(self.dialog, bg=bg_color, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label (e.g., "Downloading model...")
        self.status_label = tk.Label(
            frame, text="Initializing...", font=("Arial", 12, "bold"),
            bg=bg_color, fg=fg_color
        )
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar
        style = ttk.Style()
        style.configure("Download.Horizontal.TProgressbar", 
                       troughcolor=bg_color, 
                       background="#4CAF50" if self.dark_mode else "#2196F3")
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var,
            maximum=100, length=350, mode='determinate',
            style="Download.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=(0, 10))
        
        # Detail label (e.g., "Model-Name (2.1 GB)")
        self.detail_label = tk.Label(
            frame, text="", font=("Arial", 10),
            bg=bg_color, fg=fg_color
        )
        self.detail_label.pack()
        
        # Prevent closing
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    
    def update(self, status: str, progress: float, detail: str):
        """Update the progress dialog.
        
        Args:
            status: Current status ("downloading", "loading", "ready", "error")
            progress: Progress value 0.0-1.0, or -1 for indeterminate
            detail: Detail text to show
        """
        if not self.dialog:
            return
        
        try:
            # Update status text
            status_texts = {
                "checking": "Checking model availability...",
                "downloading": "Downloading model...",
                "loading": "Loading model into GPU...",
                "ready": "Model ready!",
                "error": "Error"
            }
            self.status_label.config(text=status_texts.get(status, status))
            
            # Update progress bar
            if progress < 0:
                # Indeterminate mode - pulse animation
                self.progress_bar.config(mode='indeterminate')
                if not self._pulse_job:
                    self._start_pulse()
            else:
                # Determinate mode
                self._stop_pulse()
                self.progress_bar.config(mode='determinate')
                self.progress_var.set(progress * 100)
            
            # Update detail
            self.detail_label.config(text=detail)
            
            # Force update
            self.dialog.update_idletasks()
            
        except tk.TclError:
            pass  # Dialog was closed
    
    def _start_pulse(self):
        """Start indeterminate animation."""
        if self.progress_bar and self.dialog:
            self.progress_bar.start(15)
            self._pulse_job = "running"
    
    def _stop_pulse(self):
        """Stop indeterminate animation."""
        if self.progress_bar and self._pulse_job:
            try:
                self.progress_bar.stop()
            except:
                pass
            self._pulse_job = None
    
    def hide(self):
        """Hide and destroy the dialog."""
        self._stop_pulse()
        if self.dialog:
            try:
                self.dialog.grab_release()
                self.dialog.destroy()
            except:
                pass
            self.dialog = None


class ChatbotGUI:
    """Full-featured GUI chatbot interface."""
    
    def __init__(self, model: str = None, system_prompt: str = None, streaming_enabled: bool = True):
        try:
            import tkinter as tk
            from tkinter import ttk, scrolledtext, messagebox
        except ImportError:
            raise RuntimeError("tkinter not available. Install: sudo apt install python3-tk")
        
        self.tk = tk
        self.ttk = ttk
        self.scrolledtext = scrolledtext
        
        # Override standard messagebox with theme-aware custom dialogs
        try:
             # Inline logic to avoid circular imports if file not ready, though we just wrote it.
             # Actually, simpler to paste logic here or import if possible.
             # Let's import from the new file.
             from chatbot.custom_dialogs import StyledMessageBox
             self.messagebox = StyledMessageBox(self)
        except ImportError:
             print("Warning: Could not load custom dialogs, falling back to native.")
             self.messagebox = messagebox

        
        self.model = model or DEFAULT_MODEL
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.streaming_enabled = streaming_enabled
        
        self.history: List[Message] = []
        self.query_history: List[str] = []
        self.themes = self._build_themes()
        saved_theme = load_theme_name()
        self.theme_name = saved_theme if saved_theme in self.themes else "Noir"
        self._theme_palette = self.themes[self.theme_name]
        self.dark_mode = self._theme_palette["dark"]
        self.use_terminal_dialogs = True
        self.command_mode = None
        self._terminal_menu_active = False
        self._terminal_menu_start_mark = "terminal_menu_start"
        self._terminal_menu_end_mark = "terminal_menu_end"
        self._terminal_menu_tag = "terminal_menu_block"
        
        # Dual mode: link_mode (default) vs response_mode
        self.link_mode = False  # Default to Response mode
        
        self.root = self.tk.Tk()
        self.root.title(f"Hermit - {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
        self.root.geometry("900x700")
        self.root.minsize(400, 300)  # Minimum window size to keep input visible
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)

        # Main container using grid for better resize behavior
        self.root.grid_rowconfigure(0, weight=1)  # Chat area expands
        self.root.grid_rowconfigure(1, weight=0)  # Input area fixed height
        self.root.grid_columnconfigure(0, weight=1)

        # Chat display
        chat_frame = self.ttk.Frame(self.root)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 5))
        
        self.scrollbar = self.ttk.Scrollbar(chat_frame, orient=self.tk.VERTICAL)
        self.scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        self.chat_display = self.tk.Text(
            chat_frame, wrap=self.tk.WORD, padx=15, pady=15,
            state=self.tk.NORMAL, font=("Arial", 11),
            borderwidth=0, highlightthickness=0,
            yscrollcommand=self.scrollbar.set
        )
        self.chat_display.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        self.scrollbar.config(command=self.chat_display.yview)
        
        # Prevent direct editing
        def prevent_edit(event):
            if event.keysym not in ['Return', 'Tab'] and event.state & 0x4 == 0:
                return "break"
            return None
        self.chat_display.bind("<Key>", prevent_edit)
        self.chat_display.bind("<Button-1>", self.on_click)
        self.chat_display.bind("<Control-Button-1>", self.on_ctrl_click)
        self.chat_display.bind("<KeyPress-Return>", self.on_highlight_enter)
        
        self.cursor_handlers = {
            "enter": lambda e: self.chat_display.config(cursor="hand2"),
            "leave": lambda e: self.chat_display.config(cursor="")
        }
        
        # Input area - fixed at bottom, never collapses
        input_container = self.ttk.Frame(self.root)
        input_container.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 10))
        input_container.grid_columnconfigure(0, weight=1)  # Center the input frame

        input_frame = self.ttk.Frame(input_container)
        input_frame.grid(row=0, column=0)

        self.input_entry = self.tk.Entry(
            input_frame, font=("Arial", 12),
            relief=self.tk.FLAT, borderwidth=0,
            highlightthickness=1, width=50
        )
        self.input_entry.pack(side=self.tk.LEFT, padx=(0, 5), ipady=4)
        self.input_entry.bind("<Return>", self.on_input_return)
        self.input_entry.bind("<KeyRelease>", self.on_input_key)
        self.input_entry.bind("<Up>", self.on_autocomplete_nav)
        self.input_entry.bind("<Down>", self.on_autocomplete_nav)
        self.input_entry.bind("<Tab>", self.on_autocomplete_select)
        self.input_entry.bind("<Escape>", self.on_input_escape)
        self.input_entry.bind("<FocusOut>", self.on_input_focus_out)
        
        # Autocomplete listbox
        self.autocomplete_listbox = self.tk.Listbox(
            self.root, height=5, font=("Arial", 11),
            borderwidth=1, relief=self.tk.SOLID,
            activestyle="none"
        )
        self.autocomplete_listbox.bind("<Button-1>", self.on_autocomplete_click)
        self.autocomplete_listbox.bind("<Return>", self.on_autocomplete_select)
        
        self.autocomplete_active = False
        self.autocomplete_suggestions: List[str] = []
        self.autocomplete_selected_index = -1
        
        # Triangle send button
        send_canvas = self.tk.Canvas(
            input_frame, width=32, height=32,
            highlightthickness=0, borderwidth=0,
            relief=self.tk.FLAT
        )
        send_canvas.pack(side=self.tk.RIGHT)
        
        self.send_canvas = send_canvas
        self.send_canvas_color = "#808080"
        self.send_canvas_hover_color = "#FFFFFF"
        
        def draw_send_button(canvas, triangle_color):
            canvas.delete("all")
            points = [9.5, 7.5, 9.5, 24.5, 24.5, 16]
            canvas.create_polygon(points, fill="", outline=triangle_color, width=1)
        
        draw_send_button(send_canvas, self.send_canvas_color)
        
        def on_send_click(event):
            self.on_send()
        
        def on_send_enter(event):
            send_canvas.config(cursor="hand2")
            draw_send_button(send_canvas, self.send_canvas_hover_color)
        
        def on_send_leave(event):
            send_canvas.config(cursor="")
            draw_send_button(send_canvas, self.send_canvas_color)
        
        send_canvas.bind("<Button-1>", on_send_click)
        send_canvas.bind("<Enter>", on_send_enter)
        send_canvas.bind("<Leave>", on_send_leave)
        
        self._draw_send_button = draw_send_button
        self.selected_text = ""
        
        # Loading state management
        self.is_loading = False
        self.loading_text = ""
        self.loading_animation_id = None
        self.loading_pulse_step = 0
        self.loading_pulse_direction = 1  # 1 = brightening, -1 = dimming
        
        # Download progress - now uses chat bubble instead of modal dialog
        self.download_dialog: Optional[DownloadProgressDialog] = None  # Legacy, kept for compatibility
        self._download_bubble_tag: Optional[str] = None  # Tag for current download bubble
        self._download_bubble_start: Optional[str] = None  # Start index of bubble
        self._download_bubble_end: Optional[str] = None  # End index of bubble
        self._download_dismiss_job: Optional[str] = None  # After job ID for auto-dismiss
        self._setup_download_callback()
        
        self.apply_theme()
        self.root.after(100, lambda: self.input_entry.focus_set())
    
    def update_status(self, text: str):
        """Update status (no-op for minimal UI)."""
        pass
    
    def _setup_download_callback(self):
        """Setup callback to receive download progress from ModelManager."""
        def on_progress(status: str, progress: float, detail: str):
            # Use after() to safely update GUI from any thread
            self.root.after(0, lambda: self._handle_download_progress(status, progress, detail))
        
        set_download_callback(on_progress)
    
    def _handle_download_progress(self, status: str, progress: float, detail: str):
        """Handle download progress updates (called on main thread).

        Shows progress as a chat bubble instead of a modal popup.
        """
        # Cancel any pending auto-dismiss
        if self._download_dismiss_job:
            self.root.after_cancel(self._download_dismiss_job)
            self._download_dismiss_job = None

        if status == "downloading":
            # Format progress message - clean, no emojis
            pct = max(0.0, min(100.0, progress * 100.0)) if progress >= 0 else 0.0
            progress_bar = self._make_progress_bar(int(pct))
            msg = f"Downloading {detail}\n{progress_bar} {pct:0.1f}%"
            self._show_download_bubble(msg)

        elif status == "loading":
            msg = f"Loading model into memory..."
            self._show_download_bubble(msg)

        elif status == "checking":
            # Skip checking status - too brief to show
            pass

        elif status == "ready":
            msg = f"Model ready"
            self._show_download_bubble(msg)
            # Auto-dismiss after 3 seconds
            self._download_dismiss_job = self.root.after(3000, self._dismiss_download_bubble)

        elif status == "error":
            msg = f"Download error: {detail}"
            self._show_download_bubble(msg)
            # Auto-dismiss after 6 seconds (longer for errors)
            self._download_dismiss_job = self.root.after(6000, self._dismiss_download_bubble)

    def _make_progress_bar(self, percent: int, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(width * percent / 100)
        empty = width - filled
        return f"[{'=' * filled}{'-' * empty}]"

    def _show_download_bubble(self, message: str):
        """Show or update the download notification bubble in chat.

        Matches existing bubble style - clean, minimal, dark gray text.
        """
        try:
            # Remove existing bubble if present
            if self._download_bubble_tag and self._download_bubble_start:
                try:
                    self.chat_display.delete(self._download_bubble_start, self._download_bubble_end)
                except self.tk.TclError:
                    pass  # Indices may be invalid

            # Insert new bubble
            self.chat_display.insert(self.tk.END, "\n")
            self._download_bubble_start = self.chat_display.index(self.tk.END + "-1c")

            self.chat_display.insert(self.tk.END, message)

            # Add padding
            self.chat_display.insert(self.tk.END, "    ")
            self._download_bubble_end = self.chat_display.index(self.tk.END)

            # Style the bubble to match existing design
            self._download_bubble_tag = f"download_bubble_{id(self)}"
            self.chat_display.tag_add(
                self._download_bubble_tag,
                self._download_bubble_start,
                self._download_bubble_end
            )

            # Match existing bubble style
            bg_color = self._theme_palette.get("bubble_bg", "#1E1E1E" if self.dark_mode else "#E0E0E0")
            fg_color = self._theme_palette.get("muted_fg", "#808080" if self.dark_mode else "#606060")

            self.chat_display.tag_config(
                self._download_bubble_tag,
                background=bg_color,
                foreground=fg_color,
                font=("Consolas", 10),
                lmargin1=12,
                lmargin2=12,
                rmargin=12,
                spacing1=6,
                spacing2=3,
                spacing3=6
            )

            self.chat_display.see(self.tk.END)

        except Exception as e:
            print(f"[GUI] Download bubble error: {e}")

    def _dismiss_download_bubble(self):
        """Remove the download notification bubble from chat."""
        self._download_dismiss_job = None
        if self._download_bubble_tag and self._download_bubble_start:
            try:
                self.chat_display.delete(self._download_bubble_start, self._download_bubble_end)
            except self.tk.TclError:
                pass  # Already removed or invalid indices
            self._download_bubble_tag = None
            self._download_bubble_start = None
            self._download_bubble_end = None

    def _hide_download_dialog(self):
        """Hide the download progress dialog (legacy method)."""
        if self.download_dialog:
            self.download_dialog.hide()
            self.download_dialog = None
    
    def show_loading(self, text: str = "Thinking"):
        """Show loading state with chat bubble."""
        def _show():
            # Remove any existing loading bubble first
            self._hide_loading_internal()
            
            self.is_loading = True
            
            # Insert slightly cleaner spacing
            self.chat_display.insert(self.tk.END, "\n")
            
            # Start tracking this bubble
            message_start = self.chat_display.index(self.tk.END + "-1c")
            
            # prefix = "AI: " # Removed
            # self.chat_display.insert(self.tk.END, prefix)
            
            # Content area
            content_start = self.chat_display.index(self.tk.END + "-1c")
            self.chat_display.insert(self.tk.END, text + "...")
            content_end = self.chat_display.index(self.tk.END + "-1c")
            
            # Padding
            self.chat_display.insert(self.tk.END, "    ")
            message_end = self.chat_display.index(self.tk.END + "-1c")
            
            # Apply tags
            # 1. Style tag (shared with normal AI messages)
            style_tag = f"ai_message_{id(self)}"
            self.chat_display.tag_add(style_tag, message_start, message_end)
            self._configure_message_tag(style_tag, "ai")
            
            # 2. Loading identification tags
            self.chat_display.tag_add("loading_bubble", message_start, message_end)
            self.chat_display.tag_add("loading_content_text", content_start, content_end)
            
            self.chat_display.insert(self.tk.END, "\n\n")
            self.chat_display.see(self.tk.END)
            
            # Start animation
            self.chat_display.tag_raise("loading_content_text")
            self._animate_loading_pulse()
            
        self.root.after(0, _show)
    
        self.root.after(0, _show)
    
    def transition_loading_to_response(self) -> str:
        """
        Transition the loading bubble into a response insertion point.
        Returns the mark name where response text should be inserted.
        """
        if not self.is_loading:
            # Fallback if no bubble exists
            self.chat_display.insert(self.tk.END, "\n")
            ai_tag_name = f"ai_message_{id(self)}"
            self._configure_message_tag(ai_tag_name, "ai")
            self.chat_display.insert(self.tk.END, "", ai_tag_name)
            return self.tk.END

        # Stop animation
        self.is_loading = False
        
        # Find the content range
        ranges = self.chat_display.tag_ranges("loading_content_text")
        if not ranges:
            return self.tk.END
            
        start, end = ranges[0], ranges[1]
        
        # Delete the "Thinking..." text
        self.chat_display.delete(start, end)
        
        # Create a mark at the insertion point for the response
        # Using RIGHT gravity so the mark moves as we insert text
        mark_name = "response_insert_mark"
        self.chat_display.mark_set(mark_name, start)
        self.chat_display.mark_gravity(mark_name, self.tk.RIGHT)
        
        # Clean up temporary loading tags (but KEEP the ai_message tag!)
        self.chat_display.tag_remove("loading_bubble", "1.0", self.tk.END)
        self.chat_display.tag_remove("loading_content_text", "1.0", self.tk.END)
        
        return mark_name

    def hide_loading(self):
        """Hide loading state."""
        self.root.after(0, self._hide_loading_internal)
        
    def _hide_loading_internal(self):
        """Internal helper to remove loading bubble immediately."""
        self.is_loading = False
        ranges = self.chat_display.tag_ranges("loading_bubble")
        if ranges:
            start, end = ranges[0], ranges[1]
            
            # Check for preceding newline to clean up spacing
            try:
                prev_idx = self.chat_display.index(f"{start}-1c")
                if self.chat_display.get(prev_idx) == "\n":
                   start = prev_idx
            except:
                pass
            
            # Delete the bubble
            self.chat_display.delete(start, end)
            
            # Also clean up the trailing newlines optionally? 
            # If we delete the block, we might want to ensure we don't leave a huge gap.
            # But simpler is often safer.
            
    def update_loading_text(self, text: str):
        """Update loading text in the bubble."""
        def _update():
            if not self.is_loading:
                # If we aren't "loading" but got an update, maybe show it?
                # No, that would be weird.
                return
            was_near_bottom = self._is_near_bottom()
            
            ranges = self.chat_display.tag_ranges("loading_content_text")
            if ranges:
                start, end = ranges[0], ranges[1]
                # Replace text
                self.chat_display.delete(start, end)
                self.chat_display.insert(start, text + "...")
                
                # Re-apply the content tag to the new text so next update works
                new_end = self.chat_display.index(f"{start} + {len(text) + 3}c")
                self.chat_display.tag_add("loading_content_text", start, new_end)
                
                # CRITICAL: Re-apply the main bubble tags to the new content
                # otherwise hide_loading won't find this text to delete!
                self.chat_display.tag_add("loading_bubble", start, new_end)
                self.chat_display.tag_add(f"ai_message_{id(self)}", start, new_end)
                
                if was_near_bottom:
                    self.chat_display.see(self.tk.END)
                
        self.root.after(0, _update)

    def _is_near_bottom(self, margin: float = 0.001) -> bool:
        """Return True when chat viewport is at/near bottom."""
        try:
            _, last = self.chat_display.yview()
            return last >= (1.0 - margin)
        except Exception:
            return True

    
    def _get_pulse_color(self) -> str:
        """Get the current pulse color based on step."""
        # Pulse between dim gray and bright text color
        if self.dark_mode:
            # Dark mode: pulse between #AAAAAA (light gray) and #FFFFFF (white)
            base_val = 170  # 0xAA
            range_val = 85   # 0xFF - 0xAA
        else:
            # Light mode: pulse between #999999 (dim) and #000000 (bright)
            base_val = 153  # 0x99
            range_val = -153  # 0x00 - 0x99
        
        # 10 steps for smooth pulsing
        progress = self.loading_pulse_step / 10.0
        val = int(base_val + (range_val * progress))
        val = max(0, min(255, val))
        return f"#{val:02x}{val:02x}{val:02x}"
    
    def _animate_loading_pulse(self):
        """Animate the loading text with pulsating brightness."""
        if not self.is_loading:
            return
        
        # Update pulse step
        self.loading_pulse_step += self.loading_pulse_direction
        if self.loading_pulse_step >= 10:
            self.loading_pulse_direction = -1
        elif self.loading_pulse_step <= 0:
            self.loading_pulse_direction = 1
        
        # Apply pulsing color
        pulse_color = self._get_pulse_color()
        
        # Target the bubble text tag instead of input_entry
        self.chat_display.tag_config("loading_content_text", foreground=pulse_color)
        self.chat_display.tag_raise("loading_content_text")
        
        # Schedule next frame (60ms for smooth animation)
        self.loading_animation_id = self.root.after(60, self._animate_loading_pulse)

    def on_input_escape(self, event):
        """Handle Escape key for command modes or autocomplete."""
        if self.command_mode:
            self._exit_command_mode(cancelled=True)
            return "break"
        return self.on_autocomplete_close(event)

    def _post_system(self, text: str):
        """Post a system message from any thread."""
        try:
            self.root.after(0, lambda: self.append_message("system", text))
        except Exception:
            print(text)

    def _render_terminal_menu(self, text: str):
        """Render or update a terminal-style menu block in chat."""
        try:
            ranges = list(self.chat_display.tag_ranges(self._terminal_menu_tag))
            if ranges:
                for i in range(len(ranges) - 2, -1, -2):
                    start = ranges[i]
                    end = ranges[i + 1]
                    self.chat_display.delete(start, end)

            block_start = self.chat_display.index(self.tk.END)
            self.chat_display.mark_set(self._terminal_menu_start_mark, block_start)
            self.chat_display.mark_gravity(self._terminal_menu_start_mark, self.tk.LEFT)

            self.chat_display.insert(self.tk.END, "\n")
            content_start = self.chat_display.index(self.tk.END)
            self.chat_display.insert(self.tk.END, text)
            content_end = self.chat_display.index(self.tk.END)
            self.chat_display.insert(self.tk.END, "\n\n")
            block_end = self.chat_display.index(self.tk.END)
            self.chat_display.mark_set(self._terminal_menu_end_mark, block_end)
            self.chat_display.mark_gravity(self._terminal_menu_end_mark, self.tk.LEFT)

            self.chat_display.tag_add(self._terminal_menu_tag, block_start, block_end)

            tag_name = f"system_message_{id(self)}"
            self.chat_display.tag_add(tag_name, content_start, content_end)
            self._configure_message_tag(tag_name, "system")
            self.chat_display.tag_config(
                tag_name,
                font=("Consolas", 10),
                lmargin1=10,
                lmargin2=10,
                rmargin=10,
                spacing1=3,
                spacing2=1,
                spacing3=3
            )

            # Highlight confirmation keywords for clarity.
            yes_color = "#5CD67A" if self.dark_mode else "#1E8E3E"
            escape_color = "#FF6B6B" if self.dark_mode else "#C62828"
            self.chat_display.tag_config("terminal_menu_yes", foreground=yes_color, font=("Consolas", 10, "bold"))
            self.chat_display.tag_config("terminal_menu_escape", foreground=escape_color, font=("Consolas", 10, "bold"))
            self._highlight_terminal_keyword(content_start, content_end, "YES", "terminal_menu_yes")
            self._highlight_terminal_keyword(content_start, content_end, "Escape", "terminal_menu_escape", nocase=True)
            self.chat_display.see(self.tk.END)
            self.chat_display.update_idletasks()

            self._terminal_menu_active = True
        except Exception as e:
            print(f"[GUI] Terminal menu error: {e}")

    def _highlight_terminal_keyword(self, start: str, end: str, keyword: str, tag: str, nocase: bool = False):
        """Apply a tag to each keyword occurrence within a terminal menu block."""
        idx = start
        while True:
            idx = self.chat_display.search(keyword, idx, stopindex=end, nocase=1 if nocase else 0)
            if not idx:
                break
            keyword_end = f"{idx}+{len(keyword)}c"
            self.chat_display.tag_add(tag, idx, keyword_end)
            idx = keyword_end

    def _clear_terminal_menu(self):
        """Remove the currently rendered terminal menu block."""
        try:
            ranges = list(self.chat_display.tag_ranges(self._terminal_menu_tag))
            if ranges:
                for i in range(len(ranges) - 2, -1, -2):
                    start = ranges[i]
                    end = ranges[i + 1]
                    self.chat_display.delete(start, end)
            self.chat_display.mark_unset(self._terminal_menu_start_mark)
            self.chat_display.mark_unset(self._terminal_menu_end_mark)
        except Exception:
            pass
        self._terminal_menu_active = False

    def _close_command_mode(self):
        """Close command mode and clear its on-screen menu block."""
        self.command_mode = None
        self._clear_terminal_menu()

    def _exit_command_mode(self, cancelled: bool = False):
        """Exit the current command mode."""
        if not self.command_mode:
            return

        mode = self.command_mode.get("name")
        if cancelled and mode == "themes":
            original = self.command_mode.get("original_theme")
            if original:
                self.set_theme(original, persist=True)

        self._close_command_mode()

    def _render_current_command_menu(self):
        if not self.command_mode:
            return
        mode = self.command_mode.get("name")
        if mode == "themes":
            self._render_themes_menu()
        elif mode == "model":
            self._render_model_menu()
        elif mode == "api":
            self._render_api_menu()
        elif mode == "forge":
            self._render_forge_menu()

    def _handle_command_input(self, user_input: str):
        """Route input to the active terminal mode handler."""
        if not self.command_mode:
            return
        mode = self.command_mode.get("name")
        if mode == "themes":
            self._handle_themes_input(user_input)
        elif mode == "model":
            self._handle_model_input(user_input)
        elif mode == "api":
            self._handle_api_input(user_input)
        elif mode == "forge":
            self._handle_forge_input(user_input)

    def _command_menu_nav(self, delta: int):
        """Navigate the active menu with Up/Down keys."""
        if not self.command_mode:
            return
        if self.command_mode.get("type") != "menu":
            return
        items = self.command_mode.get("items", [])
        if not items:
            return
        idx = self.command_mode.get("index", 0)
        idx = (idx + delta) % len(items)
        self.command_mode["index"] = idx
        if self.command_mode.get("name") == "themes":
            self._preview_theme(idx)
        self._render_current_command_menu()

    def _preview_theme(self, index: int):
        items = self.command_mode.get("items", [])
        if not items or index < 0 or index >= len(items):
            return
        self.set_theme(items[index], persist=False)

    def _enter_themes_mode(self):
        items = list(self.themes.keys())
        if not items:
            self.append_message("system", "No themes available.")
            return
        current = self.theme_name
        idx = items.index(current) if current in items else 0
        self.command_mode = {
            "name": "themes",
            "type": "menu",
            "items": items,
            "index": idx,
            "original_theme": current,
        }
        self._render_themes_menu()

    def _render_themes_menu(self):
        items = self.command_mode.get("items", [])
        idx = self.command_mode.get("index", 0)
        lines = []
        for i, name in enumerate(items):
            mode = "Dark" if self.themes[name]["dark"] else "Light"
            marker = ">" if i == idx else " "
            current = " *" if name == self.theme_name else ""
            lines.append(f"{marker} {i+1}. {name} ({mode}){current}")
        text = (
            "THEMES\n"
            "Use Up/Down to preview, Enter to apply, Esc to cancel (revert).\n"
            "You can also type a number or theme name.\n\n"
            + "\n".join(lines)
        )
        self._render_terminal_menu(text)

    def _handle_themes_input(self, user_input: str):
        text = user_input.strip()
        if not text:
            save_theme_name(self.theme_name)
            self._close_command_mode()
            self.append_message("system", f"Theme set to {self.theme_name}.")
            return
        lowered = text.lower()
        if lowered in {"cancel", "exit", "quit"}:
            self._exit_command_mode(cancelled=True)
            return
        if text.isdigit():
            idx = int(text) - 1
            items = self.command_mode.get("items", [])
            if 0 <= idx < len(items):
                self.set_theme(items[idx])
                self._close_command_mode()
                self.append_message("system", f"Theme set to {items[idx]}.")
                return
        resolved = self._resolve_theme_name(text)
        if resolved:
            self.set_theme(resolved)
            self._close_command_mode()
            self.append_message("system", f"Theme set to {resolved}.")
            return
        self.append_message("system", f"Unknown theme '{text}'.")

    def _format_model_display(self, model_name: str) -> str:
        import re
        display_name = model_name.split('/')[-1] if '/' in model_name else model_name
        display_name = re.sub(r'-(\d+)-of-(\d+)\.gguf$', '.gguf', display_name)
        if "qwen2.5-3b" in display_name.lower():
            quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
            quant = quant_match.group(1).upper() if quant_match else "Unknown"
            display_name = f"Qwen 2.5 3B ({quant})"
        elif "qwen2.5-7b" in display_name.lower():
            quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
            quant = quant_match.group(1).upper() if quant_match else "Unknown"
            display_name = f"Qwen 2.5 7B ({quant})"
        elif "qwen2.5-1.5b" in display_name.lower() or "qwen2.5-1b" in display_name.lower():
            quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
            quant = quant_match.group(1).upper() if quant_match else "Unknown"
            display_name = f"Qwen 2.5 1.5B ({quant})"
        elif "llama-3.2-3b" in display_name.lower():
            quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
            quant = quant_match.group(1).upper() if quant_match else "Unknown"
            display_name = f"Llama 3.2 3B ({quant})"
        elif "llama-3" in display_name.lower() and "8b" in display_name.lower():
            quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
            quant = quant_match.group(1).upper() if quant_match else "Unknown"
            display_name = f"Llama 3 8B ({quant})"
        return display_name

    def _enter_model_mode(self):
        models = self.get_installed_models()
        if not models:
            self.append_message("system", "No models found.")
            return
        model_list = [m for m, _ in models]
        display_list = [self._format_model_display(m) for m in model_list]
        idx = model_list.index(self.model) if self.model in model_list else 0
        self.command_mode = {
            "name": "model",
            "type": "menu",
            "items": model_list,
            "display": display_list,
            "index": idx,
            "pending": None,
        }
        self._render_model_menu()

    def _render_model_menu(self):
        items = self.command_mode.get("items", [])
        display = self.command_mode.get("display", [])
        idx = self.command_mode.get("index", 0)
        pending = self.command_mode.get("pending")
        lines = []
        for i, name in enumerate(items):
            marker = ">" if i == idx else " "
            current = " *" if name == self.model else ""
            label = display[i] if i < len(display) else name
            lines.append(f"{marker} {i+1}. {label}{current}")

        footer = "Commands: Enter=select | type 'delete' on highlighted | paste HF repo/url=download | Esc=cancel"
        download_hint = "Download: paste HF repo/url (GGUF only)"
        if pending:
            footer = f"PENDING: {pending.get('prompt', '')}"

        text = (
            "MODEL SELECTOR\n"
            f"Current model: {self.model}\n"
            + (f"{download_hint}\n" if not pending else "")
            + "\n".join(lines)
            + "\n\n"
            + footer
        )
        self._render_terminal_menu(text)

    def _queue_model_download(self, repo_id: str):
        """Queue a model download confirmation prompt."""
        if not repo_id:
            self.append_message("system", "Usage: paste a Hugging Face repo id (owner/repo).")
            return
        prompt = f"Confirm download '{repo_id}'. Type YES to confirm, or press Escape to cancel."
        if "gguf" not in repo_id.lower():
            prompt = (
                f"'{repo_id}' does not include 'GGUF'. Hermit only supports GGUF.\n"
                "Type YES to continue, or press Escape to cancel."
            )
        self.command_mode["pending"] = {
            "action": "download",
            "repo_id": repo_id,
            "prompt": prompt,
        }
        self._render_model_menu()

    def _queue_model_delete(self, index: Optional[int] = None):
        """Queue a delete confirmation for a model index or the highlighted item."""
        items = self.command_mode.get("items", [])
        if not items:
            self.append_message("system", "No models available to delete.")
            return

        if index is None:
            index = self.command_mode.get("index", 0)

        if index < 0 or index >= len(items):
            self.append_message("system", "No valid highlighted model selected.")
            return

        model_id = items[index]
        self.command_mode["pending"] = {
            "action": "delete",
            "model_id": model_id,
            "prompt": f"Confirm delete '{model_id}'. Type YES to confirm, or press Escape to cancel.",
        }
        self._render_model_menu()

    def _normalize_hf_repo_input(self, text: str) -> Optional[str]:
        """Normalize pasted Hugging Face repo id or URL to owner/repo."""
        candidate = (text or "").strip().strip("\"'").rstrip("/")
        if not candidate:
            return None

        lowered = candidate.lower()
        if lowered.startswith("hf:"):
            candidate = candidate[3:].strip()
        elif lowered.startswith("repo:"):
            candidate = candidate[5:].strip()

        if candidate.lower().startswith("hf.co/"):
            candidate = "https://" + candidate

        if candidate.lower().startswith(("http://", "https://")):
            from urllib.parse import urlparse

            parsed = urlparse(candidate)
            host = parsed.netloc.lower()
            if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
                return None

            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) < 2:
                return None

            # Support URLs like /owner/repo and /models/owner/repo
            if parts[0] in {"models", "spaces", "datasets"} and len(parts) >= 3:
                owner, repo = parts[1], parts[2]
            else:
                owner, repo = parts[0], parts[1]

            repo_id = f"{owner}/{repo}".strip().rstrip("/")
            return repo_id if owner and repo else None

        candidate = candidate.split("?", 1)[0].split("#", 1)[0].strip().rstrip("/")
        if " " in candidate or "/" not in candidate:
            return None
        owner, repo = candidate.split("/", 1)
        return candidate if owner and repo else None

    def _apply_model_selection(self, model_name: str):
        if model_name != self.model:
            self.model = model_name
            self.root.title(f"Chatbot - {model_name} ({'Link Mode' if self.link_mode else 'Response Mode'})")
            self.append_message("system", f"Model changed to: {model_name}")
            self.update_status(f"Model: {model_name}")

    def _handle_model_input(self, user_input: str):
        text = user_input.strip()
        if not text:
            items = self.command_mode.get("items", [])
            idx = self.command_mode.get("index", 0)
            if items:
                self._apply_model_selection(items[idx])
            self._close_command_mode()
            return

        lowered = text.lower()
        if lowered in {"cancel", "exit", "quit"}:
            self._exit_command_mode(cancelled=True)
            return

        pending = self.command_mode.get("pending")
        if pending:
            if lowered in {"yes", "y"}:
                action = pending.get("action")
                if action == "delete":
                    self._perform_model_delete(pending.get("model_id"))
                elif action == "download":
                    self._perform_model_download(pending.get("repo_id"))
                self._close_command_mode()
            else:
                self._close_command_mode()
                self.append_message("system", "Canceled.")
            return

        if lowered == "delete":
            self._queue_model_delete()
            return

        if lowered.startswith("delete "):
            self.append_message("system", "Type 'delete' to remove the highlighted model.")
            return

        if lowered.startswith("dl ") or lowered.startswith("download "):
            parts = text.split(maxsplit=1)
            if len(parts) == 2:
                repo_id = parts[1].strip()
                if not repo_id:
                    self.append_message("system", "Usage: dl <repo_id>")
                    return
                normalized_repo = self._normalize_hf_repo_input(repo_id) or repo_id
                self._queue_model_download(normalized_repo)
                return

        if text.isdigit():
            idx = int(text) - 1
            items = self.command_mode.get("items", [])
            if 0 <= idx < len(items):
                self._apply_model_selection(items[idx])
                self._close_command_mode()
                return

        items = self.command_mode.get("items", [])
        matches = [m for m in items if m.lower() == lowered]
        if matches:
            self._apply_model_selection(matches[0])
            self._close_command_mode()
            return

        # Treat pasted Hugging Face repo ids/URLs as a direct download request.
        repo_id = self._normalize_hf_repo_input(text)
        if repo_id:
            self._queue_model_download(repo_id)
            return

        self.append_message("system", "Unknown model command.")

    def _perform_model_delete(self, model_id: str):
        if not model_id:
            return
        if not model_id.lower().endswith(".gguf"):
            self.append_message("system", "Only downloaded GGUF files can be deleted.")
            return
        import os
        from chatbot.model_manager import ModelManager
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, "shared_models", model_id)
        was_active_model = (self.model == model_id)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.append_message("system", f"Deleted: {model_id}")
                items = self.command_mode.get("items", [])
                display = self.command_mode.get("display", [])
                if model_id in items:
                    idx = items.index(model_id)
                    items.pop(idx)
                    if idx < len(display):
                        display.pop(idx)
                    if items:
                        self.command_mode["index"] = min(idx, len(items) - 1)

                # Ensure deleted models are not kept alive in RAM cache.
                ModelManager.close_all()

                if was_active_model:
                    fallback_model = config.DEFAULT_MODEL
                    if fallback_model == model_id and items:
                        fallback_model = items[0]
                    self.model = fallback_model
                    self.root.title(f"Chatbot - {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
                    self.append_message("system", f"Active model removed. Reverted to: {self.model}")
                    self.update_status(f"Model: {self.model}")
            else:
                self.append_message("system", f"File not found: {file_path}")
        except Exception as e:
            self.append_message("system", f"Delete failed: {e}")

    def _perform_model_download(self, repo_id: str):
        if not repo_id:
            return
        self._start_model_download(repo_id)
        self.append_message("system", f"Downloading: {repo_id}")

    def _enter_api_mode(self):
        self.command_mode = {
            "name": "api",
            "type": "input",
            "api_mode": config.API_MODE,
            "url": config.API_BASE_URL,
            "key": config.API_KEY,
            "model": config.API_MODEL_NAME,
        }
        self._render_api_menu()

    def _render_api_menu(self):
        mask = lambda v: "*" * len(v) if v else "None"
        data = self.command_mode
        text = (
            "API CONFIG\n"
            f"mode: {'on' if data['api_mode'] else 'off'}\n"
            f"url: {data['url'] or 'None'}\n"
            f"key: {mask(data['key'])}\n"
            f"model: {data['model'] or 'None'}\n\n"
            "Commands:\n"
            "  mode on|off\n"
            "  url <base_url>\n"
            "  key <api_key>\n"
            "  model <model_name>\n"
            "  test\n"
            "  save\n"
            "  cancel (Esc)\n"
        )
        self._render_terminal_menu(text)

    def _handle_api_input(self, user_input: str):
        text = user_input.strip()
        lowered = text.lower()
        if not text:
            self._render_api_menu()
            return
        if lowered in {"cancel", "exit", "quit"}:
            self._exit_command_mode(cancelled=True)
            return
        if lowered == "test":
            self._test_api_connection()
            return
        if lowered == "save":
            self._save_api_config()
            self._close_command_mode()
            return

        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            key, value = parts[0].lower(), parts[1].strip()
            if key == "mode":
                self.command_mode["api_mode"] = value.lower() in {"on", "true", "1", "yes"}
            elif key == "url":
                self.command_mode["url"] = value
            elif key == "key":
                self.command_mode["key"] = value
            elif key == "model":
                self.command_mode["model"] = value
            else:
                self.append_message("system", "Unknown API command.")
                return
            self._render_api_menu()
            return
        self.append_message("system", "Invalid command. Use 'save' or 'cancel' or 'test'.")

    def _save_api_config(self):
        data = self.command_mode
        config.API_MODE = data["api_mode"]
        config.API_BASE_URL = data["url"]
        config.API_KEY = data["key"]
        config.API_MODEL_NAME = data["model"]

        from chatbot.model_manager import ModelManager
        ModelManager.close_all()

        mode_str = "External API" if config.API_MODE else "Local Internal"
        self.append_message("system", f"Configuration saved. Switched to {mode_str} Mode.")
        if config.API_MODE:
            self.model = config.API_MODEL_NAME
            self.root.title(f"Chatbot - API: {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")

    def _test_api_connection(self):
        data = self.command_mode
        url = data.get("url", "")
        key = data.get("key", "")
        model = data.get("model", "")

        self.append_message("system", "Testing API connection...")

        def run_test():
            try:
                from chatbot.api_client import OpenAIClientWrapper
                client = OpenAIClientWrapper(url, key, model)
                resp = client.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5
                )
                if resp:
                    self._post_system("API connection successful.")
                else:
                    self._post_system("API connection returned empty response.")
            except Exception as e:
                self._post_system(f"API connection error: {e}")

        threading.Thread(target=run_test, daemon=True).start()

    def _enter_forge_mode(self, args_line: Optional[str] = None):
        self.command_mode = {
            "name": "forge",
            "type": "input",
        }
        if args_line:
            self._run_forge_cli(args_line)
            self._close_command_mode()
            return
        self._render_forge_menu()

    def _render_forge_menu(self):
        text = (
            "FORGE\n"
            "Enter Forge CLI arguments (same as terminal usage).\n"
            "Example:\n"
            "  /path/to/docs -o kb.zim -t \"My Knowledge Base\"\n\n"
            "Type 'cancel' to abort."
        )
        self._render_terminal_menu(text)

    def _handle_forge_input(self, user_input: str):
        text = user_input.strip()
        if not text or text.lower() in {"cancel", "exit", "quit"}:
            self._exit_command_mode(cancelled=True)
            return
        self._run_forge_cli(text)
        self._close_command_mode()

    def _run_forge_cli(self, args_line: str):
        import shlex
        import subprocess
        import sys
        import os

        args = shlex.split(args_line)
        if not args:
            self.append_message("system", "No Forge arguments provided.")
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        forge_path = os.path.join(project_root, "forge.py")

        self.append_message("system", f"Running Forge: {args_line}")

        def run():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", forge_path, *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if proc.stdout:
                    for line in proc.stdout:
                        if line:
                            self._post_system(f"[forge] {line.rstrip()}")
                if proc.stderr:
                    for line in proc.stderr:
                        if line:
                            self._post_system(f"[forge][err] {line.rstrip()}")
                proc.wait()
                self._post_system(f"Forge finished with code {proc.returncode}.")
            except Exception as e:
                self._post_system(f"Forge error: {e}")

        threading.Thread(target=run, daemon=True).start()
    
    def get_installed_models(self) -> List[Tuple[str, ModelPlatform]]:
        """Get list of supported local models from config and shared_models directory."""
        models: List[Tuple[str, ModelPlatform]] = []
        
        # 1. Add models explicitly defined in Config
        if hasattr(config, 'MODEL_QWEN_3B'):
            models.append((config.MODEL_QWEN_3B, ModelPlatform.LOCAL))
            
        # 2. Scan shared_models directory for manually downloaded GGUFs
        import os
        import glob
        
        # Determine shared_models path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        shared_models_dir = os.path.join(project_root, "shared_models")
        
        # We only look for .gguf files
        if os.path.exists(shared_models_dir):
            gguf_files = glob.glob(os.path.join(shared_models_dir, "*.gguf"))
            # Sort by modification time (newest first) for convenience
            gguf_files.sort(key=os.path.getmtime, reverse=True)
            
            for file_path in gguf_files:
                filename = os.path.basename(file_path)
                
                # Filter out non-primary shards of split models
                # Only show: single-file models OR the first shard (-00001-of-XXXXX)
                # Support both 5-digit (00001) and variable-length (0001, 001) formats
                import re
                shard_match = re.search(r'-(\d+)-of-(\d+)\.gguf$', filename)
                if shard_match:
                    shard_num = int(shard_match.group(1))
                    if shard_num != 1:
                        # Skip non-primary shards (00002, 00003, etc.)
                        continue
                
                # Avoid duplicates if they match the config model exactly
                # (Config model usually is a repo_id, filename is, well, a filename)
                
                # We use the filename as the unique ID for these User-Downloaded models
                # This works if we update ModelManager to handle filenames as inputs
                models.append((filename, ModelPlatform.LOCAL))
                
        # Deduplicate by name just in case
        unique_models = []
        seen = set()
        for m, p in models:
            if m not in seen:
                unique_models.append((m, p))
                seen.add(m)
                
        return unique_models
    
    def show_model_menu(self):
        """Show model selection menu."""
        models = self.get_installed_models()
        if not models:
            self.messagebox.showwarning(
                "No Models",
                "No models found in config."
            )
            return
        
        # NOTE: self.tk.Toplevel now automatically inherits *Background/*Foreground
        # from options_add set in apply_theme(), so minimal explicit config is needed.
        model_window = self.tk.Toplevel(self.root)
        model_window.title("Select Model")
        model_window.geometry("500x600")
        model_window.transient(self.root)
        model_window.grab_set()
        
        # Explicitly configure bg for container window to be safe, though option_add handles children
        bg_color = "#2A2A2A" if self.dark_mode else "#FFFFFF"
        model_window.configure(bg=bg_color)
        
        title_label = self.ttk.Label(model_window, text="Select Model", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        current_label = self.ttk.Label(model_window, text=f"Current: {self.model}", font=("Arial", 10))
        current_label.pack(pady=5)
        
        listbox_frame = self.ttk.Frame(model_window)
        listbox_frame.pack(fill=self.tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = self.ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        # Listbox will inherit colors from option_add in apply_theme
        model_listbox = self.tk.Listbox(
            listbox_frame,
            font=("Arial", 11),
            selectbackground="#333333" if self.dark_mode else "lightblue",
            selectforeground="#FFFFFF" if self.dark_mode else "#000000",
            yscrollcommand=scrollbar.set,
            activestyle="none", borderwidth=0, highlightthickness=1
        )
        model_listbox.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.config(command=model_listbox.yview)
        
        selected_index = 0
        model_list: List[str] = []
        
        for model_name, platform in models:
            # Create friendly display name
            display_name = model_name.split('/')[-1] if '/' in model_name else model_name
            
            # Remove shard suffixes from split models for cleaner display
            import re
            display_name = re.sub(r'-(\d+)-of-(\d+)\.gguf$', '.gguf', display_name)
            
            # Apply human-readable labels for known model families
            if "qwen2.5-3b" in display_name.lower():
                quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
                quant = quant_match.group(1).upper() if quant_match else "Unknown"
                display_name = f"Qwen 2.5 3B ({quant})"
            elif "qwen2.5-7b" in display_name.lower():
                # Extract quantization if present
                quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
                quant = quant_match.group(1).upper() if quant_match else "Unknown"
                display_name = f"Qwen 2.5 7B ({quant})"
            elif "qwen2.5-1.5b" in display_name.lower() or "qwen2.5-1b" in display_name.lower():
                quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
                quant = quant_match.group(1).upper() if quant_match else "Unknown"
                display_name = f"Qwen 2.5 1.5B ({quant})"
            elif "llama-3.2-3b" in display_name.lower():
                quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
                quant = quant_match.group(1).upper() if quant_match else "Unknown"
                display_name = f"Llama 3.2 3B ({quant})"
            elif "llama-3" in display_name.lower() and "8b" in display_name.lower():
                quant_match = re.search(r'(q\d+_k_[msl]|q[2-8]_0)', display_name, re.IGNORECASE)
                quant = quant_match.group(1).upper() if quant_match else "Unknown"
                display_name = f"Llama 3 8B ({quant})"
                
            model_listbox.insert(self.tk.END, display_name)
            model_list.append(model_name)
            if model_name == self.model:
                selected_index = len(model_list) - 1
        
        if selected_index < len(model_list):
            model_listbox.selection_set(selected_index)
            model_listbox.see(selected_index)
        model_listbox.focus_set()
        
        instructions = self.ttk.Label(
            model_window,
            text="↑↓ Navigate  |  Enter: Select  |  Esc: Cancel",
            font=("Arial", 9)
        )
        instructions.pack(pady=5)
        
        button_frame = self.ttk.Frame(model_window)
        button_frame.pack(pady=10)
        
        def select_model():
            selection = model_listbox.curselection()
            if selection and selection[0] < len(model_list):
                new_model = model_list[selection[0]]
                if new_model != self.model:
                    self.model = new_model
                    self.root.title(f"Chatbot - {new_model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
                    self.append_message("system", f"Model changed to: {new_model}")
                    self.update_status(f"Model: {new_model}")
                model_window.destroy()
        
        def cancel():
            model_window.destroy()
        
        def delete_model():
            selection = model_listbox.curselection()
            if not selection:
                self.messagebox.showwarning("No Selection", "Please select a model to delete.")
                return
            
            selected_idx = selection[0]
            model_id = model_list[selected_idx]
            was_active_model = (self.model == model_id)
            
            # Only allow deletion of local .gguf files, not config-defined models
            if not model_id.lower().endswith(".gguf"):
                self.messagebox.showwarning("Cannot Delete", "Only downloaded GGUF files can be deleted.\nConfig-defined models cannot be removed from here.")
                return
            
            # Confirm deletion
            confirm = self.tk.messagebox.askyesno(
                "Confirm Delete",
                f"Delete model file:\n{model_id}\n\nThis cannot be undone."
            )
            if not confirm:
                return
            
            # Actually delete the file
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            file_path = os.path.join(project_root, "shared_models", model_id)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.append_message("system", f"Deleted: {model_id}")
                    # Refresh list
                    model_listbox.delete(selected_idx)
                    model_list.pop(selected_idx)

                    from chatbot.model_manager import ModelManager
                    ModelManager.close_all()

                    if was_active_model:
                        fallback_model = config.DEFAULT_MODEL
                        if fallback_model == model_id and model_list:
                            fallback_model = model_list[0]
                        self.model = fallback_model
                        self.root.title(f"Chatbot - {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
                        self.append_message("system", f"Active model removed. Reverted to: {self.model}")
                        self.update_status(f"Model: {self.model}")
                else:
                    self.messagebox.showerror("Error", f"File not found:\n{file_path}")
            except Exception as e:
                self.messagebox.showerror("Delete Failed", f"Could not delete file:\n{e}")
        
        select_btn = self.ttk.Button(button_frame, text="Select", command=select_model, style="Accent.TButton")
        select_btn.pack(side=self.tk.LEFT, padx=5)
        delete_btn = self.ttk.Button(button_frame, text="Delete", command=delete_model)
        delete_btn.pack(side=self.tk.LEFT, padx=5)
        cancel_btn = self.ttk.Button(button_frame, text="Cancel", command=cancel)
        cancel_btn.pack(side=self.tk.LEFT, padx=5)
        
        def on_listbox_key(event):
            if event.keysym == "Return":
                select_model()
                return "break"
            elif event.keysym == "Escape":
                cancel()
                return "break"
        
        def on_window_key(event):
            if event.keysym == "Return":
                select_model()
                return "break"
            elif event.keysym == "Escape":
                cancel()
                return "break"
        
        model_listbox.bind("<KeyPress>", on_listbox_key)
        model_window.bind("<KeyPress>", on_window_key)
        model_listbox.focus_set()

        # Separator
        self.ttk.Separator(model_window, orient='horizontal').pack(fill='x', pady=15)

        # Download Section
        dl_frame = self.ttk.Frame(model_window)
        dl_frame.pack(fill='x', padx=20, pady=(0, 20))

        self.ttk.Label(dl_frame, text="Download New Models (GGUF Only)", font=("Arial", 11, "bold")).pack(anchor='w')
        self.ttk.Label(dl_frame, text="Hermit requires GGUF format. Look for repos with 'GGUF' in the name.", font=("Arial", 9)).pack(anchor='w', pady=(0, 5))
        
        search_frame = self.ttk.Frame(dl_frame)
        search_frame.pack(fill='x', pady=5)
        
        search_var = self.tk.StringVar()
        search_entry = self.tk.Entry(search_frame, textvariable=search_var) # Global theme applies here
        search_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Placeholder logic for search entry
        search_entry.insert(0, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        def on_focus_in(e):
             if "GGUF" in search_entry.get() and "/" in search_entry.get():
                 # Only clear if it's the placeholder-like format
                 if search_entry.get() == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF":
                     search_entry.delete(0, 'end')
        def on_focus_out(e):
             if not search_entry.get():
                search_entry.insert(0, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        
        search_entry.bind("<FocusIn>", on_focus_in)
        search_entry.bind("<FocusOut>", on_focus_out)

        def download_action():
            repo_id = search_var.get().strip()
            if not repo_id or repo_id == "TheBloke/Mistral-7B-Instruct-v0.2-GGUF":
                self.messagebox.showwarning("Input Required", "Please enter a Hugging Face Repo ID.\n\nExample: TheBloke/Llama-2-7B-Chat-GGUF")
                return
            
            # Warn if it doesn't look like a GGUF repo
            if "gguf" not in repo_id.lower():
                warn = self.tk.messagebox.askyesno(
                    "Warning: Possibly Incompatible",
                    f"The repo '{repo_id}' does not contain 'GGUF' in its name.\n\n"
                    "Hermit ONLY works with GGUF format models.\n"
                    "Non-GGUF models (SafeTensors, MLX, etc.) will NOT work.\n\n"
                    "Continue anyway?"
                )
                if not warn:
                    return

            # Confirm download
            confirm = self.tk.messagebox.askyesno(
                "Confirm Download",
                f"Download {repo_id}?\n\nThe system will find the best Q4_K_M or similar quantization."
            )
            if not confirm:
                return
            
            model_window.destroy()
            self._start_model_download(repo_id)

        self.ttk.Button(search_frame, text="Download", command=download_action).pack(side='right')

    def _start_model_download(self, repo_id: str):
        """Handle the background download process."""
        self.append_message("system", f"Starting download for {repo_id}...")
        
        def run_dl():
            try:
                # Use ModelManager's existing logic to 'ensure' path, which triggers download
                from chatbot.model_manager import ModelManager
                # We force a fresh download check essentially by calling ensure_model_path
                # Note: ModelManager might need specific file patterns if generic repo is given.
                # Ideally ModelManager handles 'user/repo' by finding best GGUF.
                
                # Check if ModelManager has a smart downloader or we rely on default file spec.
                # If the user just gave a repo, we might need to be smart.
                # For now, let's assume ModelManager handles standard logic.
                
                path = ModelManager.ensure_model_path(repo_id)
                self.root.after(0, lambda: self.append_message("system", f"Download complete: {repo_id}"))
                self.root.after(0, lambda: self.update_status(f"Installed: {repo_id}"))
                
                # Trigger a refresh of the model menu (optional, or just ready for next use)
                
            except Exception as e:
                 self.root.after(0, lambda: self.messagebox.showerror("Download Error", f"Failed to download {repo_id}:\n{e}"))
                 self.root.after(0, lambda: self.append_message("system", f"Download failed: {e}"))
        
        threading.Thread(target=run_dl, daemon=True).start()

    def _build_themes(self) -> dict:
        """Return theme palette definitions."""
        return {
            "Noir": {
                "dark": True,
                "bg": "#2A2A2A",
                "fg": "#E0E0E0",
                "input_bg": "#1E1E1E",
                "input_fg": "#FFFFFF",
                "accent": "#808080",
                "button_bg": "#333333",
                "button_fg": "#FFFFFF",
                "border": "#444444",
                "concept": "#81D4FA",
                "bubble_bg": "#1E1E1E",
                "muted_fg": "#808080",
            },
            "Paper": {
                "dark": False,
                "bg": "#FFFFFF",
                "fg": "#000000",
                "input_bg": "#F5F5F5",
                "input_fg": "#000000",
                "accent": "#666666",
                "button_bg": "#E0E0E0",
                "button_fg": "#000000",
                "border": "#CCCCCC",
                "concept": "#0277BD",
                "bubble_bg": "#E0E0E0",
                "muted_fg": "#606060",
            },
            "Slate": {
                "dark": True,
                "bg": "#24272B",
                "fg": "#E6E8EA",
                "input_bg": "#1D2024",
                "input_fg": "#F3F4F5",
                "accent": "#4F7CAC",
                "button_bg": "#2E3237",
                "button_fg": "#F1F1F1",
                "border": "#3A3F45",
                "concept": "#7BB7FF",
                "bubble_bg": "#1D2024",
                "muted_fg": "#9AA3AD",
            },
            "Forest": {
                "dark": True,
                "bg": "#1F2420",
                "fg": "#E5E7E1",
                "input_bg": "#171B18",
                "input_fg": "#F1F3EE",
                "accent": "#6AA84F",
                "button_bg": "#2A302A",
                "button_fg": "#F1F3EE",
                "border": "#3A433B",
                "concept": "#9AD18B",
                "bubble_bg": "#1A1F1B",
                "muted_fg": "#9AA88E",
            },
            "Sunrise": {
                "dark": False,
                "bg": "#FFF6ED",
                "fg": "#2B2520",
                "input_bg": "#FFF1E2",
                "input_fg": "#2B2520",
                "accent": "#D97B30",
                "button_bg": "#F2E4D7",
                "button_fg": "#2B2520",
                "border": "#E2CDB8",
                "concept": "#B85B1E",
                "bubble_bg": "#F2E4D7",
                "muted_fg": "#6B5B4C",
            },
            "Nord": {
                "dark": True,
                "bg": "#2B303A",
                "fg": "#ECEFF4",
                "input_bg": "#242933",
                "input_fg": "#ECEFF4",
                "accent": "#88C0D0",
                "button_bg": "#343A46",
                "button_fg": "#ECEFF4",
                "border": "#3B4252",
                "concept": "#8FBCBB",
                "bubble_bg": "#232833",
                "muted_fg": "#A7B1C2",
            },
            "Lagoon": {
                "dark": True,
                "bg": "#1E2A2F",
                "fg": "#E3F0F0",
                "input_bg": "#162026",
                "input_fg": "#E3F0F0",
                "accent": "#4FB6B2",
                "button_bg": "#26343A",
                "button_fg": "#E3F0F0",
                "border": "#304249",
                "concept": "#6AD7D1",
                "bubble_bg": "#151F24",
                "muted_fg": "#93A8A8",
            },
            "Cinder": {
                "dark": True,
                "bg": "#2C2522",
                "fg": "#F1E7DF",
                "input_bg": "#241E1B",
                "input_fg": "#F5EEE7",
                "accent": "#C27B5E",
                "button_bg": "#3A2F2A",
                "button_fg": "#F1E7DF",
                "border": "#4A3A34",
                "concept": "#E0A685",
                "bubble_bg": "#231D1A",
                "muted_fg": "#B9A79B",
            },
            "Ivory": {
                "dark": False,
                "bg": "#FAF7F2",
                "fg": "#2B2A28",
                "input_bg": "#F2ECE4",
                "input_fg": "#2B2A28",
                "accent": "#B07D62",
                "button_bg": "#E7DDD2",
                "button_fg": "#2B2A28",
                "border": "#D7C9BC",
                "concept": "#8A5A44",
                "bubble_bg": "#EDE2D6",
                "muted_fg": "#6A5A4D",
            },
            "Mint": {
                "dark": False,
                "bg": "#F4FBF7",
                "fg": "#203327",
                "input_bg": "#EAF4EE",
                "input_fg": "#203327",
                "accent": "#4E9A6B",
                "button_bg": "#DCEDE2",
                "button_fg": "#203327",
                "border": "#C7DED0",
                "concept": "#2E7D5A",
                "bubble_bg": "#E2F0E7",
                "muted_fg": "#5D6F63",
            },
            "High Contrast Dark": {
                "dark": True,
                "bg": "#000000",
                "fg": "#FFFFFF",
                "input_bg": "#000000",
                "input_fg": "#FFFFFF",
                "accent": "#FFD400",
                "button_bg": "#111111",
                "button_fg": "#FFFFFF",
                "border": "#FFFFFF",
                "concept": "#00E5FF",
                "bubble_bg": "#000000",
                "muted_fg": "#CFCFCF",
            },
            "High Contrast Light": {
                "dark": False,
                "bg": "#FFFFFF",
                "fg": "#000000",
                "input_bg": "#FFFFFF",
                "input_fg": "#000000",
                "accent": "#005FCC",
                "button_bg": "#F5F5F5",
                "button_fg": "#000000",
                "border": "#000000",
                "concept": "#003B80",
                "bubble_bg": "#F0F0F0",
                "muted_fg": "#2A2A2A",
            },
            "Amber Terminal": {
                "dark": True,
                "bg": "#000000",
                "fg": "#FFC34D",
                "input_bg": "#000000",
                "input_fg": "#FFD37A",
                "accent": "#FFB300",
                "button_bg": "#1A1200",
                "button_fg": "#FFD37A",
                "border": "#A66B00",
                "concept": "#FFE082",
                "bubble_bg": "#120D00",
                "muted_fg": "#D7A64A",
            },
            "Cyan Terminal": {
                "dark": True,
                "bg": "#000000",
                "fg": "#7CF9FF",
                "input_bg": "#000000",
                "input_fg": "#B5FDFF",
                "accent": "#00C2D1",
                "button_bg": "#00161A",
                "button_fg": "#B5FDFF",
                "border": "#00A7B5",
                "concept": "#E0FEFF",
                "bubble_bg": "#001014",
                "muted_fg": "#7EC7CC",
            },
            "Hacker": {
                "dark": True,
                "bg": "#000000",
                "fg": "#00FF66",
                "input_bg": "#000000",
                "input_fg": "#76FFB1",
                "accent": "#00CC55",
                "button_bg": "#001A0D",
                "button_fg": "#76FFB1",
                "border": "#00A347",
                "concept": "#B3FFCC",
                "bubble_bg": "#001108",
                "muted_fg": "#4FD68C",
            },
        }

    def _resolve_theme_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        token = name.strip().lower()
        aliases = {
            "dark": "Noir",
            "light": "Paper",
            "default": "Noir",
            "classic": "Noir",
            "hc-dark": "High Contrast Dark",
            "hc-light": "High Contrast Light",
            "amber": "Amber Terminal",
            "cyan": "Cyan Terminal",
            "hacker": "Hacker",
            "matrix": "Hacker",
            "green": "Hacker",
        }
        if token in aliases:
            return aliases[token]
        for theme_name in self.themes.keys():
            if theme_name.lower() == token or theme_name.lower().startswith(token):
                return theme_name
        return None

    def set_theme(self, name: str, persist: bool = True) -> bool:
        resolved = self._resolve_theme_name(name)
        if not resolved or resolved not in self.themes:
            return False
        self.theme_name = resolved
        self._theme_palette = self.themes[resolved]
        self.dark_mode = self._theme_palette["dark"]
        self.apply_theme()
        if persist:
            save_theme_name(resolved)
        return True

    def toggle_theme(self):
        """Toggle between Dark and Light mode."""
        if self.dark_mode:
            self.set_theme("Paper")
        else:
            self.set_theme("Noir")
    
    def apply_theme(self):
        """Apply dark/light theme."""
        style = self.ttk.Style()
        style.theme_use('clam')
        palette = self.themes.get(self.theme_name, self.themes["Noir"])
        self._theme_palette = palette
        self.dark_mode = palette["dark"]

        bg_color = palette["bg"]
        fg_color = palette["fg"]
        input_bg = palette["input_bg"]
        input_fg = palette["input_fg"]
        accent_color = palette["accent"]
        button_bg = palette["button_bg"]
        button_fg = palette["button_fg"]
        border_color = palette["border"]
        concept_color = palette["concept"]

        # Global option database for consistency across all standard widgets
        self.root.option_add("*Background", bg_color)
        self.root.option_add("*Foreground", fg_color)
        self.root.option_add("*Entry.Background", input_bg)
        self.root.option_add("*Entry.Foreground", input_fg)
        self.root.option_add("*Listbox.Background", input_bg)
        self.root.option_add("*Listbox.Foreground", input_fg)
        self.root.option_add("*Text.Background", bg_color)
        self.root.option_add("*Text.Foreground", fg_color)
        self.root.option_add("*Button.Background", button_bg)
        self.root.option_add("*Button.Foreground", button_fg)

        self.root.configure(bg=bg_color)

        style.configure(".", background=bg_color, foreground=fg_color, font=("Arial", 10))
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        
        style.configure("TButton",
            background=button_bg, foreground=button_fg,
            borderwidth=0, focuscolor="none", padding=(15, 8)
        )
        style.map("TButton",
            background=[('active', accent_color)],
            foreground=[('active', '#FFFFFF')]
        )
        
        style.configure("Accent.TButton",
            background=accent_color, foreground="#FFFFFF",
            font=("Arial", 10, "bold")
        )
        style.map("Accent.TButton",
            background=[('active', button_bg)],
            foreground=[('active', button_fg)]
        )
        
        style.configure("Vertical.TScrollbar",
            gripcount=0, background=button_bg,
            darkcolor=bg_color, lightcolor=bg_color,
            troughcolor=bg_color, bordercolor=bg_color,
            arrowcolor=fg_color
        )
        style.map("Vertical.TScrollbar",
            background=[('active', accent_color), ('!disabled', button_bg)],
            arrowcolor=[('active', accent_color)]
        )
        
        self.chat_display.configure(
            bg=bg_color, fg=fg_color, insertbackground=fg_color,
            selectbackground=accent_color, selectforeground="#FFFFFF"
        )
        
        self.input_entry.configure(
            bg=bg_color, fg=input_fg, insertbackground=fg_color,
            highlightbackground=border_color, highlightcolor=accent_color,
            selectbackground=accent_color, selectforeground="#FFFFFF"
        )
        
        self.autocomplete_listbox.configure(
            bg=input_bg, fg=input_fg,
            selectbackground=accent_color, selectforeground="#FFFFFF",
            highlightthickness=1, highlightbackground=border_color,
            borderwidth=1, relief=self.tk.SOLID
        )
        
        if hasattr(self, 'send_canvas') and hasattr(self, '_draw_send_button'):
            self.send_canvas_color = border_color
            self.send_canvas_hover_color = "#FFFFFF" if self.dark_mode else "#000000"
            self._draw_send_button(self.send_canvas, self.send_canvas_color)
            self.send_canvas.configure(bg=bg_color)
        
        for tag in self.chat_display.tag_names():
            if tag.startswith("concept"):
                self.chat_display.tag_config(tag, foreground=concept_color, underline=True)
            elif tag.startswith("link"):
                self.chat_display.tag_config(tag, foreground=concept_color, underline=True)
            elif tag.endswith("_message") or "_message_" in tag:
                role = "user" if tag.startswith("user") else "ai" if tag.startswith("ai") else "system"
                self._configure_message_tag(tag, role)

        if self.command_mode and self._terminal_menu_active:
            self._render_current_command_menu()
    
    def _configure_message_tag(self, tag_name: str, role: str):
        """Configure styling for message border tags."""
        border_bg = self._theme_palette.get("bubble_bg", "#1E1E1E" if self.dark_mode else "#E0E0E0")
        
        if role == "ai":
            modern_font = ("Georgia", 11)
        else:
            modern_font = None
        
        config_options = {
            "background": border_bg,
            "lmargin1": 12, "lmargin2": 12, "rmargin": 12,
            "spacing1": 6, "spacing2": 3, "spacing3": 6
        }
        
        if modern_font:
            config_options["font"] = modern_font
        
        self.chat_display.tag_config(tag_name, **config_options)
    
    def get_autocomplete_suggestions(self, text: str) -> List[str]:
        """Get autocomplete suggestions."""
        suggestions: List[str] = []
        text_lower = text.lower()
        
        commands = ["/help", "/exit", "/clear", "/themes", "/model", "/status", "/api", "/forge"]
        
        if text.startswith("/"):
            for cmd in commands:
                if cmd.lower().startswith(text_lower):
                    suggestions.append(cmd)
        else:
            seen = set()
            for query in reversed(self.query_history):
                if query.lower().startswith(text_lower) and query not in seen and len(query) > len(text):
                    suggestions.append(query)
                    seen.add(query)
                if len(suggestions) >= 10:
                    break
        
        return suggestions[:10]
    
    def show_autocomplete(self, suggestions: List[str]):
        """Show autocomplete dropdown."""
        if not suggestions:
            self.hide_autocomplete()
            return
        
        self.autocomplete_suggestions = suggestions
        self.autocomplete_listbox.delete(0, self.tk.END)
        for item in suggestions:
            self.autocomplete_listbox.insert(self.tk.END, item)
        
        self.root.update_idletasks()
        
        entry_x = self.input_entry.winfo_rootx() - self.root.winfo_rootx()
        entry_y = self.input_entry.winfo_rooty() - self.root.winfo_rooty() + self.input_entry.winfo_height() + 2
        
        listbox_width = self.input_entry.winfo_width()
        listbox_height = min(150, max(25, len(suggestions) * 22 + 4))
        
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        if root_width > 100 and root_height > 100:
            if entry_x + listbox_width > root_width - 10:
                entry_x = max(10, root_width - listbox_width - 10)
            if entry_y + listbox_height > root_height - 10:
                entry_y = max(10, entry_y - self.input_entry.winfo_height() - listbox_height - 2)
        
        self.autocomplete_listbox.place(
            x=entry_x, y=entry_y,
            width=listbox_width, height=listbox_height
        )
        self.autocomplete_listbox.lift()
        self.input_entry.focus_set()
        self.autocomplete_active = True
        self.autocomplete_selected_index = -1
    
    def hide_autocomplete(self):
        """Hide autocomplete dropdown."""
        self.autocomplete_listbox.place_forget()
        self.autocomplete_active = False
        self.autocomplete_suggestions = []
        self.autocomplete_selected_index = -1
    
    def on_input_return(self, event):
        """Handle Return key."""
        if self.autocomplete_active and self.autocomplete_suggestions:
            suggestion = self.autocomplete_suggestions[0]
            self.input_entry.delete(0, self.tk.END)
            self.input_entry.insert(0, suggestion)
            self.hide_autocomplete()
            self.on_send()
            return "break"
        else:
            return self.on_send(event)
    
    def on_input_key(self, event):
        """Handle key release in input entry."""
        if self.command_mode:
            self.hide_autocomplete()
            return
        if event.keysym in ["Up", "Down", "Tab", "Return", "Escape"]:
            return
        
        text = self.input_entry.get()
        if len(text) < 1:
            self.hide_autocomplete()
            return
        
        suggestions = self.get_autocomplete_suggestions(text)
        if suggestions:
            self.show_autocomplete(suggestions)
        else:
            self.hide_autocomplete()
    
    def on_autocomplete_nav(self, event):
        """Handle Up/Down arrow navigation."""
        if self.command_mode:
            delta = -1 if event.keysym == "Up" else 1
            self._command_menu_nav(delta)
            return "break"
        if not self.autocomplete_active or not self.autocomplete_suggestions:
            return None
        
        if event.keysym == "Up":
            if self.autocomplete_selected_index > 0:
                self.autocomplete_selected_index -= 1
            elif self.autocomplete_selected_index == -1:
                self.autocomplete_selected_index = len(self.autocomplete_suggestions) - 1
        elif event.keysym == "Down":
            if self.autocomplete_selected_index < len(self.autocomplete_suggestions) - 1:
                self.autocomplete_selected_index += 1
            else:
                self.autocomplete_selected_index = 0
        
        if 0 <= self.autocomplete_selected_index < len(self.autocomplete_suggestions):
            self.autocomplete_listbox.selection_clear(0, self.tk.END)
            self.autocomplete_listbox.selection_set(self.autocomplete_selected_index)
            self.autocomplete_listbox.see(self.autocomplete_selected_index)
        
        return "break"
    
    def on_autocomplete_select(self, event):
        """Select autocomplete suggestion."""
        if not self.autocomplete_active:
            if event.keysym == "Tab":
                return None
            return None
        
        selected_idx = self.autocomplete_selected_index
        if selected_idx == -1:
            selected_idx = 0
        
        if 0 <= selected_idx < len(self.autocomplete_suggestions):
            suggestion = self.autocomplete_suggestions[selected_idx]
            self.input_entry.delete(0, self.tk.END)
            self.input_entry.insert(0, suggestion)
            self.hide_autocomplete()
            
            if event.keysym == "Return" and hasattr(event, 'widget') and event.widget == self.autocomplete_listbox:
                self.on_send()
        
        return "break"
    
    def on_autocomplete_click(self, event):
        """Handle mouse click on autocomplete."""
        if not self.autocomplete_active:
            return
        
        selection = self.autocomplete_listbox.curselection()
        if selection:
            idx = selection[0]
            suggestion = self.autocomplete_suggestions[idx]
            self.input_entry.delete(0, self.tk.END)
            self.input_entry.insert(0, suggestion)
            self.hide_autocomplete()
            self.input_entry.focus_set()
    
    def on_autocomplete_close(self, event):
        """Close autocomplete with Escape."""
        self.hide_autocomplete()
        return "break"
    
    def on_input_focus_out(self, event):
        """Hide autocomplete when input loses focus."""
        if event.widget != self.input_entry:
            return
        self.root.after_idle(lambda: self._check_focus_for_autocomplete())
    
    def _check_focus_for_autocomplete(self):
        """Check if focus is on autocomplete listbox."""
        try:
            focused_widget = self.root.focus_get()
            if focused_widget != self.autocomplete_listbox and focused_widget != self.input_entry:
                self.hide_autocomplete()
        except KeyError:
            # Handle case where widget (e.g. messagebox) is destroyed but focus reference lingers
            pass
    
    def append_message(self, role: str, content: str, is_concept: bool = False):
        """Append message to chat display."""
        self.chat_display.insert(self.tk.END, "\n")
        
        message_start = self.chat_display.index(self.tk.END + "-1c")
        
        prefix = "System: " if role == "system" else ""
        if prefix:
            self.chat_display.insert(self.tk.END, prefix)
        
        if is_concept:
            start = self.chat_display.index(self.tk.END + "-1c")
            self.chat_display.insert(self.tk.END, content)
            end = self.chat_display.index(self.tk.END + "-1c")
            self.chat_display.tag_add("concept", start, end)
            concept_color = "#5DB9FF" if self.dark_mode else "blue"
            self.chat_display.tag_config("concept", foreground=concept_color, underline=True)
        else:
            self.chat_display.insert(self.tk.END, content)
        
        message_end = self.chat_display.index(self.tk.END + "-1c")
        
        padding = "    "
        self.chat_display.insert(self.tk.END, padding)
        message_end_with_padding = self.chat_display.index(self.tk.END + "-1c")
        
        tag_name = f"{role}_message_{id(self)}"
        self.chat_display.tag_add(tag_name, message_start, message_end_with_padding)
        self._configure_message_tag(tag_name, role)
        
        self.chat_display.insert(self.tk.END, "\n\n")
        self.chat_display.see(self.tk.END)
    
    def append_links(self, query: str, links: List[dict]):
        """Append search results as clickable links."""
        self.chat_display.insert(self.tk.END, "\n")
        
        message_start = self.chat_display.index(self.tk.END + "-1c")
        
        prefix = f"Search Results for '{query}':\n"
        self.chat_display.insert(self.tk.END, prefix)
        
        for i, link in enumerate(links, 1):
            title = link.get('title', 'Unknown Title')
            score = link.get('score', 0.0)
            snippet = link.get('snippet', '')
            path = link.get('path', '')
            
            # Create clickable link
            link_text = f"\n{i}. {title} (Score: {score:.3f})\n"
            if snippet:
                link_text += f"   {snippet[:150]}{'...' if len(snippet) > 150 else ''}\n"
            
            start_pos = self.chat_display.index(self.tk.END + "-1c")
            self.chat_display.insert(self.tk.END, link_text)
            end_pos = self.chat_display.index(self.tk.END + "-1c")
            
            # Make the title clickable
            link_tag = f"link_{i}_{id(self)}"
            self.chat_display.tag_add(link_tag, start_pos, end_pos)
            self.chat_display.tag_config(link_tag, foreground="#5DB9FF" if self.dark_mode else "blue", underline=True)
            
            # Store the path and source ZIM for this link
            highlight_terms = link.get('search_context', {}).get('entities', [])
            source_zim = link.get('metadata', {}).get('source_zim', None)
            self.chat_display.tag_bind(link_tag, "<Button-1>", lambda e, p=path, ht=highlight_terms, sz=source_zim: self.open_zim_article(p, highlight_terms=ht, source_zim=sz))
            self.chat_display.tag_bind(link_tag, "<Enter>", lambda e: self.chat_display.config(cursor="hand2"))
            self.chat_display.tag_bind(link_tag, "<Leave>", lambda e: self.chat_display.config(cursor=""))
        
        message_end = self.chat_display.index(self.tk.END + "-1c")
        
        tag_name = f"links_message_{id(self)}"
        self.chat_display.tag_add(tag_name, message_start, message_end)
        self._configure_message_tag(tag_name, "system")
        
        self.chat_display.insert(self.tk.END, "\n\n")
        self.chat_display.see(self.tk.END)
    
    def open_zim_article(self, path, highlight_terms=None, source_zim=None):
        """
        Open a ZIM article in a new window.
        Multi-ZIM aware: uses source_zim if provided, otherwise searches all ZIMs.
        """
        print(f"[GUI] Opening ZIM path: '{path}' (source: {source_zim or 'auto-detect'})")
        try:
            try:
                import libzim
            except ImportError:
                self.messagebox.showerror("Error", "libzim not installed")
                return

            from chatbot.rag import TextProcessor
            import os
            
            # === MULTI-ZIM SUPPORT ===
            # If source_zim is provided, use it directly; otherwise search all ZIMs
            zim_files_to_try = []
            
            if source_zim and os.path.exists(source_zim):
                zim_files_to_try = [source_zim]
            else:
                # Fall back to discovering all ZIM files
                zim_files_to_try = [os.path.abspath(f) for f in os.listdir('.') if f.endswith('.zim')]
            
            if not zim_files_to_try:
                self.messagebox.showerror("Error", "No ZIM files found")
                return
            
            entry = None
            zim = None
            used_zim = None
            
            # Helper to try finding an entry in a specific ZIM
            def try_find(archive, p):
                try:
                    return archive.get_entry_by_path(p)
                except:
                    return None

            # Try each ZIM file until we find the article
            for zim_file in zim_files_to_try:
                try:
                    zim = libzim.Archive(zim_file)
                except Exception as e:
                    print(f"[GUI] Failed to open {zim_file}: {e}")
                    continue
                
                print(f"[GUI] Searching in: {os.path.basename(zim_file)}")
                
                # Strategy 1: Direct path
                entry = try_find(zim, path)
                
                # Strategy 2: Title lookup
                if not entry:
                    try:
                        entry = zim.get_entry_by_title(path)
                    except:
                        pass

                # Strategy 3: Variations (Smart Fallback)
                if not entry:
                    variations = []
                    # Common ZIM variations
                    if ' ' in path:
                        variations.append(path.replace(' ', '_'))
                    if '_' in path:
                        variations.append(path.replace('_', ' '))
                    
                    # Title Case
                    variations.append(path.title())
                    if ' ' in path:
                        variations.append(path.title().replace(' ', '_'))
                    
                    # Slash handling
                    paths_to_try = [path] + variations
                    
                    for candidate in paths_to_try:
                        # Try raw, with leading slash, without leading slash
                        attempts = [candidate]
                        if not candidate.startswith('/'): attempts.append('/' + candidate)
                        if candidate.startswith('/'): attempts.append(candidate[1:])
                        
                        for attempt in attempts:
                            entry = try_find(zim, attempt)
                            if entry: 
                                print(f"[GUI] Found match: '{attempt}' in {os.path.basename(zim_file)}")
                                break
                        if entry: break
                
                if entry:
                    used_zim = zim_file
                    break  # Found in this ZIM, stop searching

            if not entry or entry.is_redirect:
                print(f"[GUI] Article not found: {path} (and variations)")
                self.messagebox.showerror("Error", f"Article not found: {path}")
                return
            
            item = entry.get_item()
            if item.mimetype != 'text/html':
                self.messagebox.showerror("Error", f"Not a text article ({item.mimetype})")
                return
            
            # Extract and display content (Formatted)
            content = TextProcessor.extract_renderable_text(item.content)
            
            # Create article viewer window
            article_window = self.tk.Toplevel(self.root)
            article_window.title(f"Article: {entry.title}")
            article_window.geometry("800x600")
            
            if self.dark_mode:
                bg_color = "#1E1E1E"
                fg_color = "#E0E0E0"
                h1_color = "#5DB9FF"
                h2_color = "#81C784"
                highlight_bg = "#555500" # Dark yellow
                highlight_fg = "#FFFFFF"
            else:
                bg_color = "#FFFFFF"
                fg_color = "#000000"
                h1_color = "#000080"
                h2_color = "#006400"
                highlight_bg = "#FFFF00" # Yellow
                highlight_fg = "#000000"
            
            article_window.configure(bg=bg_color)
            
            # Content Area (Rich Text)
            text_area = self.scrolledtext.ScrolledText(article_window, wrap=self.tk.WORD, bg=bg_color, fg=fg_color, font=("Helvetica", 12))
            text_area.pack(expand=True, fill='both', padx=10, pady=10)

            # Configure Tags
            text_area.tag_config("h1", font=("Helvetica", 20, "bold"), foreground=h1_color, spacing3=10)
            text_area.tag_config("h2", font=("Helvetica", 16, "bold"), foreground=h2_color, spacing3=5)
            text_area.tag_config("h3", font=("Helvetica", 14, "bold"), spacing3=2)
            text_area.tag_config("bullet", lmargin1=20, lmargin2=30)
            text_area.tag_config("para", spacing2=2)
            text_area.tag_config("highlight", background=highlight_bg, foreground=highlight_fg)

            # Parse and render
            lines = content.split('\n')
            
            # Add Title
            text_area.insert('end', f"{entry.title}\n", "h1")
            
            for line in lines:
                line = line.rstrip()
                if not line:
                    text_area.insert('end', '\n')
                    continue
                    
                tag = None
                if line.startswith('# '):
                    tag = "h1"
                    line = line[2:].strip()
                elif line.startswith('## '):
                    tag = "h2"
                    line = line[3:].strip()
                elif line.startswith('### '):
                    tag = "h3"
                    line = line[4:].strip()
                elif line.startswith('• '):
                    tag = "bullet"
                
                # Insert text
                start_index = text_area.index("end-1c")
                text_area.insert("end", line + "\n")
                if tag:
                    text_area.tag_add(tag, start_index, "end-1c")
                else:
                    # Generic Paragraph
                    text_area.insert('end', line + '\n', "para")
            
            text_area.configure(state='disabled') # Read-only
            
            # Close button (floating or packed at bottom)
            # For simplicity, we just rely on window close, but can add one if needed.

            
            # Handle Escape key
            def on_escape(event):
                article_window.destroy()
            article_window.bind("<Escape>", on_escape)
            
        except Exception as e:
            self.messagebox.showerror("Error", f"Failed to open article: {e}")
    
    def on_click(self, event):
        """Handle regular click."""
        try:
            index = self.chat_display.index(f"@{event.x},{event.y}")
            tags = list(self.chat_display.tag_names(index))
            for tag in tags:
                if tag.startswith("concept") or tag.startswith("link"):
                    return
        except Exception:
            pass
    
    def on_ctrl_click(self, event):
        """Handle Ctrl+Click - select word."""
        try:
            index = self.chat_display.index(f"@{event.x},{event.y}")
            word_start = index + " wordstart"
            word_end = index + "wordend"
            word = self.chat_display.get(word_start, word_end).strip()
            if word and len(word) > 2:
                self.selected_text = word
                self.chat_display.tag_remove("selected", "1.0", self.tk.END)
                self.chat_display.tag_add("selected", word_start, word_end)
                select_bg = "#333333" if self.dark_mode else "lightblue"
                self.chat_display.tag_config("selected", background=select_bg)
                self.input_entry.delete(0, self.tk.END)
                self.input_entry.insert(0, f"Explain {word} in detail")
        except Exception:
            pass
    
    def on_highlight_enter(self, event):
        """Handle highlight + Enter."""
        try:
            if self.chat_display.tag_ranges("sel"):
                selected = self.chat_display.get("sel.first", "sel.last").strip()
                if selected and len(selected) > 0:
                    self.input_entry.delete(0, self.tk.END)
                    self.input_entry.insert(0, selected)
                    self.chat_display.tag_remove("sel", "1.0", self.tk.END)
                    self.input_entry.focus_set()
                    self.on_send()
                    return "break"
        except Exception:
            pass
        return None
    
    def on_clear(self):
        """Clear chat history."""
        self.history.clear()
        self.query_history.clear()
        clear_runtime_memory(reset_rag=False)
        self._close_command_mode()
        self.hide_autocomplete()
        self.chat_display.delete("1.0", self.tk.END)
        self.update_status("History and memory cleared")

    def on_app_close(self):
        """Persist settings, clear runtime memory, and close the GUI."""
        try:
            save_theme_name(self.theme_name)
        except Exception as e:
            print(f"[GUI] Failed to save theme on close: {e}")
        try:
            self.history.clear()
            self.query_history.clear()
            clear_runtime_memory(reset_rag=True)
            from chatbot.model_manager import ModelManager
            ModelManager.close_all()
        except Exception as e:
            print(f"[GUI] Failed to clear runtime memory on close: {e}")
        self.root.destroy()
    
    def show_help(self):
        """Show help dialog."""
        help_content = """Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /help              Show this help menu
  /exit, :q, quit    Quit the application
  /clear             Clear chat history
  /themes            Choose a theme
  /model             Select different model
  /status            Show system status
  /api               Configure external API mode
  /forge             Build a ZIM from local docs

Current Mode: Chat Mode

Mode Description:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAT MODE (Default): Full AI responses using RAG with detailed explanations and synthesis.

Mouse Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Highlight text and press Enter to auto-paste and query
  • Ctrl+Click on a word to select and query it
  • Click on article links to open full content

Keyboard Shortcuts:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Enter in input field: Send message
  • Enter with text selected in chat: Auto-paste and send
  • Ctrl+Click: Select word for query
  • ↑↓ Arrow keys: Navigate autocomplete suggestions
  • Tab: Select autocomplete suggestion
  • Esc: Close dialogs
"""

        if getattr(self, "use_terminal_dialogs", False):
            self.append_message("system", help_content)
            return

        help_window = self.tk.Toplevel(self.root)
        help_window.title("Help & Settings")
        help_window.geometry("800x600")
        help_window.transient(self.root)
        help_window.grab_set()
        
        if self.dark_mode:
            bg_color = "#000000"
            fg_color = "#E0E0E0"
            select_bg = "#333333"
            select_fg = "#FFFFFF"
            button_bg = "#2a2a2a"
            button_fg = "#E0E0E0"
        else:
            bg_color = "#FFFFFF"
            fg_color = "#000000"
            select_bg = "lightblue"
            select_fg = "#000000"
            button_bg = "#F0F0F0"
            button_fg = "#000000"
        
        help_window.configure(bg=bg_color)
        
        title_label = self.tk.Label(
            help_window,
            text="Chatbot - Help & Settings",
            font=("Arial", 16, "bold"),
            bg=bg_color, fg=fg_color
        )
        title_label.pack(pady=10)
        
        content_text = self.scrolledtext.ScrolledText(
            help_window,
            wrap=self.tk.WORD, padx=10, pady=10,
            font=("Arial", 10),
            bg=bg_color, fg=fg_color,
            state=self.tk.DISABLED
        )
        content_text.pack(fill=self.tk.BOTH, expand=True, padx=20, pady=10)
        
        content_text.config(state=self.tk.NORMAL)
        content_text.insert("1.0", help_content)
        content_text.config(state=self.tk.DISABLED)
        
        button_frame = self.tk.Frame(help_window, bg=bg_color)
        button_frame.pack(pady=10)
        
        close_btn = self.tk.Button(
            button_frame,
            text="Close",
            command=help_window.destroy,
            bg=button_bg, fg=button_fg,
            activebackground=select_bg,
            activeforeground=button_fg,
            font=("Arial", 10), width=15
        )
        close_btn.pack()
        
        def on_key(event):
            if event.keysym in ["Return", "Escape"]:
                help_window.destroy()
                return "break"
        
        help_window.bind("<KeyPress>", on_key)
    
    def on_send(self, event=None):
        """Send message."""
        user_input = self.input_entry.get().strip()
        if not user_input:
            if self.command_mode:
                self._handle_command_input("")
            return
        
        if not user_input.startswith("/") and user_input not in {"help", "quit", "exit"}:
            if user_input not in self.query_history:
                self.query_history.append(user_input)
                if len(self.query_history) > 50:
                    self.query_history = self.query_history[-50:]
        
        self.input_entry.delete(0, self.tk.END)
        self.hide_autocomplete()
        
        self.append_message("user", user_input)
        
        # Handle commands
        if user_input.lower() in {"/help", "help"}:
            self.show_help()
            return
        if user_input.lower() in {"/exit", ":q", "quit", "exit"}:
            self.on_app_close()
            return
        if self.command_mode:
            self._handle_command_input(user_input)
            return
        if user_input.lower() == "/clear":
            self.on_clear()
            return
        if user_input.lower() == "/dark":
            self.append_message("system", "The /dark command was removed. Use /themes.")
            return
        if user_input.lower().startswith("/themes"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                self._enter_themes_mode()
            else:
                theme_name = parts[1].strip()
                if not self.set_theme(theme_name):
                    self.append_message("system", f"Unknown theme '{theme_name}'. Use /themes to pick.")
            return
        if user_input.lower() == "/model":
            self._enter_model_mode()
            return
        if user_input.lower() in {"/response", "/responce", "/links"}:
            self.append_message("system", "This command was removed in public. Hermit now uses chat response mode only.")
            return
        if user_input.lower() == "/status":
            self.show_status_dialog()
            return
        if user_input.lower() in ["/api", "/connect"]:
            self._enter_api_mode()
            return
        if user_input.lower().startswith("/forge"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                self._enter_forge_mode(parts[1].strip())
            else:
                self._enter_forge_mode()
            return
        
        # Add to history and get response
        self.history.append(Message(role="user", content=user_input))
        
        # Show loading state
        self.show_loading("Processing Request")
        
        # Get response in background
        threading.Thread(target=self.get_response, args=(user_input,), daemon=True).start()
    
    def get_response(self, query: str):
        """Get response based on current mode."""
        try:
            # Set up status callback for real-time updates during RAG processing
            def status_callback(status):
                # Use status bar AND loading bubble
                self.root.after(0, lambda s=status: self.update_status(s))
                self.update_loading_text(status)
            set_status_callback(status_callback)
            
            if self.link_mode:
                # Link mode: Show clickable links
                links = retrieve_and_display_links(query)
                self.hide_loading()
                self.root.after(0, lambda: self.append_links(query, links))
            else:
                # Response mode: Full AI response
                messages = build_messages(self.system_prompt, self.history)
                
                # Update to show we're about to generate
                self.root.after(0, lambda: self.update_status("Generating response..."))
                
                # Use transition for seamless look
                insert_mark = self.transition_loading_to_response()
                
                ai_tag_name = f"ai_message_{id(self)}"
                
                if self.streaming_enabled:
                    accumulated: List[str] = []
                    for chunk in stream_chat(self.model, messages):
                        follow_now = self._is_near_bottom()
                        accumulated.append(chunk)
                        self.chat_display.insert(insert_mark, chunk, ai_tag_name)
                        if follow_now:
                            self.chat_display.see(self.tk.END)
                        self.root.update_idletasks()
                    
                    assistant_reply = "".join(accumulated)
                else:
                    auto_follow = self._is_near_bottom()
                    assistant_reply = full_chat(self.model, messages)
                    self.chat_display.insert(insert_mark, assistant_reply, ai_tag_name)
                    if auto_follow:
                        self.chat_display.see(self.tk.END)
                
                # Padding is already present from the bubble, but let's ensure it's correct
                # We reused the bubble structure which ends with padding + \n\n
                # So we don't need to add it again unless we were in fallback mode.
                
                if self._is_near_bottom():
                    self.chat_display.see(self.tk.END)
                
                if assistant_reply:
                    self.history.append(Message(role="assistant", content=assistant_reply))
            
            self.update_status("Ready")
        
        except RuntimeError as err:
            self.hide_loading()
            self.update_status(f"Error: {err}")
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            self.append_message("system", f"[error] {err}")
        except Exception as e:
            self.hide_loading()
            self.messagebox.showerror("Error", f"Failed to get response: {e}")
    
    def show_api_config_dialog(self):
        """Show dialog to configure external API."""
        dialog = self.tk.Toplevel(self.root)
        dialog.title("External API Configuration")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        if self.dark_mode:
            dialog.configure(bg="#2A2A2A")
            style_prefix = ""
        else:
            dialog.configure(bg="#FFFFFF")
            style_prefix = ""
            
        # Variables (referencing config directly for simplicity in this session, 
        # ideally should use vars and save back)
        api_mode_var = self.tk.BooleanVar(value=config.API_MODE)
        url_var = self.tk.StringVar(value=config.API_BASE_URL)
        key_var = self.tk.StringVar(value=config.API_KEY)
        model_var = self.tk.StringVar(value=config.API_MODEL_NAME)
        
        main_frame = self.ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=self.tk.BOTH, expand=True)
        
        # Enable Toggle
        self.ttk.Checkbutton(
            main_frame, text="Enable External API Mode (LM Studio / Ollama)", 
            variable=api_mode_var
        ).pack(anchor=self.tk.W, pady=(0, 20))
        
        # Grid layout for inputs
        input_frame = self.ttk.Frame(main_frame)
        input_frame.pack(fill=self.tk.X, pady=10)
        
        # Custom Entry Style Helper
        def create_entry(parent, var):
            entry = self.tk.Entry(
                parent, textvariable=var,
                font=("Arial", 11),
                bg="#1E1E1E" if self.dark_mode else "#FFFFFF",
                fg="#FFFFFF" if self.dark_mode else "#000000",
                insertbackground="#FFFFFF" if self.dark_mode else "#000000", # Cursor color
                relief=self.tk.FLAT, borderwidth=1,
                highlightthickness=1,
                highlightbackground="#444444" if self.dark_mode else "#CCCCCC",
                highlightcolor="#808080" if self.dark_mode else "#666666"
            )
            return entry

        self.ttk.Label(input_frame, text="Base URL:").grid(row=0, column=0, sticky=self.tk.W, pady=5)
        # self.ttk.Entry(input_frame, textvariable=url_var, width=40).grid(row=0, column=1, padx=10, pady=5)
        create_entry(input_frame, url_var).grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        self.ttk.Label(input_frame, text="(e.g. http://localhost:1234/v1)").grid(row=1, column=1, sticky=self.tk.W, padx=10)
        
        self.ttk.Label(input_frame, text="API Key:").grid(row=2, column=0, sticky=self.tk.W, pady=5)
        # self.ttk.Entry(input_frame, textvariable=key_var, width=40).grid(row=2, column=1, padx=10, pady=5)
        create_entry(input_frame, key_var).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        self.ttk.Label(input_frame, text="Model Name:").grid(row=3, column=0, sticky=self.tk.W, pady=5)
        # self.ttk.Entry(input_frame, textvariable=model_var, width=40).grid(row=3, column=1, padx=10, pady=5)
        create_entry(input_frame, model_var).grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        # Test Connection Button
        status_label = self.ttk.Label(main_frame, text="")
        status_label.pack(pady=5)
        
        def test_connection():
            status_label.config(text="Connecting...", foreground="orange")
            dialog.update()
            try:
                from chatbot.api_client import OpenAIClientWrapper
                client = OpenAIClientWrapper(url_var.get(), key_var.get(), model_var.get())
                # Quick test
                resp = client.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5
                )
                if resp:
                    status_label.config(text="Connection Successful!", foreground="green")
                else:
                    status_label.config(text="Empty Response", foreground="red")
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}", foreground="red")
        
        self.ttk.Button(main_frame, text="Test Connection", command=test_connection).pack(pady=10)
        
        # Save/Cancel
        btn_frame = self.ttk.Frame(main_frame)
        btn_frame.pack(fill=self.tk.X, pady=20, side=self.tk.BOTTOM)
        
        def save():
            config.API_MODE = api_mode_var.get()
            config.API_BASE_URL = url_var.get()
            config.API_KEY = key_var.get()
            config.API_MODEL_NAME = model_var.get()
            
            # Reset ModelManager to force reload next time
            from chatbot.model_manager import ModelManager
            ModelManager.close_all()
            
            mode_str = "External API" if config.API_MODE else "Local Internal"
            self.append_message("system", f"Configuration saved. Switched to {mode_str} Mode.")
            if config.API_MODE:
                self.model = config.API_MODEL_NAME
                self.root.title(f"Chatbot - API: {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
            
            dialog.destroy()
            
        self.ttk.Button(btn_frame, text="Save & Apply", command=save, style="Accent.TButton").pack(side=self.tk.RIGHT)
        self.ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=self.tk.RIGHT, padx=10)

    def show_status_dialog(self):
        """Show system status summary."""
        # AI Backend Status
        if config.API_MODE:
             backend_type = "External API"
             backend_detail = f"URL: {config.API_BASE_URL}\nKey: {'*' * len(config.API_KEY) if config.API_KEY else 'None'}"
        else:
             backend_type = "Local (GGUF)"
             backend_detail = "Engine: llama-cpp-python"

        model_status = f"Model: {self.model}"
        
        # RAG Status
        # We need to peek into chat module to get global rag
        rag_status = "Inactive"
        rag_detail = "No index loaded."
        
        from chatbot.chat import get_rag_system
        rag = get_rag_system()
        
        if rag:
            rag_status = "Active"
            count_docs = len(rag.indexed_paths) if rag.indexed_paths else 0
            count_chunks = len(rag.doc_chunks) if rag.doc_chunks else 0
            rag_detail = f"JIT Index: {count_docs} articles ({count_chunks} chunks)\n" \
                         f"Encoder: {rag.model_name}"
            if rag.faiss_index:
                 rag_detail += f"\nVectors: {rag.faiss_index.ntotal}"

        msg = (
            f"=== SYSTEM STATUS ===\n\n"
            f"AI BACKEND: {backend_type}\n"
            f"{backend_detail}\n"
            f"{model_status}\n\n"
            f"KNOWLEDGE BASE (RAG): {rag_status}\n"
            f"{rag_detail}\n\n"
            f"GUI Mode: {'Link Search' if self.link_mode else 'Chat Response'}\n"
            f"Theme: {self.theme_name} ({'Dark' if self.dark_mode else 'Light'})"
        )
        
        self.messagebox.showinfo("System Status", msg)

    def show_forge_dialog(self):
        """Show the unified Forge ZIM creator dialog."""
        import sys
        import os
        
        try:
            # Ensure project root is in path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Import the unified ForgeGUI
            from forge import ForgeGUI
            
            # Launch as a child window
            forge_app = ForgeGUI(parent=self.root)
            # ForgeGUI handles its own event loop or we can just let it run
            # Since it's a Toplevel, it will stay until closed.
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Forge Error", f"Could not launch Forge:\n{e}")

    def run(self):
        """Start the GUI."""
        self.root.mainloop()
