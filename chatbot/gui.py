
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
from chatbot.chat import stream_chat, full_chat, build_messages, set_status_callback, retrieve_and_display_links
from chatbot import config
from chatbot.config import DEFAULT_MODEL
from chatbot.model_manager import set_download_callback


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
        
        # Detail label (e.g., "Aletheia-3B (2.1 GB)")
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
                "checking": "🔍 Checking model availability...",
                "downloading": "⬇️ Downloading model...",
                "loading": "🔄 Loading model into GPU...",
                "ready": "✅ Model ready!",
                "error": "❌ Error"
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
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        
        self.model = model or DEFAULT_MODEL
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.streaming_enabled = streaming_enabled
        
        self.history: List[Message] = []
        self.query_history: List[str] = []
        self.dark_mode = True
        
        # Dual mode: link_mode (default) vs response_mode
        self.link_mode = False  # Default to Response mode
        
        self.root = self.tk.Tk()
        self.root.title(f"Hermit - {self.model} ({'Link Mode' if self.link_mode else 'Response Mode'})")
        self.root.geometry("900x700")
        
        # Chat display
        chat_frame = self.ttk.Frame(self.root)
        chat_frame.pack(fill=self.tk.BOTH, expand=True, padx=15, pady=(15, 5))
        
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
        
        # Input area
        input_container = self.ttk.Frame(self.root)
        input_container.pack(fill=self.tk.X, pady=(5, 10))
        
        input_frame = self.ttk.Frame(input_container)
        input_frame.pack(anchor=self.tk.CENTER)
        
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
        self.input_entry.bind("<Escape>", self.on_autocomplete_close)
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
        
        # Download progress dialog
        self.download_dialog: Optional[DownloadProgressDialog] = None
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
        """Handle download progress updates (called on main thread)."""
        if status in ("checking", "downloading", "loading"):
            # Show dialog if not already showing
            if not self.download_dialog:
                self.download_dialog = DownloadProgressDialog(self.root, self.dark_mode)
                self.download_dialog.show("Preparing Model...")
            self.download_dialog.update(status, progress, detail)
            
        elif status == "ready":
            # Hide dialog after a brief delay to show completion
            if self.download_dialog:
                self.download_dialog.update(status, 1.0, detail)
                self.root.after(500, self._hide_download_dialog)
                
        elif status == "error":
            # Show error briefly then hide
            if self.download_dialog:
                self.download_dialog.update(status, -1, detail)
                self.root.after(2000, self._hide_download_dialog)
    
    def _hide_download_dialog(self):
        """Hide the download progress dialog."""
        if self.download_dialog:
            self.download_dialog.hide()
            self.download_dialog = None
    
    def show_loading(self, text: str = "Thinking"):
        """Show loading state in input entry with pulsating text."""
        self.is_loading = True
        self.loading_text = text
        self.loading_pulse_step = 0
        self.loading_pulse_direction = 1
        # Store original colors to restore later
        self._loading_original_fg = self.input_entry.cget('fg')
        self._update_loading_display()
        self._animate_loading_pulse()
    
    def hide_loading(self):
        """Hide loading state and restore input entry."""
        self.is_loading = False
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        # Restore original foreground color
        if hasattr(self, '_loading_original_fg'):
            self.input_entry.config(fg=self._loading_original_fg)
        self.input_entry.delete(0, self.tk.END)
        self.input_entry.focus_set()
    
    def update_loading_text(self, text: str):
        """Update the loading text while keeping animation."""
        if self.is_loading:
            self.loading_text = text
            self._update_loading_display()
    
    def _update_loading_display(self):
        """Update the loading display with current text."""
        if not self.is_loading:
            return
        display_text = f"{self.loading_text}..."
        self.input_entry.delete(0, self.tk.END)
        self.input_entry.insert(0, display_text)
    
    def _get_pulse_color(self) -> str:
        """Get the current pulse color based on step."""
        # Pulse between dim gray and bright text color
        if self.dark_mode:
            # Dark mode: pulse between #555555 (dim) and #FFFFFF (bright)
            base_val = 85  # 0x55
            range_val = 170  # 0xFF - 0x55
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
        self.input_entry.config(fg=pulse_color)
        
        # Schedule next frame (60ms for smooth animation)
        self.loading_animation_id = self.root.after(60, self._animate_loading_pulse)
    
    def get_installed_models(self) -> List[Tuple[str, ModelPlatform]]:
        """Get list of supported local models from config."""
        models: List[Tuple[str, ModelPlatform]] = []
        
        # Add models from config

        if hasattr(config, 'MODEL_ALETHEIA_3B'):
            models.append((config.MODEL_ALETHEIA_3B, ModelPlatform.LOCAL))
            
        return models
    
    def show_model_menu(self):
        """Show model selection menu."""
        models = self.get_installed_models()
        if not models:
            self.messagebox.showwarning(
                "No Models",
                "No models found in config."
            )
            return
        
        model_window = self.tk.Toplevel(self.root)
        model_window.title("Select Model")
        model_window.geometry("500x600")
        model_window.transient(self.root)
        model_window.grab_set()
        
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
        
        model_window.configure(bg=bg_color)
        
        title_label = self.ttk.Label(model_window, text="Select Model", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        current_label = self.ttk.Label(model_window, text=f"Current: {self.model}", font=("Arial", 10))
        current_label.pack(pady=5)
        
        listbox_frame = self.ttk.Frame(model_window)
        listbox_frame.pack(fill=self.tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = self.ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        
        model_listbox = self.tk.Listbox(
            listbox_frame,
            font=("Arial", 11),
            bg=bg_color, fg=fg_color,
            selectbackground=select_bg, selectforeground=select_fg,
            yscrollcommand=scrollbar.set,
            activestyle="none", borderwidth=0, highlightthickness=1,
            highlightbackground=button_bg
        )
        model_listbox.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.config(command=model_listbox.yview)
        
        selected_index = 0
        model_list: List[str] = []
        
        for model_name, platform in models:
            # Create friendly display name
            display_name = model_name.split('/')[-1] if '/' in model_name else model_name
            if "Aletheia" in display_name:
                display_name = "Aletheia 3B (Fast)"
                
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
                model_window.destroy()
        
        def cancel():
            model_window.destroy()
        
        select_btn = self.ttk.Button(button_frame, text="Select", command=select_model, style="Accent.TButton")
        select_btn.pack(side=self.tk.LEFT, padx=5)
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
    
    def apply_theme(self):
        """Apply dark/light theme."""
        style = self.ttk.Style()
        style.theme_use('clam')
        
        if self.dark_mode:
            bg_color = "#2A2A2A"
            fg_color = "#E0E0E0"
            input_bg = "#1E1E1E"
            input_fg = "#FFFFFF"
            accent_color = "#808080"
            button_bg = "#333333"
            button_fg = "#FFFFFF"
            border_color = "#444444"
            concept_color = "#81D4FA"
        else:
            bg_color = "#FFFFFF"
            fg_color = "#000000"
            input_bg = "#F5F5F5"
            input_fg = "#000000"
            accent_color = "#666666"
            button_bg = "#E0E0E0"
            button_fg = "#000000"
            border_color = "#CCCCCC"
            concept_color = "#0277BD"
        
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
    
    def _configure_message_tag(self, tag_name: str, role: str):
        """Configure styling for message border tags."""
        if self.dark_mode:
            border_bg = "#1E1E1E"
        else:
            border_bg = "#E0E0E0"
        
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
        
        commands = ["/help", "/exit", "/clear", "/dark", "/model", "/response", "/links"]
        
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
        
        prefix = "You: " if role == "user" else "AI: " if role == "ai" else "System: "
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
        
        prefix = f"🔍 Search Results for '{query}':\n"
        self.chat_display.insert(self.tk.END, prefix)
        
        for i, link in enumerate(links, 1):
            title = link.get('title', 'Unknown Title')
            score = link.get('score', 0.0)
            snippet = link.get('snippet', '')
            path = link.get('path', '')
            
            # Create clickable link
            link_text = f"\n{i}. 📄 {title} (Score: {score:.3f})\n"
            if snippet:
                link_text += f"   {snippet[:150]}{'...' if len(snippet) > 150 else ''}\n"
            
            start_pos = self.chat_display.index(self.tk.END + "-1c")
            self.chat_display.insert(self.tk.END, link_text)
            end_pos = self.chat_display.index(self.tk.END + "-1c")
            
            # Make the title clickable
            link_tag = f"link_{i}_{id(self)}"
            self.chat_display.tag_add(link_tag, start_pos, end_pos)
            self.chat_display.tag_config(link_tag, foreground="#5DB9FF" if self.dark_mode else "blue", underline=True)
            
            # Store the path for this link
            highlight_terms = link.get('search_context', {}).get('entities', [])
            self.chat_display.tag_bind(link_tag, "<Button-1>", lambda e, p=path, ht=highlight_terms: self.open_zim_article(p, highlight_terms=ht))
            self.chat_display.tag_bind(link_tag, "<Enter>", lambda e: self.chat_display.config(cursor="hand2"))
            self.chat_display.tag_bind(link_tag, "<Leave>", lambda e: self.chat_display.config(cursor=""))
        
        message_end = self.chat_display.index(self.tk.END + "-1c")
        
        tag_name = f"links_message_{id(self)}"
        self.chat_display.tag_add(tag_name, message_start, message_end)
        self._configure_message_tag(tag_name, "system")
        
        self.chat_display.insert(self.tk.END, "\n\n")
        self.chat_display.see(self.tk.END)
    
    def open_zim_article(self, path, highlight_terms=None):
        """Open a ZIM article in a new window."""
        print(f"[GUI] Opening ZIM path: '{path}'")
        try:
            try:
                import libzim
            except ImportError:
                self.messagebox.showerror("Error", "libzim not installed")
                return

            from chatbot.rag import TextProcessor
            
            # Find the ZIM file
            import os
            zim_files = [f for f in os.listdir('.') if f.endswith('.zim')]
            if not zim_files:
                self.messagebox.showerror("Error", "No ZIM files found")
                return
            
            zim_file = zim_files[0]
            print(f"[GUI] Using ZIM file: {zim_file}")
            zim = libzim.Archive(zim_file)
            
            entry = None
            
            # Helper to try finding an entry
            def try_find(p):
                try:
                    return zim.get_entry_by_path(p)
                except:
                    return None

            # Strategy 1: Direct path
            entry = try_find(path)
            
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
                        print(f"[GUI] Trying variation: '{attempt}'")
                        entry = try_find(attempt)
                        if entry: 
                            print(f"[GUI] Found match: '{attempt}'")
                            break
                    if entry: break

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
        self.chat_display.delete("1.0", self.tk.END)
        self.update_status("History cleared")
    
    def show_help(self):
        """Show help dialog."""
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
        
        help_content = """Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /help              Show this help menu
  /exit, :q, quit    Quit the application
  /clear             Clear chat history
  /dark              Toggle dark mode
  /model             Select different model
  /response          Switch to Response Mode (Full AI)
  /links             Switch to Link Mode (Search Results)

Current Mode: {current_mode}

Mode Description:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 RESPONSE MODE (Default): Full AI responses using RAG with detailed explanations and synthesis.
   Use /response command to switch to this mode.

🔗 LINK MODE: Fast semantic search returns clickable links to relevant articles.
   Like Reddit AI - quickly find sources without waiting for AI generation.
   Use /links command to switch to this mode.

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
""".format(current_mode="🔗 Link Mode" if self.link_mode else "🤖 Response Mode")
        
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
            self.root.quit()
            return
        if user_input.lower() == "/clear":
            self.on_clear()
            return
        if user_input.lower() == "/dark":
            self.dark_mode = not self.dark_mode
            self.apply_theme()
            mode_text = "Dark mode enabled" if self.dark_mode else "Light mode enabled"
            self.update_status(mode_text)
            return
        if user_input.lower() == "/model":
            self.show_model_menu()
            return
        if user_input.lower() == "/response":
            self.link_mode = False
            self.root.title(f"Chatbot - {self.model} (Response Mode)")
            self.append_message("system", "Switched to Response Mode")
            return
        if user_input.lower() == "/links":
            self.link_mode = True
            self.root.title(f"Chatbot - {self.model} (Link Mode)")
            self.append_message("system", "Switched to Link Mode")
            return
        
        # Add to history and get response
        self.history.append(Message(role="user", content=user_input))
        
        # Show loading state
        self.show_loading("Searching")
        
        # Get response in background
        threading.Thread(target=self.get_response, args=(user_input,), daemon=True).start()
    
    def get_response(self, query: str):
        """Get response based on current mode."""
        try:
            # Set up status callback for real-time updates during RAG processing
            def status_callback(status):
                self.root.after(0, lambda s=status: self.update_loading_text(s))
            set_status_callback(status_callback)
            
            if self.link_mode:
                # Link mode: Show clickable links
                links = retrieve_and_display_links(query)
                self.root.after(0, self.hide_loading)
                self.root.after(0, lambda: self.append_links(query, links))
            else:
                # Response mode: Full AI response
                messages = build_messages(self.system_prompt, self.history)
                
                # Update to show we're about to generate
                self.root.after(0, lambda: self.update_loading_text("Generating response"))
                
                self.chat_display.insert(self.tk.END, "\n")
                
                ai_tag_name = f"ai_message_{id(self)}"
                self._configure_message_tag(ai_tag_name, "ai")
                
                self.chat_display.insert(self.tk.END, "AI: ", ai_tag_name)
                start_pos = self.chat_display.index(self.tk.END + "-1c")
                
                # Hide loading before streaming starts
                self.root.after(0, self.hide_loading)
                
                if self.streaming_enabled:
                    accumulated: List[str] = []
                    for chunk in stream_chat(self.model, messages):
                        accumulated.append(chunk)
                        self.chat_display.insert(self.tk.END, chunk, ai_tag_name)
                        self.chat_display.see(self.tk.END)
                        self.root.update_idletasks()
                    
                    assistant_reply = "".join(accumulated)
                else:
                    assistant_reply = full_chat(self.model, messages)
                    self.chat_display.insert(self.tk.END, assistant_reply, ai_tag_name)
                
                # Add padding and final newline
                padding = "    "
                self.chat_display.insert(self.tk.END, padding)
                ai_message_end_with_padding = self.chat_display.index(self.tk.END + "-1c")
                
                self.chat_display.tag_add(ai_tag_name, start_pos, ai_message_end_with_padding)
                
                self.chat_display.insert(self.tk.END, "\n\n")
                self.chat_display.see(self.tk.END)
                
                if assistant_reply:
                    self.history.append(Message(role="assistant", content=assistant_reply))
            
            self.update_status("Ready")
        
        except RuntimeError as err:
            self.root.after(0, self.hide_loading)
            self.update_status(f"Error: {err}")
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            self.append_message("system", f"[error] {err}")
        except Exception as e:
            self.root.after(0, self.hide_loading)
            self.messagebox.showerror("Error", f"Failed to get response: {e}")
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()