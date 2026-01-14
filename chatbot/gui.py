
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
        # ONLY show dialog for actual downloading (network).
        # "loading" (disk->RAM) and "checking" happen too often and are annoying.
        if status == "downloading":
            # Show dialog if not already showing
            if not self.download_dialog:
                self.download_dialog = DownloadProgressDialog(self.root, self.dark_mode)
                self.download_dialog.show("Downloading Model...")
            self.download_dialog.update(status, progress, detail)
        
        elif status == "loading" or status == "checking":
             # Just update status bar if possible (no-op here since update_status is disabled)
             # But definitely DO NOT show the popup dialog.
             pass

            
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
                
                self.chat_display.see(self.tk.END)
                
        self.root.after(0, _update)

    
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
        
        # Target the bubble text tag instead of input_entry
        self.chat_display.tag_config("loading_content_text", foreground=pulse_color)
        
        # Schedule next frame (60ms for smooth animation)
        self.loading_animation_id = self.root.after(60, self._animate_loading_pulse)
    
    def get_installed_models(self) -> List[Tuple[str, ModelPlatform]]:
        """Get list of supported local models from config and shared_models directory."""
        models: List[Tuple[str, ModelPlatform]] = []
        
        # 1. Add models explicitly defined in Config
        if hasattr(config, 'MODEL_ALETHEIA_3B'):
            models.append((config.MODEL_ALETHEIA_3B, ModelPlatform.LOCAL))
            
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
            
            # Global option database for Light mode
            self.root.option_add("*Background", bg_color)
            self.root.option_add("*Foreground", fg_color)
            self.root.option_add("*Entry.Background", input_bg)
            self.root.option_add("*Entry.Foreground", input_fg)
            self.root.option_add("*Listbox.Background", input_bg)
            self.root.option_add("*Listbox.Foreground", input_fg)

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
        
        commands = ["/help", "/exit", "/clear", "/dark", "/model", "/status", "/response", "/links", "/api", "/forge"]
        
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

RESPONSE MODE (Default): Full AI responses using RAG with detailed explanations and synthesis.
   Use /response command to switch to this mode.

LINK MODE: Fast semantic search returns clickable links to relevant articles.
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
""".format(current_mode="Link Mode" if self.link_mode else "Response Mode")
        
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
        if user_input.lower() == "/status":
            self.show_status_dialog()
            return
        if user_input.lower() in ["/api", "/connect"]:
            self.show_api_config_dialog()
            return
        if user_input.lower() == "/forge":
            self.show_forge_dialog()
            return
        
        # Add to history and get response
        self.history.append(Message(role="user", content=user_input))
        
        # Show loading state
        self.show_loading("Thinking")
        
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
                        accumulated.append(chunk)
                        self.chat_display.insert(insert_mark, chunk, ai_tag_name)
                        self.chat_display.see(self.tk.END)
                        self.root.update_idletasks()
                    
                    assistant_reply = "".join(accumulated)
                else:
                    assistant_reply = full_chat(self.model, messages)
                    self.chat_display.insert(insert_mark, assistant_reply, ai_tag_name)
                
                # Padding is already present from the bubble, but let's ensure it's correct
                # We reused the bubble structure which ends with padding + \n\n
                # So we don't need to add it again unless we were in fallback mode.
                
                self.chat_display.see(self.tk.END)
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
            f"Theme: {'Dark' if self.dark_mode else 'Light'}"
        )
        
        self.messagebox.showinfo("System Status", msg)

    def show_forge_dialog(self):
        """Show the Forge ZIM creator dialog with drag-and-drop support."""
        import tkinter as tk
        from tkinter import scrolledtext, filedialog, messagebox
        import sys
        import os
        
        # Import forge module components
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from forge import DocumentParser, ZIMCreator, LIBZIM_AVAILABLE
        except ImportError as e:
            self.messagebox.showerror("Forge Unavailable", f"Could not load Forge:\n{e}")
            return
        
        if not LIBZIM_AVAILABLE:
            self.messagebox.showerror("Missing Dependency", "libzim not installed.\npip install libzim")
            return
        
        # Try TkinterDnD - create completely standalone window
        dnd_available = False
        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES
            root = TkinterDnD.Tk()
            dnd_available = True
        except ImportError:
            root = tk.Tk()
        
        root.title("Hermit Forge")
        root.geometry("550x450")
        
        # Position near main app
        try:
            root.geometry(f"+{self.root.winfo_x() + 50}+{self.root.winfo_y() + 50}")
        except:
            pass
        
        # Dark theme
        bg = "#1E1E1E"
        fg = "#E0E0E0"
        accent = "#5865F2"
        input_bg = "#2A2A2A"
        btn_bg = "#333333"
        
        root.configure(bg=bg)
        
        # State
        files_list = []
        extensions = {'.txt', '.md', '.pdf', '.docx', '.html', '.htm', '.epub'}
        
        # --- UI ---
        
        # Title
        tk.Label(root, text="Hermit Forge", font=("Arial", 18, "bold"), bg=bg, fg=fg).pack(pady=(15, 5))
        
        # Drop zone frame
        drop_frame = tk.Frame(root, bg="#252525", highlightbackground="#444", highlightthickness=2)
        drop_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Drop zone content (centered)
        if dnd_available:
            tk.Label(drop_frame, text="[+]", font=("Arial", 40), bg="#252525", fg=fg).pack(pady=(30, 5))
            status_label = tk.Label(drop_frame, text="Drop files here or use buttons below", font=("Arial", 14), bg="#252525", fg=fg)
        else:
            tk.Label(drop_frame, text="[+]", font=("Arial", 40), bg="#252525", fg=fg).pack(pady=(30, 5))
            status_label = tk.Label(drop_frame, text="Use buttons below to add files", font=("Arial", 14), bg="#252525", fg=fg)
        status_label.pack()
        count_label = tk.Label(drop_frame, text="", font=("Arial", 11, "bold"), bg="#252525", fg=accent)
        count_label.pack(pady=(10, 20))
        
        def update_count():
            if files_list:
                count_label.config(text=f"{len(files_list)} files ready")
                status_label.config(text="Drop more or click Accept")
            else:
                count_label.config(text="")
                status_label.config(text="Drop files here")
        
        def add_path(path):
            from pathlib import Path
            p = Path(path)
            added = 0
            if p.is_file() and p.suffix.lower() in extensions:
                if str(p) not in files_list:
                    files_list.append(str(p))
                    added = 1
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in extensions:
                        if str(f) not in files_list:
                            files_list.append(str(f))
                            added += 1
            return added
        
        def on_drop(event):
            data = event.data
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, f"DEBUG: Drop event received\n")
            log_text.insert(tk.END, f"DEBUG: Raw data = {repr(data)}\n")
            log_text.config(state=tk.DISABLED)
            
            # Parse paths (handle braces for paths with spaces)
            if '{' in data:
                import re
                paths = re.findall(r'\{([^}]+)\}', data)
                rest = re.sub(r'\{[^}]+\}', '', data).strip()
                if rest:
                    paths.extend(rest.split())
            else:
                paths = data.split()
            
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, f"DEBUG: Parsed {len(paths)} paths: {paths}\n")
            log_text.config(state=tk.DISABLED)
            
            total = 0
            for p in paths:
                added = add_path(p)
                log_text.config(state=tk.NORMAL)
                log_text.insert(tk.END, f"DEBUG: Path '{p}' -> added {added} files\n")
                log_text.config(state=tk.DISABLED)
                total += added
            
            update_count()
            if total > 0:
                log_text.config(state=tk.NORMAL)
                log_text.insert(tk.END, f"SUCCESS: Dropped {total} files\n")
                log_text.see(tk.END)
                log_text.config(state=tk.DISABLED)
            else:
                log_text.config(state=tk.NORMAL)
                log_text.insert(tk.END, f"WARNING: No valid files found in drop\n")
                log_text.see(tk.END)
                log_text.config(state=tk.DISABLED)
        
        # Register drag-drop on the root window
        if dnd_available:
            try:
                root.drop_target_register(DND_FILES)
                root.dnd_bind('<<Drop>>', on_drop)
                log_text.config(state=tk.NORMAL)
                log_text.insert(tk.END, "DEBUG: Drag-drop registered successfully\n")
                log_text.config(state=tk.DISABLED)
            except Exception as e:
                dnd_available = False
                log_text.config(state=tk.NORMAL)
                log_text.insert(tk.END, f"DEBUG: Drag-drop registration failed: {e}\n")
                log_text.config(state=tk.DISABLED)
        
        # Output filename
        out_frame = tk.Frame(root, bg=bg)
        out_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(out_frame, text="Output:", bg=bg, fg=fg).pack(side=tk.LEFT)
        out_var = tk.StringVar(value="knowledge.zim")
        tk.Entry(out_frame, textvariable=out_var, bg=input_bg, fg=fg, insertbackground=fg, relief=tk.FLAT, width=25).pack(side=tk.LEFT, padx=5)
        
        # Button row
        btn_frame = tk.Frame(root, bg=bg)
        btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        def pick_files():
            sel = filedialog.askopenfilenames(filetypes=[("Supported", "*.txt *.md *.pdf *.docx *.html *.epub")])
            for f in sel:
                add_path(f)
            update_count()
        
        def pick_folder():
            folder = filedialog.askdirectory()
            if folder:
                add_path(folder)
                update_count()
        
        def clear_all():
            files_list.clear()
            update_count()
        
        tk.Button(btn_frame, text="Add Files", command=pick_files, bg=btn_bg, fg=fg, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Add Folder", command=pick_folder, bg=btn_bg, fg=fg, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Clear", command=clear_all, bg=btn_bg, fg=fg, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        # Log
        log_text = scrolledtext.ScrolledText(root, height=3, bg=input_bg, fg=fg, state=tk.DISABLED, font=("Consolas", 9))
        log_text.pack(fill=tk.X, padx=20, pady=5)
        
        if dnd_available:
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, "Drag-drop ready! Drop files anywhere.\n")
            log_text.config(state=tk.DISABLED)
        else:
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, "Use buttons to add files.\n")
            log_text.config(state=tk.DISABLED)
        
        # Action row
        action_frame = tk.Frame(root, bg=bg)
        action_frame.pack(fill=tk.X, padx=20, pady=(5, 15))
        
        def do_create():
            if not files_list:
                messagebox.showwarning("No Files", "Add files first!")
                return
            
            out_path = out_var.get()
            from pathlib import Path
            title = Path(out_path).stem.replace("_", " ").title()
            
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, f"Creating {out_path}...\n")
            log_text.config(state=tk.DISABLED)
            root.update()
            
            try:
                creator = ZIMCreator(out_path, title)
                for fp in files_list:
                    doc = DocumentParser.parse_file(fp)
                    if doc:
                        creator.add_document(doc)
                zim_path = creator.create()
                size = os.path.getsize(zim_path) / (1024 * 1024)
                
                messagebox.showinfo("Done!", f"Created {os.path.basename(zim_path)}\n{len(creator.documents)} docs, {size:.1f} MB")
                self.append_message("system", f"Forge: {os.path.basename(zim_path)} ({len(creator.documents)} docs)")
                root.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        tk.Button(action_frame, text="Cancel", command=root.destroy, bg=btn_bg, fg=fg, relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(action_frame, text="Accept & Create", command=do_create, bg=accent, fg="#FFF", font=("Arial", 11, "bold"), relief=tk.FLAT, padx=15, pady=5).pack(side=tk.RIGHT)
        
        root.bind("<Escape>", lambda e: root.destroy())
        root.bind("<Return>", lambda e: do_create())
        
        root.mainloop()

    def run(self):
        """Start the GUI."""
        self.root.mainloop()