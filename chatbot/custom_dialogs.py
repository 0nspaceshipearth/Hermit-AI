import tkinter as tk
from tkinter import ttk

class CustomMessageDialog:
    """A theme-aware replacement for messagebox."""
    
    def __init__(self, parent: tk.Tk, title: str, message: str, type_: str = "info", dark_mode: bool = True):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Determine colors based on theme
        if dark_mode:
            bg_color = "#2A2A2A"
            fg_color = "#E0E0E0"
            btn_bg = "#444444"
            btn_fg = "#FFFFFF"
        else:
            bg_color = "#FFFFFF"
            fg_color = "#000000"
            btn_bg = "#E0E0E0"
            btn_fg = "#000000"
            
        self.dialog.configure(bg=bg_color)
        
        # Calculate size based on message length (rough)
        lines = message.split('\n')
        width = min(600, max(300, max(len(l) for l in lines) * 8 + 40))
        height = min(500, max(150, len(lines) * 20 + 100))
        
        # Center dialog
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        
        x = parent_x + (parent_w - width) // 2
        y = parent_y + (parent_h - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Main content frame
        frame = tk.Frame(self.dialog, bg=bg_color, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Icon (Text-based for minimal deps)
        icon_map = {
            "info": "",
            "warning": "",
            "error": "",
            "question": ""
        }
        icon_char = icon_map.get(type_, "")
        
        tk.Label(frame, text=icon_char, font=("Arial", 32), bg=bg_color, fg=fg_color).pack(pady=(0, 10))
        
        # Scrollable Message (if long)
        text_frame = tk.Frame(frame, bg=bg_color)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        msg_label = tk.Text(
            text_frame, wrap=tk.WORD, font=("Arial", 11),
            bg=bg_color, fg=fg_color,
            borderwidth=0, highlightthickness=0,
            height=len(lines) + 2
        )
        msg_label.insert("1.0", message)
        msg_label.config(state="disabled") # Read-only
        msg_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Button
        btn_frame = tk.Frame(self.dialog, bg=bg_color, pady=15)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(btn_frame, text="OK", command=self.dialog.destroy).pack()
        
        self.dialog.bind("<Return>", lambda e: self.dialog.destroy())
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())
        
        self.dialog.focus_set()
        self.dialog.wait_window()

class StyledMessageBox:
    """Wrapper to replace standard messagebox with CustomMessageDialog."""
    def __init__(self, parent_gui: 'ChatbotGUI'):
        self.gui = parent_gui
        
    def showinfo(self, title, message, parent=None):
        CustomMessageDialog(self.gui.root, title, message, "info", self.gui.dark_mode)
        
    def showwarning(self, title, message, parent=None):
        CustomMessageDialog(self.gui.root, title, message, "warning", self.gui.dark_mode)
        
    def showerror(self, title, message, parent=None):
        CustomMessageDialog(self.gui.root, title, message, "error", self.gui.dark_mode)
