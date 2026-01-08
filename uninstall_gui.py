#!/usr/bin/env python3
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from pathlib import Path

class UninstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hermit Uninstaller")
        self.root.geometry("500x450")
        
        # Style
        style = ttk.Style()
        style.configure("Bold.TLabel", font=("Helvetica", 10, "bold"))
        style.configure("Warning.TLabel", foreground="red")
        
        # Header
        header = ttk.Label(root, text="Hermit Cleanup Tool", font=("Helvetica", 16, "bold"))
        header.pack(pady=10)
        
        ttk.Label(root, text="Select components to remove:", style="Bold.TLabel").pack(anchor="w", padx=20)
        
        # Checkbox Variables
        self.vars = {
            "venv": tk.BooleanVar(value=True),
            "models": tk.BooleanVar(value=False),
            "data": tk.BooleanVar(value=True),
            "hermit": tk.BooleanVar(value=False),
            "forge": tk.BooleanVar(value=False)
        }
        
        # Paths
        self.base_dir = Path(__file__).parent.absolute()
        self.paths = {
            "venv": self.base_dir / "venv",
            "models": self.base_dir / "shared_models",
            "data": self.base_dir / "data",
            "hermit": Path("/usr/local/bin/hermit"),
            "forge": Path("/usr/local/bin/forge")
        }
        
        # Components Frame
        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        # Checkboxes
        self.create_checkbox("venv", "Virtual Environment (venv/)", 
                           "Removes python libraries and environment.")
        self.create_checkbox("models", "AI Models (shared_models/)", 
                           "Removes downloaded GGUF models (Large files!).")
        self.create_checkbox("data", "Search Indexes (data/)", 
                           "Removes cached vectors and JIT indexes.")
        self.create_checkbox("hermit", "System Command (hermit)", 
                           "Removes the /usr/local/bin/hermit wrapper (Requires Sudo).")
        self.create_checkbox("forge", "System Command (forge)", 
                           "Removes the /usr/local/bin/forge wrapper (Requires Sudo).")
        
        # Protection Note
        protection_frame = ttk.LabelFrame(root, text="🛡️ Safety Protection", padding=10)
        protection_frame.pack(fill="x", padx=20, pady=10)
        ttk.Label(protection_frame, text="Your Wikipedia .zim files are SAFE.", foreground="green").pack(anchor="w")
        ttk.Label(protection_frame, text="This tool will NEVER delete .zim files.", foreground="green").pack(anchor="w")
        
        # Size Label
        self.size_label = ttk.Label(root, text="Calculating space...", font=("Helvetica", 10))
        self.size_label.pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=20, pady=20)
        
        ttk.Button(btn_frame, text="Cancel", command=root.quit).pack(side="left")
        ttk.Button(btn_frame, text="Uninstall Selected", command=self.confirm_uninstall).pack(side="right")
        
        # Initial calculation
        self.update_size_estimate()

    def create_checkbox(self, key, title, desc):
        frame = ttk.Frame(self.frame)
        frame.pack(fill="x", pady=5)
        
        cb = ttk.Checkbutton(frame, text=title, variable=self.vars[key], command=self.update_size_estimate)
        cb.pack(anchor="w")
        
        ttk.Label(frame, text=f"   {desc}", font=("Helvetica", 8), foreground="gray").pack(anchor="w")

    def get_dir_size(self, path):
        total = 0
        try:
            if path.is_file():
                return path.stat().st_size
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    def format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def update_size_estimate(self):
        total_size = 0
        for key, var in self.vars.items():
            if var.get():
                path = self.paths[key]
                if path.exists():
                    total_size += self.get_dir_size(path)
        
        self.size_label.config(text=f"Space to be freed: {self.format_size(total_size)}")

    def confirm_uninstall(self):
        # Double check
        selected = [k for k, v in self.vars.items() if v.get()]
        if not selected:
            messagebox.showinfo("Nothing Selected", "Please select components to uninstall.")
            return

        msg = "Are you sure you want to remove:\n\n"
        for key in selected:
            msg += f"- {key}\n"
        msg += "\nThis action cannot be undone."
        
        if messagebox.askyesno("Confirm Uninstall", msg):
            self.perform_uninstall(selected)

    def perform_uninstall(self, selected_keys):
        errors = []
        success = []
        
        for key in selected_keys:
            path = self.paths[key]
            
            # EXTRA SAFETY CHECK: Ensure we never delete a ZIM file
            if str(path).endswith(".zim"):
                errors.append(f"Skipped {key}: Safety check prevented deleting .zim file")
                continue
                
            try:
                if not path.exists():
                    continue
                    
                if key in ("hermit", "forge"):
                    # Requires sudo
                    self.run_sudo_remove(path)
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                success.append(key)
            except Exception as e:
                errors.append(f"Failed to remove {key}: {str(e)}")

        # Clean __pycache__ if removing venv
        if "venv" in selected_keys:
            try:
                for p in self.base_dir.rglob("__pycache__"):
                    shutil.rmtree(p)
            except:
                pass

        if errors:
            messagebox.showerror("Completed with Errors", "\n".join(errors))
        else:
            messagebox.showinfo("Success", f"Successfully removed: {', '.join(success)}")
        
        self.root.quit()

    def run_sudo_remove(self, path):
        # Run sudo rm command
        cmd = ["sudo", "rm", "-f", str(path)]
        # This might fail if no gui sudo prompt, but apt/pkexec usually handles it 
        # or it is run from a terminal that can prompt.
        # Alternatively, we assume user runs uninstall script with rights or we fail.
        # Using pkexec for gui prompt preferred if available
        if shutil.which("pkexec"):
             subprocess.run(["pkexec", "rm", "-f", str(path)], check=True)
        else:
             # Fallback to sudo (terminal prompt)
             subprocess.run(cmd, check=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = UninstallerGUI(root)
    root.mainloop()
