
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

import sys
import os
import cmd
from typing import List, Optional
import libzim

from chatbot.rag import RAGSystem, TextProcessor
from chatbot import config
from chatbot.chat import build_messages, stream_chat, clear_runtime_memory
from chatbot.models import Message

class ChatbotCLI(cmd.Cmd):
    """Command-line interface for Hermit."""
    
    intro = 'Welcome to Hermit CLI. Type help or ? to list commands.\n'
    prompt = '(hermit) '
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.rag = None
        self.last_results = []
        
        print(f"Initializing RAG System (Model: {model_name})...")
        try:
            self.rag = RAGSystem()
            # self.rag.load_resources() # Handled in __init__
            
            # Inject our RAG instance into the chat module so it doesn't try to reload it
            import chatbot.chat
            chatbot.chat._rag_system = self.rag
            
            print("RAG System Ready.")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            print("Some functionality may be limited.")
            
        self.history = []

    def do_clear(self, arg):
        """Clear in-session conversation memory."""
        self.history.clear()
        clear_runtime_memory(reset_rag=False)
        print("Session memory cleared.")

    def do_search(self, arg):
        """Search for articles: search <query>"""
        if not arg:
            print("Usage: search <query>")
            return
            
        print(f"Searching for '{arg}'...")
        if not self.rag:
            print("RAG system not available.")
            return
            
        try:
            results = self.rag.retrieve(arg, top_k=10)
            self.last_results = results
            
            if not results:
                print("No results found.")
                return
                
            print(f"\nFound {len(results)} results:")
            for i, res in enumerate(results, 1):
                meta = res.get('metadata', {})
                title = meta.get('title', 'Unknown')
                path = meta.get('path', 'Unknown')
                score = res.get('score', 0.0)
                print(f"{i}. {title} (Score: {score:.2f}) [Path: {path}]")
            print("\nType 'read <number>' or 'read <path>' to view an article.")
            
        except Exception as e:
            print(f"Search failed: {e}")

    def do_read(self, arg):
        """Read an article: read <number> or read <path>"""
        if not arg:
            print("Usage: read <result_number> or read <path>")
            return
            
        path_to_read = None
        
        # Try to parse as index
        if arg.isdigit() and self.last_results:
            idx = int(arg) - 1
            if 0 <= idx < len(self.last_results):
                path_to_read = self.last_results[idx]['metadata'].get('path')
                print(f"Selected result #{arg}: {path_to_read}")
            else:
                 print(f"Invalid index. Valid range: 1-{len(self.last_results)}")
                 return
        else:
            # Treat as path
            path_to_read = arg
            
        if not path_to_read:
            print("No path resolved.")
            return
            
            
        # Get context if available
        terms = []
        if self.last_results and arg.isdigit():
             idx = int(arg) - 1
             if 0 <= idx < len(self.last_results):
                 ctx = self.last_results[idx].get('search_context', {})
                 terms = ctx.get('entities', [])

        self._open_zim_entry(path_to_read, highlight_terms=terms)

    def _open_zim_entry(self, path, highlight_terms=None):
        """Open ZIM entry with smart fallback logic."""
        # Find ZIM file
        zim_files = [f for f in os.listdir('.') if f.endswith('.zim')]
        if not zim_files:
            print("Error: No ZIM files found in current directory.")
            return

        zim_file = zim_files[0]
        try:
            zim = libzim.Archive(zim_file)
        except Exception as e:
            print(f"Error opening ZIM archive: {e}")
            return

        entry = None
        
        # Helper
        def try_find(p):
            try:
                return zim.get_entry_by_path(p)
            except:
                return None

        # Strategy 1: Direct
        entry = try_find(path)
        
        # Strategy 2: Title
        if not entry:
            try:
                entry = zim.get_entry_by_title(path)
            except:
                pass
                
        # Strategy 3: Smart Fallback (Variations)
        if not entry:
            variations = []
            if ' ' in path: variations.append(path.replace(' ', '_'))
            if '_' in path: variations.append(path.replace('_', ' '))
            variations.append(path.title())
            if ' ' in path: variations.append(path.title().replace(' ', '_')) # Title_Case
            
            paths_to_try = [path] + variations
            for candidate in paths_to_try:
                attempts = [candidate]
                if not candidate.startswith('/'): attempts.append('/' + candidate)
                if candidate.startswith('/'): attempts.append(candidate[1:])
                
                for attempt in attempts:
                    if config.DEBUG: print(f"[DEBUG] Trying: {attempt}")
                    entry = try_find(attempt)
                    if entry: break
                if entry: break
        
        if not entry or entry.is_redirect:
            print(f"Article not found: '{path}'")
            return
            
        item = entry.get_item()
        if item.mimetype != 'text/html':
            print(f"Cannot render non-text content ({item.mimetype})")
            return
            
        print(f"\n=== {entry.title} ===\n")
        try:
            # Use the robust renderable extraction
            content = TextProcessor.extract_renderable_text(item.content)
            
            # Apply ANSI Highlighting
            if highlight_terms:
                import re
                # ANSI Yellow + Bold
                START_HL = "\033[1;33m"
                END_HL = "\033[0m"
                
                for term in highlight_terms:
                    if not term or len(term) < 3: continue
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    content = pattern.sub(lambda m: f"{START_HL}{m.group()}{END_HL}", content)
            
            print(content)
            print("\n" + "="*40 + "\n")
        except Exception as e:
            print(f"Error rendering content: {e}")

    def do_quit(self, arg):
        """Exit the CLI."""
        self.history.clear()
        clear_runtime_memory(reset_rag=True)
        print("Goodbye!")
        return True
        
    def do_exit(self, arg):
        """Exit the CLI."""
        return self.do_quit(arg)
    
    def default(self, line):
        """Handle chat interactions."""
        # Treat as chat
        if not line: return
        
        # Check for empty lines or comments
        if line.strip().startswith('#'): return

        # Append to history FIRST (Critical for model to see the question)
        self.history.append(Message(role="user", content=line))

        print(f"\nThinking...")
        try:
            # Build messages
            messages = build_messages(config.SYSTEM_PROMPT, self.history)
            
            # Stream response
            print(f"Hermit: ", end="", flush=True)
            full_response = ""
            for chunk in stream_chat(self.model_name, messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            
            # Update history with assistant response
            self.history.append(Message(role="assistant", content=full_response))
            
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"\nError: {e}")
        
    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        self.history.clear()
        clear_runtime_memory(reset_rag=True)
        print("")
        return True

if __name__ == '__main__':
    cli = ChatbotCLI(config.DEFAULT_MODEL)
    cli.cmdloop()
