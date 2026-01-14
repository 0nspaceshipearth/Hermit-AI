
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

"""Local LLM chat integration."""

import sys
import json
from typing import List, Iterable
from chatbot.models import Message
from chatbot import config
from chatbot.model_manager import ModelManager



def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:CHAT] {msg}", file=sys.stderr)


# Global status callback for UI updates
_status_callback = None

def set_status_callback(callback):
    """Set a callback function to receive status updates during RAG processing."""
    global _status_callback
    _status_callback = callback

def _update_status(status: str):
    """Call the status callback if set."""
    global _status_callback
    if _status_callback:
        try:
            _status_callback(status)
        except:
            pass


def stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """Stream chat with local model."""
    debug_print(f"stream_chat called with model='{model}'")
    
    try:
        # Get model instance (caching handled by manager)
        # Use global config context or default to 16384
        n_ctx = getattr(config, 'DEFAULT_CONTEXT_SIZE', 16384)
        llm = ModelManager.get_model(model, n_ctx=n_ctx)
        
        debug_print("Starting local generation stream...")
        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.3, # Lower temp for more focused answers
            repeat_penalty=1.2, # Stronger penalty to prevent "But wait" loops
            max_tokens=None  # Allow full generation
        )
        
        buffer = ""
        in_thought_block = False

        for chunk in stream:
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            if 'content' in delta and delta['content'] is not None:
                content = str(delta['content'])
                buffer += content

                while True:
                    if in_thought_block:
                        end_tag = "</thought>"
                        if end_tag in buffer:
                            _, after = buffer.split(end_tag, 1)
                            buffer = after
                            in_thought_block = False
                        else:
                            break
                    else:
                        start_tag = "<thought>"
                        if start_tag in buffer:
                            before, after = buffer.split(start_tag, 1)
                            if before:
                                yield before
                            buffer = after
                            in_thought_block = True
                        else:
                            # Optimization: only check potential start tag
                            if "<" in buffer:
                                safe_len = buffer.find("<")
                                if safe_len > 0:
                                    yield buffer[:safe_len]
                                    buffer = buffer[safe_len:]
                                break
                            else:
                                if buffer:
                                    yield buffer
                                    buffer = ""
                                break
        
        # Yield remaining buffer if not in thought block
        if buffer and not in_thought_block:
            yield buffer
            
        debug_print(f"Stream complete.")
            
    except Exception as e:
        debug_print(f"Local inference error: {e}")
        raise RuntimeError(f"Local model generation failed: {e}")


def full_chat(model: str, messages: List[dict]) -> str:
    """Full chat with local model."""
    debug_print(f"full_chat called with model='{model}'")
    
    try:
        n_ctx = getattr(config, 'DEFAULT_CONTEXT_SIZE', 16384)
        llm = ModelManager.get_model(model, n_ctx=n_ctx)
        
        resp = llm.create_chat_completion(
            messages=messages,
            stream=False,
            temperature=0.3,
            repeat_penalty=1.2
        )
        debug_print(f"RAW LLM RESP: {resp}")
        
        return resp['choices'][0]['message']['content']
            
    except Exception as e:
        debug_print(f"Local inference error: {e}")
        raise RuntimeError(f"Local model generation failed: {e}")


from chatbot.rag import RAGSystem

# Global RAG instance
_rag_system = None

def get_rag_system():
    global _rag_system
    debug_print("get_rag_system called")
    if _rag_system is None:
        debug_print("RAG system not initialized, checking for resources...")
        # Initialize only if index exists
        import os
        if os.path.exists("data/index/faiss.index") or os.path.exists("wikipedia_en_all_maxi_2025-08.zim") or any(f.endswith(".zim") for f in os.listdir('.')):
            try:
                print("Initializing RAG system (Hybrid/Fast)...")
                _rag_system = RAGSystem()
                _rag_system.load_resources()
                debug_print("RAG system initialized successfully")
            except Exception as e:
                print(f"Failed to load RAG: {e}")
                debug_print(f"RAG initialization failed: {e}")
                _rag_system = None
        else:
            debug_print("No RAG resources found (no index or ZIM files)")
    else:
        debug_print("RAG system already initialized")
    return _rag_system

def retrieve_and_display_links(query: str) -> List[dict]:
    """Retrieve and format links for link mode."""
    debug_print("="*60)
    debug_print(f"retrieve_and_display_links called with query='{query}'")
    
    # Get RAG system
    rag = get_rag_system()
    if not rag:
        debug_print("No RAG system available")
        return []
    
    
    try:
        # Retrieve relevant documents
        _update_status("Searching knowledge base...")
        results = rag.retrieve(query, top_k=8)
        debug_print(f"RAG retrieved {len(results)} results")
        
        if not results:
            _update_status("No results found")
            return []
        
        _update_status("Formatting results...")
        # Convert to link format
        links = []
        seen_titles = set()
        
        for result in results:
            metadata = result.get('metadata', {})
            title = metadata.get('title', 'Unknown Title')
            
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Create snippet from text
            text = result.get('text', '')
            snippet = text[:200] + '...' if len(text) > 200 else text
            
            link_data = {
                'title': title,
                'path': metadata.get('path', ''),
                'score': result.get('score', 0.0),
                'snippet': snippet,
                'metadata': metadata,
                'search_context': result.get('search_context', {})
            }
            links.append(link_data)
        
        debug_print(f"Returning {len(links)} unique links")
        return links[:10]  # Limit to 10 links
        
    except Exception as e:
        debug_print(f"Error in retrieve_and_display_links: {type(e).__name__}: {e}")
        return []


def build_messages(system_prompt: str, history: List[Message], user_query: str = None) -> List[dict]:
    """Build message list for local LLM with RAG augmentation."""
    debug_print("="*60)
    debug_print("build_messages START")
    debug_print(f"system_prompt length: {len(system_prompt)} chars")
    debug_print(f"history length: {len(history)} messages")
    debug_print(f"user_query: '{user_query}'")
    
    # 1. Retrieve context if we have a user query
    context_text = ""
    rag = get_rag_system()
    
    # 1. Detect Intent
    from chatbot.intent import detect_intent
    # Identify the actual query. If user_query is provided, use it.
    # Otherwise check the last message in history if it's from user.
    query_text = user_query
    if not query_text and history and history[-1].role == 'user':
        query_text = history[-1].content
        debug_print(f"Using last message from history as query: '{query_text}'")
    else:
        debug_print(f"Using provided user_query: '{query_text}'")
        
    intent = detect_intent(query_text or "")
    debug_print(f"Intent Detection Result: mode='{intent.mode_name}', should_retrieve={intent.should_retrieve}")
    _update_status("Analyzing query")
    
    # 2. Retrieve context (If Intent allows)
    debug_print("-" * 60)
    debug_print("RAG RETRIEVAL PHASE")
    context_text = ""
    rag = get_rag_system()
        
    if rag and query_text and intent.should_retrieve:
        debug_print(f"Conditions met for RAG retrieval: rag={rag is not None}, query_text='{query_text}', should_retrieve={intent.should_retrieve}")
        try:
            _update_status("Searching knowledge base")
            debug_print(f"Calling rag.retrieve with query='{query_text}', top_k=4")
            results = rag.retrieve(query_text, top_k=4)
            _update_status("Processing results")
            debug_print(f"RAG retrieve returned {len(results)} results")
            
            if results:
                debug_print("Processing RAG results...")
                context_text = "\n\nRelevant Context via RAG:\n"
                total_context_chars = 0
                max_context_chars = 16000 # ~4000 tokens, leaving plenty for generation
                
                for i, r in enumerate(results, 1):
                    meta = r['metadata']
                    text = r['text']
                    title = meta.get('title', 'Unknown')
                    score = r.get('score', 0.0)
                    
                    # Truncate extremely long chunks just in case
                    if len(text) > 4000:
                        text = text[:4000] + "...(truncated)"
                    
                    chunk_text = f"\n--- Source {i}: {title} ---\n{text}\n"
                    
                    if total_context_chars + len(chunk_text) > max_context_chars:
                        debug_print(f"Context limit reached ({max_context_chars} chars). Stopping at result {i}.")
                        break
                        
                    context_text += chunk_text
                    total_context_chars += len(chunk_text)
                    debug_print(f"Result {i}: title='{title}', score={score:.4f}, text_length={len(text)} chars")
                
                # Append verified facts found by Joint 4
                facts_list = results[0].get('search_context', {}).get('facts', []) if results else []
                
                # SOFT BLOCK: Check for premise verification failure (SYSTEM ALERT)
                # If Joint 4 determined the sources are irrelevant, add disclaimer but still let model answer
                if facts_list and any("[SYSTEM ALERT" in str(fact) for fact in facts_list):
                    debug_print("SOFT BLOCK: Premise verification failed - adding disclaimer")
                    context_text += "\n\nIMPORTANT DISCLAIMER \n"
                    context_text += "The retrieved sources do NOT contain relevant information for this query.\n"
                    context_text += "You may answer using your general knowledge, but clearly state at the START:\n"
                    context_text += "'Note: I could not find sources for this in my knowledge base. The following is based on general knowledge and may contain inaccuracies.'\n"
                    context_text += "Then provide your best answer from training data.\n"
                
                if facts_list and not any("[SYSTEM ALERT" in str(fact) for fact in facts_list):
                    debug_print(f"Adding {len(facts_list)} verified facts to context")
                    context_text += "\n\n=== VERIFIED FACTUAL DETAILS (Extracted from Source) ===\n"
                    context_text += "The following details were explicitly found in the source text for your query:\n"
                    for fact in facts_list:
                        context_text += f"- {fact}\n"
                    context_text += "======================================================\n"

                debug_print(f"Context assembled: {len(context_text)} chars total")
            else:
                debug_print("No results returned from RAG")
                if config.STRICT_RAG_MODE:
                    debug_print("STRICT_RAG_MODE=True, will refuse to answer")
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index.\n" \
                                   "Instructions: You MUST refuse to answer the user's question because no relevant context was found.\n" \
                                   "Reply EXACTLY with: 'I do not have enough information in my knowledge base to answer this question.'"
                else:
                    debug_print("STRICT_RAG_MODE=False, will use general knowledge")
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index. Answering based on general knowledge.\n"
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            debug_print(f"RAG retrieval exception: {type(e).__name__}: {e}")
    else:
        debug_print(f"Skipping RAG retrieval: rag={rag is not None}, query_text={bool(query_text)}, should_retrieve={intent.should_retrieve}")

    # 3. Augment system prompt with Context AND Intent Instructions
    debug_print("-" * 60)
    debug_print("MESSAGE CONSTRUCTION PHASE")
    
    # Define instructions clearly
    instructions = f"\n\nCRITICAL INSTRUCTIONS:\n" \
                   f"1. USE THE CONTEXT: Answer based ONLY on the provided context below.\n" \
                   f"2. BE ACCURATE: Do not make up facts. If the answer isn't there, say 'I don't know' and STOP.\n" \
                   f"3. VERIFY PREMISES: If the user asks a leading question (e.g., 'When did X do Y?') and the context says X *never* did Y, you MUST correct the premise. Do NOT assume the user is right.\n" \
                   f"4. HANDLE CONFLICTS: If context has conflicting info, state BOTH sides. Do NOT debate yourself (e.g. 'But wait').\n" \
                   f"5. SYNTHESIZE: Combine the 'Verified Factual Details' and the 'Source' text to provide a complete, natural answer that directly addresses the user's question.\n" \
                   f"6. KEEP IT CONCISE BUT COMPLETE: Do not be overly verbose, but ensure all relevant details answering 'How', 'Why', 'When', etc. are included.\n" \
                   f"7. IGNORE IRRELEVANT TEXT: The context may contain unrelated articles. Focus only on what answers the question."

    final_system_prompt = system_prompt + intent.system_instruction + instructions
    
    debug_print(f"Base system_prompt + intent instruction + instructions = {len(final_system_prompt)} chars")
    
    if context_text:
        final_system_prompt += "\n\n=== RETRIEVED CONTEXT ===\n" + context_text
        debug_print(f"Added context. Final system_prompt = {len(final_system_prompt)} chars")
    else:
        debug_print("No context to add")

    messages = [{"role": "system", "content": final_system_prompt}]
    debug_print(f"Added system message (length: {len(final_system_prompt)} chars)")
    
    for msg in history:
        if msg.role in ["user", "assistant", "system"]:
            messages.append({"role": msg.role, "content": msg.content})
            debug_print(f"Added {msg.role} message (length: {len(msg.content)} chars)")
            
    debug_print(f"Total messages constructed: {len(messages)}")
    debug_print("build_messages END")
    debug_print("="*60)
    print(f"\nGenerating response...")
    return messages