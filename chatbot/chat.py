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
import time
from typing import Dict, Iterable, List, Optional, Tuple
from chatbot.models import Message
from chatbot import config
from chatbot.model_manager import ModelManager
from chatbot.runtime_profiles import record_generation_metrics, get_model_lane_stats


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
        except Exception:
            pass


def clear_runtime_memory(reset_rag: bool = False) -> None:
    """Clear in-process session caches (no disk persistence)."""
    try:
        from chatbot.rag.search import clear_title_cache
        clear_title_cache()
    except Exception:
        pass

    if reset_rag:
        global _rag_system
        _rag_system = None


def _estimate_tokens_from_chars(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _extract_query_text(messages: List[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _extract_context_from_system_prompt(system_content: str) -> Tuple[str, str]:
    marker = "\n\n=== REFERENCE INFORMATION ===\n"
    if marker in system_content:
        prefix, suffix = system_content.split(marker, 1)
        return prefix, marker + suffix
    return system_content, ""


def _extract_answer_type_from_messages(messages: List[dict]) -> Optional[str]:
    system_content = messages[0].get("content", "") if messages else ""
    marker = "The user specifically asked for **"
    if marker not in system_content:
        return None
    try:
        return system_content.split(marker, 1)[1].split("**", 1)[0].strip().lower()
    except Exception:
        return None


def _collect_verified_facts(context_text: str) -> List[str]:
    facts: List[str] = []
    marker = "=== VERIFIED FACTUAL DETAILS (Extracted from Source) ==="
    if marker not in context_text:
        return facts
    section = context_text.split(marker, 1)[1]
    if "======================================================" in section:
        section = section.split("======================================================", 1)[0]
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            facts.append(line[2:].strip())
    return facts


def _summarize_context_for_lane(context_text: str, lane: str) -> str:
    if not context_text:
        return ""

    facts = _collect_verified_facts(context_text)
    budgets = {
        "sprint": getattr(config, "FINAL_MAX_SOURCE_CHARS_SPRINT", 2500),
        "cruise": getattr(config, "FINAL_MAX_SOURCE_CHARS_CRUISE", 7000),
        "beast": getattr(config, "FINAL_MAX_SOURCE_CHARS_BEAST", 12000),
    }
    budget = budgets.get(lane, budgets["cruise"])

    if lane == "sprint" and facts and getattr(config, "FINAL_FACTS_FIRST", True):
        fact_lines = "\n".join(f"- {fact}" for fact in facts[:8])
        compact = (
            "\n\n=== COMPACT FACTS ===\n"
            "Use these verified facts first. Keep the answer direct and concise.\n"
            f"{fact_lines}\n"
            "=====================\n"
        )
        remaining = max(0, budget - len(compact))
        if remaining <= 0:
            return compact
        return compact + context_text[:remaining]

    return context_text[:budget]


def _choose_generation_lane(model: str, messages: List[dict]) -> Dict[str, object]:
    mode = getattr(config, "FINAL_GENERATION_MODE", "auto")
    query_text = _extract_query_text(messages)
    query_lower = query_text.lower()
    answer_type = _extract_answer_type_from_messages(messages)
    system_content = messages[0].get("content", "") if messages else ""
    _, context_text = _extract_context_from_system_prompt(system_content)

    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    prompt_tokens = _estimate_tokens_from_chars("".join(str(m.get("content", "")) for m in messages))
    context_chars = len(context_text)
    fact_count = len(_collect_verified_facts(context_text))

    if mode in {"sprint", "cruise", "beast"}:
        lane = mode
    else:
        simple_answer_types = {"birthdate", "birthplace", "education", "inventor", "death_date", "death_cause", "language", "measurement"}
        complex_markers = ["compare", "difference", "versus", " vs ", "relationship", "why", "how does", "cause"]
        is_complex = any(marker in query_lower for marker in complex_markers)

        if is_complex or prompt_tokens > 2400 or context_chars > 9000:
            lane = "beast"
        elif answer_type in simple_answer_types and fact_count >= 2 and prompt_tokens < 2200:
            lane = "sprint"
        else:
            lane = "cruise"

        stats = get_model_lane_stats(model, lane)
        if lane == "beast" and stats and stats.get("samples", 0) >= 3:
            avg_first = stats.get("avg_first_token_seconds")
            avg_duration = stats.get("avg_duration_seconds")
            if (avg_first and avg_first > 6.0) or (avg_duration and avg_duration > 20.0 and fact_count >= 3):
                lane = "cruise"

    ctx_map = {
        "sprint": getattr(config, "FINAL_SPRINT_CONTEXT_SIZE", 4096),
        "cruise": getattr(config, "FINAL_CRUISE_CONTEXT_SIZE", 8192),
        "beast": getattr(config, "FINAL_BEAST_CONTEXT_SIZE", 12288),
    }
    max_tokens_map = {
        "sprint": getattr(config, "FINAL_SPRINT_MAX_TOKENS", 160),
        "cruise": getattr(config, "FINAL_CRUISE_MAX_TOKENS", 320),
        "beast": getattr(config, "FINAL_BEAST_MAX_TOKENS", 640),
    }

    if answer_type in {"birthdate", "birthplace", "inventor", "death_date"} and lane == "cruise":
        max_tokens_map["cruise"] = min(max_tokens_map["cruise"], 220)

    return {
        "lane": lane,
        "n_ctx": ctx_map[lane],
        "max_tokens": max_tokens_map[lane],
        "answer_type": answer_type,
        "prompt_chars": prompt_chars,
        "context_chars": context_chars,
        "fact_count": fact_count,
    }


def _prepare_messages_for_generation(messages: List[dict], profile: Dict[str, object]) -> List[dict]:
    if not messages:
        return messages

    prepared = [dict(message) for message in messages]
    lane = str(profile.get("lane", "cruise"))
    system_content = str(prepared[0].get("content", ""))
    prefix, context_text = _extract_context_from_system_prompt(system_content)
    compact_context = _summarize_context_for_lane(context_text, lane)

    lane_instructions = {
        "sprint": "\n\nFINAL ANSWER MODE: SPRINT\nKeep the answer compact, direct, and low-latency. Prefer the verified facts. Avoid extra exposition unless necessary.",
        "cruise": "\n\nFINAL ANSWER MODE: CRUISE\nAim for a complete but efficient answer. Prefer verified facts and only use source detail when needed.",
        "beast": "\n\nFINAL ANSWER MODE: BEAST\nDo deeper synthesis when helpful, but stay grounded in the verified facts and cited source material.",
    }

    prepared[0]["content"] = prefix + lane_instructions.get(lane, lane_instructions["cruise"]) + compact_context
    return prepared


def stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """Stream chat with local model."""
    debug_print(f"stream_chat called with model='{model}'")

    try:
        profile = _choose_generation_lane(model, messages)
        prepared_messages = _prepare_messages_for_generation(messages, profile)
        n_ctx = int(profile["n_ctx"])
        max_tokens = int(profile["max_tokens"])
        llm = ModelManager.get_model(model, n_ctx=n_ctx)

        debug_print(f"Starting local generation stream with lane={profile['lane']} ctx={n_ctx} max_tokens={max_tokens}...")
        _update_status(f"Generating response ({profile['lane']})...")
        stream = llm.create_chat_completion(
            messages=prepared_messages,
            stream=True,
            temperature=0.3,
            repeat_penalty=1.2,
            max_tokens=max_tokens
        )

        buffer = ""
        in_thought_block = False
        started_at = time.time()
        first_token_at = None
        yielded_chars = 0

        for chunk in stream:
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            if 'content' in delta and delta['content'] is not None:
                content = str(delta['content'])
                if content and first_token_at is None:
                    first_token_at = time.time()
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
                                yielded_chars += len(before)
                                yield before
                            buffer = after
                            in_thought_block = True
                        else:
                            if "<" in buffer:
                                safe_len = buffer.find("<")
                                if safe_len > 0:
                                    chunk_out = buffer[:safe_len]
                                    yielded_chars += len(chunk_out)
                                    yield chunk_out
                                    buffer = buffer[safe_len:]
                                break
                            else:
                                if buffer:
                                    yielded_chars += len(buffer)
                                    yield buffer
                                    buffer = ""
                                break

        if buffer and not in_thought_block:
            yielded_chars += len(buffer)
            yield buffer

        duration = time.time() - started_at
        record_generation_metrics(
            model=model,
            lane=str(profile["lane"]),
            prompt_chars=sum(len(str(m.get("content", ""))) for m in prepared_messages),
            output_chars=yielded_chars,
            duration_seconds=duration,
            first_token_seconds=(first_token_at - started_at) if first_token_at else None,
            metadata={
                "n_ctx": n_ctx,
                "max_tokens": max_tokens,
                "answer_type": profile.get("answer_type"),
                "fact_count": profile.get("fact_count"),
            },
        )
        debug_print("Stream complete.")

    except Exception as e:
        debug_print(f"Local inference error: {e}")
        raise RuntimeError(f"Local model generation failed: {e}")


def full_chat(model: str, messages: List[dict]) -> str:
    """Full chat with local model."""
    debug_print(f"full_chat called with model='{model}'")

    try:
        profile = _choose_generation_lane(model, messages)
        prepared_messages = _prepare_messages_for_generation(messages, profile)
        n_ctx = int(profile["n_ctx"])
        max_tokens = int(profile["max_tokens"])
        llm = ModelManager.get_model(model, n_ctx=n_ctx)

        _update_status(f"Generating response ({profile['lane']})...")
        started_at = time.time()
        resp = llm.create_chat_completion(
            messages=prepared_messages,
            stream=False,
            temperature=0.3,
            repeat_penalty=1.2,
            max_tokens=max_tokens,
        )
        debug_print(f"RAW LLM RESP: {resp}")

        content = resp['choices'][0]['message']['content']
        duration = time.time() - started_at
        record_generation_metrics(
            model=model,
            lane=str(profile["lane"]),
            prompt_chars=sum(len(str(m.get("content", ""))) for m in prepared_messages),
            output_chars=len(content or ""),
            duration_seconds=duration,
            metadata={
                "n_ctx": n_ctx,
                "max_tokens": max_tokens,
                "answer_type": profile.get("answer_type"),
                "fact_count": profile.get("fact_count"),
            },
        )
        return content

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
        import os
        import glob

        zim_files = glob.glob("*.zim")
        zim_paths = [os.path.abspath(z) for z in zim_files]

        has_index = os.path.exists("data/indices/content_index.faiss") or os.path.exists("data/indices/title_index.faiss")

        if has_index or zim_paths:
            try:
                print(f"Initializing RAG system (Multi-ZIM Mode)...")
                if zim_paths:
                    print(f"Found {len(zim_paths)} ZIM file(s)")
                else:
                    print("Warning: No ZIM files found, relying on existing index only.")

                _rag_system = RAGSystem(zim_paths=zim_paths)
                debug_print("RAG system initialized successfully (multi-ZIM)")
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

    rag = get_rag_system()
    if not rag:
        debug_print("No RAG system available")
        return []

    try:
        _update_status("Searching knowledge base...")
        results = rag.retrieve(query, top_k=8)
        debug_print(f"RAG retrieved {len(results)} results")

        if not results:
            _update_status("No results found")
            return []

        _update_status("Formatting results...")
        links = []
        seen_titles = set()

        for result in results:
            metadata = result.get('metadata', {})
            title = metadata.get('title', 'Unknown Title')

            if title in seen_titles:
                continue
            seen_titles.add(title)

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
        return links[:10]

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

    context_text = ""

    from chatbot.intent import detect_intent
    query_text = user_query
    if not query_text and history and history[-1].role == 'user':
        query_text = history[-1].content
        debug_print(f"Using last message from history as query: '{query_text}'")
    else:
        debug_print(f"Using provided user_query: '{query_text}'")

    intent = detect_intent(query_text or "")
    debug_print(f"Intent Detection Result: mode='{intent.mode_name}', should_retrieve={intent.should_retrieve}")
    _update_status("Analyzing query")

    debug_print("-" * 60)
    debug_print("RAG RETRIEVAL PHASE")
    rag = get_rag_system()
    results = []

    if rag and query_text and intent.should_retrieve:
        debug_print(f"Conditions met for RAG retrieval: rag={rag is not None}, query_text='{query_text}', should_retrieve={intent.should_retrieve}")
        try:
            _update_status("Searching knowledge base")
            debug_print(f"Calling rag.retrieve with query='{query_text}', top_k=3")
            results = rag.retrieve(query_text, top_k=3)
            _update_status("Processing results")
            debug_print(f"RAG retrieve returned {len(results)} results")

            if results:
                debug_print("Processing RAG results...")
                context_text = "\n\n=== REFERENCE INFORMATION ===\n"
                total_context_chars = 0
                max_context_chars = 10000

                for i, r in enumerate(results, 1):
                    meta = r['metadata']
                    text = r['text']
                    title = meta.get('title', 'Unknown')
                    score = r.get('score', 0.0)

                    if len(text) > 4000:
                        text = text[:4000] + "...(truncated)"

                    chunk_text = f"\n--- Source {i}: {title} ---\n{text}\n"

                    if total_context_chars + len(chunk_text) > max_context_chars:
                        debug_print(f"Context limit reached ({max_context_chars} chars). Stopping at result {i}.")
                        break

                    context_text += chunk_text
                    total_context_chars += len(chunk_text)
                    debug_print(f"Result {i}: title='{title}', score={score:.4f}, text_length={len(text)} chars")

                comparison_data = results[0].get('search_context', {}).get('comparison_data')
                if comparison_data:
                    debug_print("Adding Comparison Card to context")
                    context_text += "\n\n=== COMPARISON DATA CARD (Structured Synthesis) ===\n"
                    context_text += f"Comparison Dimension: {comparison_data.get('dimension')}\n"
                    context_text += f"Conclusion: {comparison_data.get('conclusion')}\n"
                    context_text += "Data Points:\n"
                    for item in comparison_data.get('data', []):
                        context_text += f"- {item.get('entity')}: {item.get('value')} (Source Chunks)\n"
                    context_text += "===================================================\n"

                facts_list = results[0].get('search_context', {}).get('facts', []) if results else []

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
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index. Please use your general knowledge and any partial matches in the context to provide the most helpful answer possible, while maintaining factual accuracy.\n"
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"RAG retrieval error: {e}")
            debug_print(f"RAG retrieval exception: {type(e).__name__}: {e}")
            debug_print(f"Full traceback:\n{error_trace}")
    else:
        debug_print(f"Skipping RAG retrieval: rag={rag is not None}, query_text={bool(query_text)}, should_retrieve={intent.should_retrieve}")

    debug_print("-" * 60)
    debug_print("MESSAGE CONSTRUCTION PHASE")

    instructions = f"\n\nCRITICAL INSTRUCTIONS:\n" \
                   f"1. PREFER CONTEXT: Answer based primarily on the provided context below.\n" \
                   f"2. BE ACCURATE: Do not make up facts. You may use general knowledge to supplement context if needed.\n" \
                   f"3. VERIFY PREMISES: If the user asks a leading question (e.g., 'When did X do Y?') and the context says X *never* did Y, you MUST correct the premise.\n" \
                   f"4. HANDLE CONFLICTS: If context has conflicting info, state BOTH sides clearly.\n" \
                   f"5. SYNTHESIZE: Combine the context with your knowledge to provide a complete, accurate answer.\n" \
                   f"6. ANSWER THE QUESTION DIRECTLY: If asked 'where was X born?', answer with a LOCATION. If asked 'when?', answer with a DATE. Do not provide tangential information.\n" \
                   f"7. FOR COMPARISONS: You MUST discuss BOTH entities being compared, not just one.\n" \
                   f"8. BE HELPFUL: If the context is partial or the match is not perfect, try to infer the answer or use related information to help the user instead of simply refusing.\n" \
                   f"9. NATURAL TONE: You are an expert. Do NOT mention 'RAG', 'context', 'retrieved documents', or 'knowledge base' in your final answer. Integrate the information naturally as if you already knew it."

    if results:
        search_ctx = results[0].get('search_context', {})
        answer_type = search_ctx.get('answer_type')
        if answer_type and answer_type not in ['general', 'information', 'unknown', None]:
            debug_print(f"Injecting prompt reinforcement for answer_type: '{answer_type}'")
            instructions += f"\n8. **ANSWER THE SPECIFIC QUESTION**: The user specifically asked for **{answer_type}** information. You MUST include this specific detail (date, location, name, etc.) in your answer. Do not just summarize the entity's biography."

    final_system_prompt = system_prompt + intent.system_instruction + instructions

    debug_print(f"Base system_prompt + intent instruction + instructions = {len(final_system_prompt)} chars")

    if context_text:
        final_system_prompt += "\n\n" + context_text
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
