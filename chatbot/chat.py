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
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from chatbot.models import Message
from chatbot import config
from chatbot.model_manager import ModelManager
from chatbot.runtime_profiles import record_generation_metrics, get_model_lane_stats


def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:CHAT] {msg}", file=sys.stderr)


# Global status callback for UI updates
_status_callback = None
_runtime_checkpoint: Optional[Dict[str, Any]] = None
_rag_system = None


def _project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def _runtime_checkpoint_path() -> str:
    data_dir = os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "runtime_checkpoint.json")


def _default_runtime_checkpoint() -> Dict[str, Any]:
    return {
        "contract_version": "contracted_cognition.v1",
        "objective": "",
        "frontier": [],
        "artifacts": [],
        "constraints": [],
        "open_loops": [],
        "risk": "unknown",
        "next_step": "",
        "drop_list": [],
        "query": "",
        "updated_at": 0,
        "source": {},
    }


def load_runtime_checkpoint() -> Dict[str, Any]:
    global _runtime_checkpoint
    if _runtime_checkpoint is not None:
        return dict(_runtime_checkpoint)

    path = _runtime_checkpoint_path()
    if not os.path.exists(path):
        _runtime_checkpoint = _default_runtime_checkpoint()
        return dict(_runtime_checkpoint)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            merged = _default_runtime_checkpoint()
            merged.update(payload)
            _runtime_checkpoint = merged
            return dict(_runtime_checkpoint)
    except Exception:
        pass

    _runtime_checkpoint = _default_runtime_checkpoint()
    return dict(_runtime_checkpoint)


def save_runtime_checkpoint(payload: Dict[str, Any]) -> None:
    global _runtime_checkpoint
    merged = _default_runtime_checkpoint()
    merged.update(payload or {})
    merged["updated_at"] = int(time.time())

    path = _runtime_checkpoint_path()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    _runtime_checkpoint = merged


def clear_runtime_checkpoint() -> None:
    global _runtime_checkpoint
    _runtime_checkpoint = _default_runtime_checkpoint()
    path = _runtime_checkpoint_path()
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _trim_text(value: str, limit: int = 280) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_preserve_order(items: List[str], limit: int) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        text = _trim_text(str(item), 180)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _artifact_summary(snapshot: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []

    for artifact in snapshot.get("artifacts", [])[-4:]:
        if not isinstance(artifact, dict):
            continue
        entry = {
            "type": artifact.get("mode", "artifact"),
            "step": artifact.get("step", ""),
            "query": _trim_text(str(artifact.get("query", "")), 120),
            "status": artifact.get("status", "ok"),
            "note": _trim_text(str(artifact.get("note", "")), 140),
        }
        artifacts.append(entry)

    if not artifacts:
        for result in results[:3]:
            meta = result.get("metadata", {}) if isinstance(result, dict) else {}
            title = _trim_text(str(meta.get("title", "")), 120)
            if title:
                artifacts.append({
                    "type": "source",
                    "step": "search",
                    "query": title,
                    "status": "ok",
                    "note": "retrieved reference",
                })

    return artifacts[:4]


def capture_runtime_checkpoint(
    query_text: str,
    history: List[Message],
    results: List[Dict[str, Any]],
    rag_snapshot: Optional[Dict[str, Any]] = None,
    rag_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = rag_snapshot or {}
    status = rag_status or {}
    previous = load_runtime_checkpoint()

    objective = (
        snapshot.get("active_goal")
        or status.get("active_goal")
        or previous.get("objective")
        or "answer_user_query"
    )

    base_mind = snapshot.get("base_mind", {}) if isinstance(snapshot.get("base_mind"), dict) else {}
    frontier = base_mind.get("frontier") or snapshot.get("objective_frontier_risk", {}).get("frontier") or []
    if not frontier and results:
        frontier = ["synthesize_answer"]

    open_loops = list(previous.get("open_loops", []))
    if not results:
        open_loops.insert(0, "No strong retrieval hits; answer may rely on general knowledge.")
    if status.get("contract_ok") is False:
        open_loops.insert(0, f"Observability contract issues: {', '.join(status.get('contract_issues', []))}")

    constraints = [
        "Prefer verified source facts when present.",
        "Preserve only compact residue across resets.",
    ]
    if results:
        constraints.append("Do not drop retrieved artifacts that support the current answer.")

    recent_user = [msg.content for msg in history if msg.role == "user"][-2:]
    recent_assistant = [msg.content for msg in history if msg.role == "assistant"][-1:]
    if recent_assistant:
        open_loops.append(f"Last assistant output: {_trim_text(recent_assistant[-1], 180)}")

    next_step = "synthesize_answer"
    if not results:
        next_step = "answer_with_care"
    elif frontier:
        next_step = str(frontier[0])

    payload = {
        "contract_version": "contracted_cognition.v1",
        "objective": str(objective),
        "frontier": _dedupe_preserve_order([str(item) for item in frontier], 4),
        "artifacts": _artifact_summary(snapshot, results),
        "constraints": _dedupe_preserve_order(constraints, 5),
        "open_loops": _dedupe_preserve_order(open_loops + recent_user, 5),
        "risk": str(status.get("ofr_risk") or base_mind.get("risk") or previous.get("risk") or "unknown"),
        "next_step": next_step,
        "drop_list": [
            "Discard transient chain-of-thought and stale scratch state.",
            "Keep only compact residue plus typed artifacts.",
        ],
        "query": _trim_text(query_text, 220),
        "source": {
            "routing_mode": status.get("mode") or snapshot.get("routing_mode") or "local_only",
            "routing_reason": status.get("routing_reason") or snapshot.get("routing_reason") or "",
            "result_count": len(results),
        },
    }
    save_runtime_checkpoint(payload)
    return payload


def _format_runtime_checkpoint_for_prompt(checkpoint: Dict[str, Any]) -> str:
    if not checkpoint:
        return ""
    if not checkpoint.get("objective") and not checkpoint.get("artifacts"):
        return ""

    lines = [
        "\n\n=== CONTRACTED COGNITION HANDOFF ===",
        f"Contract: {checkpoint.get('contract_version', 'contracted_cognition.v1')}",
        f"Objective: {checkpoint.get('objective', '')}",
        f"Risk: {checkpoint.get('risk', 'unknown')}",
        f"Next Step: {checkpoint.get('next_step', '')}",
    ]

    frontier = checkpoint.get("frontier", []) or []
    if frontier:
        lines.append("Frontier:")
        for item in frontier[:4]:
            lines.append(f"- {item}")

    artifacts = checkpoint.get("artifacts", []) or []
    if artifacts:
        lines.append("Artifacts:")
        for item in artifacts[:4]:
            if not isinstance(item, dict):
                continue
            label = item.get("query") or item.get("note") or item.get("step") or "artifact"
            note = item.get("note", "")
            lines.append(f"- {item.get('type', 'artifact')}::{label} [{item.get('status', 'ok')}] {note}".strip())

    open_loops = checkpoint.get("open_loops", []) or []
    if open_loops:
        lines.append("Open Loops:")
        for item in open_loops[:4]:
            lines.append(f"- {item}")

    constraints = checkpoint.get("constraints", []) or []
    if constraints:
        lines.append("Persistence Rules:")
        for item in constraints[:4]:
            lines.append(f"- {item}")

    lines.append("Use this handoff to reconstruct task-relevant state after resets. Preserve residue and typed artifacts; discard transient scratch reasoning.")
    lines.append("====================================")
    return "\n".join(lines)


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


def clear_runtime_memory(reset_rag: bool = False, preserve_checkpoint: bool = True) -> None:
    """Clear in-process session caches while optionally preserving contracted residue."""
    try:
        from chatbot.rag.search import clear_title_cache
        clear_title_cache()
    except Exception:
        pass

    if not preserve_checkpoint:
        clear_runtime_checkpoint()

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
                        end_idx = buffer.find('</think>')
                        if end_idx == -1:
                            buffer = ""
                            break
                        buffer = buffer[end_idx + len('</think>'):]
                        in_thought_block = False
                        continue

                    start_idx = buffer.find('<think>')
                    if start_idx == -1:
                        if buffer:
                            yielded_chars += len(buffer)
                            yield buffer
                            buffer = ""
                        break

                    visible = buffer[:start_idx]
                    if visible:
                        yielded_chars += len(visible)
                        yield visible
                    buffer = buffer[start_idx + len('<think>'):]
                    in_thought_block = True

        if buffer and not in_thought_block:
            yielded_chars += len(buffer)
            yield buffer

        duration = time.time() - started_at
        first_token_seconds = (first_token_at - started_at) if first_token_at is not None else None
        record_generation_metrics(
            model=model,
            lane=str(profile["lane"]),
            prompt_chars=int(profile.get("prompt_chars", 0)),
            output_chars=yielded_chars,
            duration_seconds=duration,
            first_token_seconds=first_token_seconds,
            metadata={
                "answer_type": profile.get("answer_type"),
                "fact_count": profile.get("fact_count"),
            },
        )
    except Exception as e:
        debug_print(f"stream_chat error: {type(e).__name__}: {e}")
        raise


def chat(model: str, messages: List[dict]) -> str:
    """Non-streaming chat with local model."""
    debug_print(f"chat called with model='{model}'")

    try:
        profile = _choose_generation_lane(model, messages)
        prepared_messages = _prepare_messages_for_generation(messages, profile)
        n_ctx = int(profile["n_ctx"])
        max_tokens = int(profile["max_tokens"])
        llm = ModelManager.get_model(model, n_ctx=n_ctx)

        _update_status(f"Generating response ({profile['lane']})...")
        started_at = time.time()
        response = llm.create_chat_completion(
            messages=prepared_messages,
            temperature=0.3,
            repeat_penalty=1.2,
            max_tokens=max_tokens,
        )
        output = response['choices'][0]['message']['content']
        duration = time.time() - started_at
        record_generation_metrics(
            model=model,
            lane=str(profile["lane"]),
            prompt_chars=int(profile.get("prompt_chars", 0)),
            output_chars=len(output),
            duration_seconds=duration,
            metadata={
                "answer_type": profile.get("answer_type"),
                "fact_count": profile.get("fact_count"),
            },
        )
        return output
    except Exception as e:
        debug_print(f"chat error: {type(e).__name__}: {e}")
        raise


def full_chat(model: str, messages: List[dict]) -> str:
    """Compatibility wrapper for legacy callers expecting full_chat."""
    return chat(model, messages)


def get_rag_system():
    """Lazy-load and cache the RAG system."""
    global _rag_system
    if _rag_system is None:
        from chatbot.rag import RAGSystem
        _rag_system = RAGSystem()
    return _rag_system


def retrieve_and_display_links(query: str) -> List[dict]:
    """Retrieve links only for link-mode displays."""
    debug_print(f"retrieve_and_display_links called with query='{query}'")

    try:
        rag = get_rag_system()
        results = rag.retrieve(query, top_k=getattr(config, 'MAX_LINK_RESULTS', 10))
        links: List[dict] = []
        seen_titles = set()

        for result in results:
            metadata = result.get('metadata', {})
            title = metadata.get('title', '')
            if not title or title in seen_titles:
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


def _artifact_id_for_result(index: int, result: Dict[str, Any]) -> str:
    metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
    title = str(metadata.get('title', 'unknown')).strip().replace(' ', '_')
    chunk_id = str(metadata.get('chunk_id', index)).strip() or str(index)
    return f"A{index}:{title}#{chunk_id}"


def _build_grounding_manifest(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    lines = ["\n\n=== GROUNDING SOURCE MANIFEST ==="]
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'Unknown')
        path = metadata.get('path', '')
        score = result.get('score', 0.0)
        artifact_id = _artifact_id_for_result(i, result)
        lines.append(f"- [{artifact_id}] title={title} score={score:.3f} path={path}")
    lines.append("================================")
    return "\n".join(lines)


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
                    artifact_id = _artifact_id_for_result(i, r)

                    if len(text) > 4000:
                        text = text[:4000] + "...(truncated)"

                    chunk_text = f"\n--- Source {i} [{artifact_id}]: {title} ---\n{text}\n"

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

                if getattr(config, 'GROUNDING_MANIFEST_ENABLED', True):
                    context_text += _build_grounding_manifest(results)

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

    checkpoint = capture_runtime_checkpoint(
        query_text=query_text or "",
        history=history,
        results=results,
        rag_snapshot=getattr(rag, 'last_orchestration_snapshot', {}) if rag else {},
        rag_status=getattr(rag, 'last_orchestration_status', {}) if rag else {},
    )

    debug_print("-" * 60)
    debug_print("MESSAGE CONSTRUCTION PHASE")

    instructions = f"\n\nCRITICAL INSTRUCTIONS:\n" \
                   f"1. PREFER CONTEXT: Answer based primarily on the provided context below.\n" \
                   f"2. BE ACCURATE: Do not make up facts. You may use general knowledge to supplement context only when grounding mode allows it.\n" \
                   f"3. VERIFY PREMISES: If the user asks a leading question (e.g., 'When did X do Y?') and the context says X *never* did Y, you MUST correct the premise.\n" \
                   f"4. HANDLE CONFLICTS: If context has conflicting info, state BOTH sides clearly.\n" \
                   f"5. SYNTHESIZE: Combine the context with your knowledge to provide a complete, accurate answer.\n" \
                   f"6. ANSWER THE QUESTION DIRECTLY: If asked 'where was X born?', answer with a LOCATION. If asked 'when?', answer with a DATE. Do not provide tangential information.\n" \
                   f"7. FOR COMPARISONS: You MUST discuss BOTH entities being compared, not just one.\n" \
                   f"8. BE HELPFUL: If the context is partial or the match is not perfect, try to infer the answer or use related information to help the user instead of simply refusing.\n" \
                   f"9. NATURAL TONE: You are an expert. Do NOT mention 'RAG', 'context', 'retrieved documents', or 'knowledge base' in your final answer. Integrate the information naturally as if you already knew it.\n" \
                   f"10. CONTRACTED COGNITION: Use the handoff block to reconstruct task-relevant state after resets. Preserve typed artifacts and compact residue; discard transient scratch reasoning."

    if getattr(config, 'GROUNDED_FACT_GATE', True):
        instructions += (
            "\n11. GROUNDED FACT GATE: For factual claims, cite at least one artifact ID from the GROUNDING SOURCE MANIFEST "
            "using bracket form like [A1:Title#chunk]."
            "\n12. NO UNSOURCED FACTS: If you cannot support a required fact with at least one artifact ID, do not guess. "
            "Reply exactly: FAIL: insufficient grounded evidence for one or more required facts."
            "\n13. MODEL-AGNOSTIC CONTRACT: This citation rule applies regardless of model size or tier."
        )

    if results:
        search_ctx = results[0].get('search_context', {})
        answer_type = search_ctx.get('answer_type')
        if answer_type and answer_type not in ['general', 'information', 'unknown', None]:
            debug_print(f"Injecting prompt reinforcement for answer_type: '{answer_type}'")
            instructions += f"\n8. **ANSWER THE SPECIFIC QUESTION**: The user specifically asked for **{answer_type}** information. You MUST include this specific detail (date, location, name, etc.) in your answer. Do not just summarize the entity's biography."

    final_system_prompt = system_prompt + _format_runtime_checkpoint_for_prompt(checkpoint) + intent.system_instruction + instructions

    debug_print(f"Base system_prompt + checkpoint + intent instruction + instructions = {len(final_system_prompt)} chars")

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


def build_messages_with_intent(system_prompt: str, history: List[Message], user_query: str = None):
    """Build messages and also return the IntentResult for teleport handling.

    Returns:
        (messages, intent) — the constructed message list and the raw IntentResult
        so callers can inspect shell_intent / teleport_envelope before generating.
    """
    from chatbot.intent import detect_intent

    query_text = user_query
    if not query_text and history and history[-1].role == 'user':
        query_text = history[-1].content

    intent = detect_intent(query_text or "")
    messages = build_messages(system_prompt, history, user_query)
    return messages, intent
