
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

import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
from chatbot import config

def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:INTENT] {msg}", file=sys.stderr)

@dataclass
class IntentResult:
    should_retrieve: bool
    system_instruction: str
    mode_name: str
    shell_intent: Optional[str] = None  # Set if shell operation detected
    teleport_envelope: Optional[object] = None  # TeleportEnvelope if shell intent


@dataclass
class QueryComplexity:
    """Classification of query complexity for pipeline optimization."""
    level: str  # "simple", "moderate", "complex"
    skip_multi_hop: bool  # Skip multi-hop resolver
    skip_coverage: bool   # Skip coverage verifier
    max_steps: int        # Reduced orchestration steps for simple queries


# Indirect-reference patterns that require multi-hop resolution.
# Matched against the lowercased query before other heuristics.
_INDIRECT_REF_PATTERNS = (
    r"\b(creator|inventor|founder|author|developer|designer|maker)\s+of\b",
    r"\b(capital|leader|president|ceo|director)\s+of\b",
    r"\b(parent|spouse|wife|husband|child|daughter|son)\s+of\b",
)


def classify_query_complexity(query: str) -> QueryComplexity:
    """
    Heuristic classifier to determine query complexity.
    Simple queries can skip expensive joints to reduce latency.

    Returns:
        QueryComplexity with routing hints for the orchestrator
    """
    debug_print(f"Classifying complexity for: '{query}'")
    q_lower = query.lower().strip()

    # COMPLEX: Indirect-reference patterns (creator/founder/spouse of X, etc.)
    # These always require multi-hop resolution; check before entity/word counts.
    for pattern in _INDIRECT_REF_PATTERNS:
        if re.search(pattern, q_lower):
            debug_print(f"COMPLEX query (indirect reference pattern: {pattern!r})")
            return QueryComplexity(
                level="complex",
                skip_multi_hop=False,
                skip_coverage=False,
                max_steps=10,
            )

    # Count entities (capitalized words, quoted strings)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    quoted = re.findall(r'"[^"]*"', query)
    entity_count = len(entities) + len(quoted)

    # Word count
    words = q_lower.split()
    word_count = len(words)

    # SIMPLE: "What is X?" patterns, single entity queries
    simple_patterns = [
        r'^what is\b',
        r'^what are\b',
        r'^who is\b',
        r'^who was\b',
        r'^define\b',
        r'^meaning of\b',
        r'^when was\b.*born',
        r'^when did\b.*die',
    ]

    for pattern in simple_patterns:
        if re.search(pattern, q_lower):
            if entity_count <= 1 and word_count <= 8:
                debug_print(f"SIMPLE query (pattern match + short)")
                return QueryComplexity(
                    level="simple",
                    skip_multi_hop=True,
                    skip_coverage=True,
                    max_steps=5
                )

    # COMPLEX: Comparisons, multi-entity, long queries
    complex_indicators = [
        r'\bcompare\b',
        r'\bversus\b',
        r'\bvs\.?\b',
        r'\bdifference between\b',
        r'\brelationship between\b',
        r'\bhow does .* affect\b',
        r'\bwhy did .* cause\b',
    ]

    for pattern in complex_indicators:
        if re.search(pattern, q_lower):
            debug_print(f"COMPLEX query (comparison/relationship pattern)")
            return QueryComplexity(
                level="complex",
                skip_multi_hop=False,
                skip_coverage=False,
                max_steps=10
            )

    # Multiple entities = moderate to complex
    if entity_count >= 3:
        debug_print(f"COMPLEX query ({entity_count} entities)")
        return QueryComplexity(
            level="complex",
            skip_multi_hop=False,
            skip_coverage=False,
            max_steps=10
        )

    if entity_count == 2:
        debug_print(f"MODERATE query (2 entities)")
        return QueryComplexity(
            level="moderate",
            skip_multi_hop=False,
            skip_coverage=True,  # Skip coverage for moderate
            max_steps=7
        )

    # Long queries are often more complex
    if word_count > 15:
        debug_print(f"MODERATE query (long: {word_count} words)")
        return QueryComplexity(
            level="moderate",
            skip_multi_hop=False,
            skip_coverage=True,
            max_steps=7
        )

    # Default: simple
    debug_print(f"SIMPLE query (default)")
    return QueryComplexity(
        level="simple",
        skip_multi_hop=True,
        skip_coverage=True,
        max_steps=5
    )

def detect_intent(query: str) -> IntentResult:
    """
    Analyze the user query to determine intent and operational mode.
    
    Returns:
        IntentResult containing:
        - should_retrieve: Whether to perform RAG search
        - system_instruction: Specific prompt instructions for this mode
        - mode_name: Human readable mode name (for debug)
    """
    debug_print(f"Starting intent detection for query: '{query}'")
    q_lower = query.lower().strip()
    debug_print(f"Normalized query: '{q_lower}'")
    
    # 1. CONVERSATION / CHIT-CHAT
    # Trigger: Greetings, phatic expressions
    debug_print("Testing CONVERSATION patterns...")
    conversation_triggers = [
        r"^hello", r"^hi\b", r"^hey\b", r"^greetings",
        r"^how are you", r"^what'?s up", r"^good (morning|afternoon|evening|night)",
        r"^thanks", r"^thank you"
    ]
    
    for pattern in conversation_triggers:
        if re.search(pattern, q_lower):
            debug_print(f"MATCH: Pattern '{pattern}' matched -> CONVERSATION mode")
            debug_print("Decision: should_retrieve=False (casual conversation)")
            return IntentResult(
                should_retrieve=False,
                system_instruction="The user is engaging in casual conversation. Be friendly, polite, and concise. Do not try to look up facts unless explicitly asked.",
                mode_name="CONVERSATION"
            )
    debug_print("No CONVERSATION pattern matched")

    # 2. TUTORIAL / INSTRUCTIONAL
    # Trigger: "How to", "Guide for", "Steps to"
    debug_print("Testing TUTORIAL patterns...")
    if re.search(r"^(how to|guide|tutorial|steps|way to)", q_lower):
        debug_print("MATCH: TUTORIAL pattern matched")
        debug_print("Decision: should_retrieve=True (tutorial mode)")
        # Note: We MIGHT want RAG for tutorials if the user asks "How to fix a flat tire",
        # but the user requested "Intent model layer needs to know when to not index".
        # For a pure "How to write a poem" or "How to be happy", RAG is noise.
        # But "How to install Linux" might need RAG.
        # Strategy: If it looks like a general skill, skip RAG. If it looks like a Specific Entity query, use RAG.
        # For simplicity in this prototype: Tutorials enable RAG (knowledge is helpful) but change the Style.
        
        # Strategy: If it looks like a general skill, skip RAG. If it looks like a Specific Entity query, use RAG.
        # For simplicity in this prototype: Tutorials enable RAG (knowledge is helpful) but change the Style.
        
        return IntentResult(
            should_retrieve=True,
            system_instruction=(
                "\nMODE: TUTORIAL\n"
                "OBJECTIVE: Provide a structured, easy-to-follow guide.\n"
                "LAYOUT RULES:\n"
                "1. TITLE: Start with a clear H1 title (e.g. '# How to...')\n"
                "2. OVERVIEW: Brief summary (1-2 sentences).\n"
                "3. STEPS: Use numbered lists for actions (1. **Do this**...).\n"
                "4. FORMATTING: Use **bold** for key terms/buttons. Use `code blocks` for commands.\n"
                "5. TONE: Helpful, instructional, and encouraging."
            ),
            mode_name="TUTORIAL"
        )
        
    debug_print("Testing DEBATE patterns...")
    # 3. DEBATE / OPINION (Experimental)
    if re.search(r"^(argue|debate|pros and cons|opinion on)", q_lower):
        debug_print("MATCH: DEBATE pattern matched")
        debug_print("Decision: should_retrieve=True (debate mode)")
        return IntentResult(
            should_retrieve=True,
            system_instruction="\nMODE: DEBATE\nPresent multiple viewpoints. Analyze pros and cons objectively. Use bullet points to contrast arguments.",
            mode_name="DEBATE"
        )

    # 4. SHELL / TELEPORT (Shell Chamber Operations)
    # Trigger: File writes, script creation, shell commands
    debug_print("Testing SHELL patterns...")
    from chatbot.teleport import build_shell_envelope, wave_mode_enabled
    
    envelope = build_shell_envelope(query)
    if envelope is not None:
        shell_intent = envelope.intent
        debug_print(f"MATCH: SHELL pattern matched -> intent={shell_intent}")
        
        # Check if this is a refused teleport (wave mode required)
        if envelope.constraints.get("refused"):
            debug_print("SHELL intent but wave mode not enabled -> providing guidance")
            return IntentResult(
                should_retrieve=False,
                system_instruction=(
                    "\nMODE: SHELL TELEPORT REQUIRED\n"
                    "The user is requesting a file/shell operation that requires the Shell Chamber.\n"
                    "IMPORTANT: This requires Wave mode. Do NOT provide a text-only approximation.\n"
                    "Instead, explain that you need Wave mode to safely execute this operation.\n"
                    "To enable: type 'mode wave' in the CLI, or set RUNTIME_MODE=wave in config.\n"
                    "This is a safety boundary - do not fake execution with text output."
                ),
                mode_name="SHELL_BLOCKED",
                shell_intent=shell_intent,
                teleport_envelope=envelope,
            )
        
        # Wave mode is active - prepare for teleport
        debug_print(f"Decision: should_retrieve=False (shell mode - will teleport)")
        
        # Determine instruction based on intent type
        agent_protocol = (
            "\nAGENT PROTOCOL:\n"
            "- Your shell commands are being executed LIVE in a real terminal.\n"
            "- You will receive [SHELL CHAMBER OBSERVATION] blocks with real command output.\n"
            "- To run a follow-up command, output it inside [HERMIT_CMD]command[/HERMIT_CMD] tags.\n"
            "- You can chain multiple commands across rounds (max 5).\n"
            "- Summarize results in plain language for the user.\n"
            "- Do NOT simulate or fake output. Real execution is happening.\n"
        )
        if shell_intent == "file_write":
            mode_instruction = (
                "\nMODE: SHELL TELEPORT - FILE WRITE\n"
                "OBJECTIVE: Write a file via the Shell Chamber.\n"
                "INSTRUCTIONS:\n"
                "1. Generate the file content the user requested.\n"
                "2. The Shell Chamber will write it to disk.\n"
                "3. Confirm the operation completed successfully.\n"
                + agent_protocol
            )
        elif shell_intent == "shell_command":
            mode_instruction = (
                "\nMODE: SHELL TELEPORT - COMMAND\n"
                "OBJECTIVE: Execute a shell command via the Shell Chamber.\n"
                "INSTRUCTIONS:\n"
                "1. The command will execute in a real terminal with timeout limits.\n"
                "2. You will see the real output in a [SHELL CHAMBER OBSERVATION] block.\n"
                "3. Summarize the results clearly for the user.\n"
                "4. If you need follow-up commands, use [HERMIT_CMD]command[/HERMIT_CMD].\n"
                + agent_protocol
            )
        elif shell_intent == "script_create":
            mode_instruction = (
                "\nMODE: SHELL TELEPORT - SCRIPT CREATE\n"
                "OBJECTIVE: Create an executable script via the Shell Chamber.\n"
                "INSTRUCTIONS:\n"
                "1. Generate appropriate script content with proper scaffolding.\n"
                "2. The script will be saved with executable permissions.\n"
                "3. After creation, confirm and provide usage instructions.\n"
                + agent_protocol
            )
        else:
            mode_instruction = (
                "\nMODE: SHELL TELEPORT\n"
                "OBJECTIVE: Execute a shell operation via the Shell Chamber.\n"
                "The chamber provides safe execution with proper boundaries.\n"
                + agent_protocol
            )
        
        return IntentResult(
            should_retrieve=False,
            system_instruction=mode_instruction,
            mode_name="SHELL",
            shell_intent=shell_intent,
            teleport_envelope=envelope,
        )
    debug_print("No SHELL pattern matched")

    # 5. FACTUAL / DEFAULT (The "Rest")
    # Trigger: "What is", "Who is", "When", "Define", or any specific query
    debug_print("No specific pattern matched -> FACTUAL (default) mode")
    debug_print("Decision: should_retrieve=True (factual mode)")
    return IntentResult(
        should_retrieve=True,
        system_instruction=(
            "\nMODE: FACTUAL REASONING\n"
            "OBJECTIVE: Provide a single, concise answer. If sources conflict, report the conflict neutrally.\n"
            "LAYOUT RULES:\n"
            "1. DIRECT ANSWER: Start directly with the answer.\n"
            "2. REPORT CONFLICTS: If sources disagree (e.g. Source A says X, Source B says Y), state BOTH. Do NOT try to resolve it.\n"
            "3. NO ARGUMENTATION: Do not debate with yourself (e.g. 'But wait...', 'However...'). just list the facts.\n"
            "4. NO REPETITION: State the facts once and STOP. Do NOT summarize at the end."
        ),
        mode_name="FACTUAL"
    )

def has_shell_intent(query: str) -> Tuple[bool, Optional[str]]:
    """Quick check if query has shell intent without full intent detection.
    
    Args:
        query: User query string
        
    Returns:
        (has_shell, intent_type) - True if shell intent detected, with the intent type
    """
    from chatbot.teleport import classify_shell_intent, ShellIntent
    
    intent = classify_shell_intent(query)
    if intent != ShellIntent.UNKNOWN:
        return True, intent.value
    return False, None


def check_teleport_available(query: str) -> Tuple[bool, str]:
    """Check if teleport is available for this query.
    
    Args:
        query: User query string
        
    Returns:
        (available, message) - Whether teleport is available and why/why not
    """
    has_shell, intent_type = has_shell_intent(query)
    
    if not has_shell:
        return True, "No shell intent detected - normal chat flow"
    
    from chatbot.teleport import wave_mode_enabled, require_wave_mode
    
    is_wave, msg = require_wave_mode()
    if is_wave:
        return True, f"Wave mode active - {intent_type} teleport available"
    
    return False, msg
