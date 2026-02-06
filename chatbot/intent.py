
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
from chatbot import config

def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:INTENT] {msg}", file=sys.stderr)

@dataclass
class IntentResult:
    should_retrieve: bool
    system_instruction: str
    mode_name: str


@dataclass
class QueryComplexity:
    """Classification of query complexity for pipeline optimization."""
    level: str  # "simple", "moderate", "complex"
    skip_multi_hop: bool  # Skip multi-hop resolver
    skip_coverage: bool   # Skip coverage verifier
    max_steps: int        # Reduced orchestration steps for simple queries


def classify_query_complexity(query: str) -> QueryComplexity:
    """
    Heuristic classifier to determine query complexity.
    Simple queries can skip expensive joints to reduce latency.

    Returns:
        QueryComplexity with routing hints for the orchestrator
    """
    debug_print(f"Classifying complexity for: '{query}'")
    q_lower = query.lower().strip()

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

    # 4. FACTUAL / DEFAULT (The "Rest")
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