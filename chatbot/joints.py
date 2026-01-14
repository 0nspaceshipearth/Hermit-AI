
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

"""
Multi-Joint RAG System - Reasoning Joints for Improved Retrieval

This module implements three reasoning joints that use small LLMs to guide
the retrieval process and prevent hallucinations:

1. EntityExtractorJoint - Extracts entities and aliases from queries
2. ArticleScorerJoint - Scores article relevance to entities
3. ChunkFilterJoint - Filters chunks by query relevance
"""

import sys
import json
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from chatbot import config
from chatbot.model_manager import ModelManager
from chatbot.grammar_utils import get_json_grammar, get_object_grammar, get_array_grammar


def debug_print(joint_name: str, msg: str):
    """Print debug message for a specific joint."""
    if config.DEBUG:
        print(f"[DEBUG:{joint_name}] {msg}", file=sys.stderr)


def extract_json_from_text(text: str) -> Any:
    """
    Robustly extract the first valid JSON object or array from text.
    Handles Markdown code blocks, conversational filler, and nested structures.
    """
    if not text:
        raise ValueError("Empty text input")

    # 1. Try to find JSON in Markdown code blocks first
    code_block_pattern = r'```(?:json)?\s*(?P<content>.*?)\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        content = match.group('content')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass # Fallback to scanning raw text if block content is invalid

    # 2. Heuristic scan for looking for first '{' or '['
    # We use a simple counter to find the matching closing brace/bracket
    
    start_indices = [m.start() for m in re.finditer(r'[\[\{]', text)]
    
    for start_idx in start_indices:
        opener = text[start_idx]
        closer = '}' if opener == '{' else ']'
        
        stack = 1
        for i in range(start_idx + 1, len(text)):
            char = text[i]
            if char == opener:
                stack += 1
            elif char == closer:
                stack -= 1
            
            if stack == 0:
                potential_json = text[start_idx : i + 1]
                try:
                    # Soft Fix: Try to clean trailing commas which are common in LLM output
                    # Simple regex replace: , ] -> ] and , } -> }
                    # This is risky but effective for simple lists
                    clean_json = re.sub(r',\s*([\]\}])', r'\1', potential_json)
                    return json.loads(clean_json)
                except json.JSONDecodeError:
                    # Try original
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        break # Try next starting point
                        
    raise ValueError("No valid JSON found in response")


def local_inference(model: str, prompt: str, temperature: float = 0.0, timeout: int = 5, use_json_grammar: bool = False) -> str:
    """
    Run local inference using ModelManager.
    
    Args:
        model: Model repo ID
        prompt: The prompt to send
        temperature: Sampling temperature
        timeout: Request timeout (not used for local, kept for compat)
        use_json_grammar: If True, enforce valid JSON output using GBNF grammar.
                          This prevents conversational filler and invalid JSON.
    
    Returns:
        The model's response text.
    """
    try:
        # Use configured context size (default 16384)
        n_ctx = getattr(config, 'DEFAULT_CONTEXT_SIZE', 16384)
        
        # Truncate prompt if it exceeds context limit (leaving room for response)
        # 1 token approx 4 chars. Safety buffer: 1000 tokens for generation/system overhead.
        max_prompt_chars = (n_ctx - 1000) * 3  # Conservative estimate
        if len(prompt) > max_prompt_chars:
            debug_print("INFERENCE", f"Truncating prompt from {len(prompt)} to {max_prompt_chars} chars")
            prompt = prompt[:max_prompt_chars] + "...(truncated)"

        llm = ModelManager.get_model(model, n_ctx=n_ctx)  # Shared instance
        
        # Build completion kwargs
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        # Add grammar constraint if requested
        if use_json_grammar:
            grammar = get_json_grammar()
            if grammar:
                completion_kwargs["grammar"] = grammar
                debug_print("INFERENCE", "Using GBNF JSON grammar constraint")
            else:
                debug_print("INFERENCE", "JSON grammar unavailable, falling back to unconstrained")
        
        # Use chat completion for instruction-tuned models
        response = llm.create_chat_completion(**completion_kwargs)
        
        return response['choices'][0]['message']['content']
    except Exception as e:
        debug_print("INFERENCE", f"Model {model} failed: {e}")
        # Soft fallback: Try one more time with shorter context if it was a context error
        if "context window" in str(e) or "exceed" in str(e):
             debug_print("INFERENCE", "Retrying with aggressive truncation...")
             try:
                 prompt = prompt[:4000] + "...(truncated)"
                 llm = ModelManager.get_model(model, n_ctx=n_ctx)
                 # Retry without grammar to maximize chances of success
                 response = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
                 )
                 return response['choices'][0]['message']['content']
             except Exception as retry_e:
                 raise RuntimeError(f"Local inference failed (retry failed): {retry_e}")
        raise RuntimeError(f"Local inference failed: {e}")


class EntityExtractorJoint:
    """
    Joint 1: Entity Extraction
    
    Extracts the main entity, type, action, and aliases from a user query.
    Uses llama3.2:1b for fast, focused entity recognition.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.ENTITY_JOINT_MODEL
        self.temperature = config.ENTITY_JOINT_TEMP
        debug_print("JOINT1:INIT", f"EntityExtractor initialized with {self.model}")
    
    def extract(self, query: str) -> Dict[str, any]:
        """
        Extract ALL entities from query, with comparison detection.
        
        Args:
            query: User query string
            
        Returns:
            Dict with keys: is_comparison, entities (list), action
            Each entity has: name, type, aliases
        """
        debug_print("JOINT1:ENTITY", f"Extracting entities from: '{query}'")
        start_time = time.time()
        
        prompt = f"""You are a precise entity extraction system optimized for Wikipedia article matching.

INSTRUCTIONS:
1. Identify ALL distinct entities (people, places, things, events) in the query.
2. For each entity, provide the NAME as it would appear as a Wikipedia article title.
3. CHECK FOR COMPARISONS: If the user compares items (e.g. "vs", "compare", "difference"), set "is_comparison": true.
4. EXTRACT ALIASES: Include alternative names AND related Wikipedia article titles the entity might appear under.

WIKIPEDIA TITLE CONVENTIONS:
- Full names for people: "Albert Einstein" not "Einstein"
- Specific names for events: "World War II" not "the war"
- Disambiguation when needed: "Java (programming language)" for the language
- For indirect queries (e.g., "who created X"), extract BOTH the person AND the thing created

Query: "{query}"

CRITICAL RULES:
- Return ONLY valid JSON.
- NO Markdown code blocks.
- Entity names should match Wikipedia article titles exactly when possible.
- Include short aliases that might also be article titles.

Return this exact JSON structure:
{{
  "is_comparison": false,
  "entities": [
    {{"name": "Exact Wikipedia Article Title", "type": "person|place|event|concept|technology|organization", "aliases": ["Alternative Title", "Short Form"]}}
  ],
  "action": "what the user wants to know"
}}
"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT1:ENTITY", f"Raw response: {response[:300]}...")
            
            # Use robust extractor
            result = extract_json_from_text(response)
            
            # Handle if model returns a list wrapper
            if isinstance(result, list):
                if result:
                    result = result[0]
                else:
                    raise ValueError("Received empty list from model")
            
            # Validate new result structure
            if 'entities' not in result:
                # Try to convert old format to new format
                if 'entity' in result:
                    debug_print("JOINT1:ENTITY", "Converting old format to new multi-entity format")
                    result = {
                        'is_comparison': False,
                        'entities': [{
                            'name': result.get('entity', query),
                            'type': result.get('entity_type', 'unknown'),
                            'aliases': result.get('aliases', [])
                        }],
                        'action': result.get('action', 'information')
                    }
                else:
                    raise ValueError(f"Missing 'entities' key. Got: {result.keys()}")
            
            # Ensure entities is a list
            if not isinstance(result.get('entities'), list):
                raise ValueError(f"'entities' must be a list, got {type(result.get('entities'))}")
            
            # Ensure each entity has required keys
            for i, entity in enumerate(result['entities']):
                if 'name' not in entity:
                    raise ValueError(f"Entity {i} missing 'name' key")
                # Set defaults for optional fields
                entity.setdefault('type', 'unknown')
                entity.setdefault('aliases', [])
            
            elapsed = time.time() - start_time
            entity_names = [e['name'] for e in result['entities']]
            debug_print("JOINT1:ENTITY", f"Extracted {len(result['entities'])} entities: {entity_names}")
            debug_print("JOINT1:ENTITY", f"Is comparison: {result.get('is_comparison', False)}")
            debug_print("JOINT1:ENTITY", f"Action: {result.get('action', 'N/A')}")
            debug_print("JOINT1:ENTITY", f"Extraction took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            debug_print("JOINT1:ENTITY", f"Extraction failed: {type(e).__name__}: {e}")
            # Fallback: return query as single entity in new format
            debug_print("JOINT1:ENTITY", "Using fallback: query as single entity")
            return {
                "is_comparison": False,
                "entities": [{
                    "name": query,
                    "type": "unknown",
                    "aliases": []
                }],
                "action": "information"
            }
            
    def suggest_expansion(self, query: str, failed_terms: List[str]) -> List[str]:
        """
        Suggest alternative search terms when initial search fails.
        
        Args:
            query: User's original query
            failed_terms: List of terms that returned no results
            
        Returns:
            List of new search terms (strings)
        """
        debug_print("JOINT1:EXPAND", f"Suggesting expansion for '{query}' (failed: {failed_terms})")
        start_time = time.time()
        
        prompt = f"""The user asked about: "{query}"
        
        We searched for these terms but found NOTHING relevant: {failed_terms}
        
        INSTRUCTIONS:
        1. Suggest 3 alternative search queries.
        2. Focus on broader concepts, related events, or key figures.
        3. If the user used a nickname, try the real name.
        4. If the user asked a specific question, try searching for the general topic.
        
        Return ONLY a JSON list of strings:
        ["Alternative 1", "Alternative 2", "Alternative 3"]
        """
        
        try:
            response = local_inference(self.model, prompt, temperature=0.3, timeout=config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT1:EXPAND", f"Raw response: {response[:200]}...")
            
            suggestions = extract_json_from_text(response)
            
            if isinstance(suggestions, list):
                # Filter out duplicates and empty strings
                filtered = [s for s in suggestions if isinstance(s, str) and s.strip() and s not in failed_terms]
                debug_print("JOINT1:EXPAND", f"Generated {len(filtered)} suggestions: {filtered}")
                return filtered[:3]
                
            return []
            
        except Exception as e:
            debug_print("JOINT1:EXPAND", f"Expansion failed: {e}")
            return []


class ArticleScorerJoint:
    """
    Joint 2: Article Scoring
    
    Scores Wikipedia article titles by relevance to the extracted entity.
    Uses qwen2.5:0.5b for fast scoring.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.SCORER_JOINT_MODEL
        self.temperature = config.SCORER_JOINT_TEMP
        debug_print("JOINT2:INIT", f"ArticleScorer initialized with {self.model}")
    
    def score(self, query: str, entity_info: Dict, article_titles: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Score article titles by relevance to entities and the original query.
        
        Args:
            query: Original user query for context
            entity_info: Entity information from EntityExtractorJoint (new multi-entity format)
            article_titles: List of Wikipedia article titles
            top_k: Return top K scored articles
            
        Returns:
            List of (title, score) tuples, sorted by score descending
        """
        if not article_titles:
            debug_print("JOINT2:SCORER", "No articles to score")
            return []
        
        # Extract all entity names and aliases from new format
        all_entity_names = []
        for entity in entity_info.get('entities', []):
            all_entity_names.append(entity.get('name', ''))
            all_entity_names.extend(entity.get('aliases', []))
        
        # Filter out empty strings
        all_entity_names = [n for n in all_entity_names if n]
        
        # Backwards compatibility: support old format
        if not all_entity_names and 'entity' in entity_info:
            all_entity_names = [entity_info['entity']] + entity_info.get('aliases', [])
        
        entities_display = [e.get('name', '') for e in entity_info.get('entities', [])]
        debug_print("JOINT2:SCORER", f"Scoring {len(article_titles)} articles for entities: {entities_display}")
        start_time = time.time()
        
        # === EXACT MATCH OVERRIDE ===
        # Before LLM scoring, check for exact entity matches
        # These get score 11.0 (above max 10) to guarantee inclusion
        exact_match_scores = []
        
        for title in article_titles:
            title_lower = title.lower().strip()
            
            # Fix: Disambiguation Trap
            # Do not auto-select disambiguation pages even if they match entity name exactly
            if "(disambiguation)" in title_lower:
                debug_print("JOINT2:SCORER", f"EXACT MATCH SKIPPED: '{title}' (Disambiguation page)")
                continue

            for entity_name in all_entity_names:
                if title_lower == entity_name.lower().strip():
                    debug_print("JOINT2:SCORER", f"EXACT MATCH OVERRIDE: '{title}' == entity '{entity_name}' -> score 11.0")
                    exact_match_scores.append((title, 11.0))
                    break  # Only add once per title
        
        debug_print("JOINT2:SCORER", f"Found {len(exact_match_scores)} exact entity matches")
        
        # Format article titles for prompt (limit to prevent token overflow)
        articles_formatted = "\n".join([f"{i+1}. {title}" for i, title in enumerate(article_titles[:20])])
        
        # Format entities for prompt
        entities_str = ", ".join([f"'{e.get('name', '')}'" for e in entity_info.get('entities', [])])
        action = entity_info.get('action', 'information about')
        
        prompt = f"""I will give you a list of Article Titles.
        
USER'S ORIGINAL QUESTION: "{query}"

Select articles relevant to answering this question.
Entities mentioned: {entities_str}
        
RULES:
1. ONLY select titles from the provided INPUT LIST below.
2. DO NOT output example titles.
3. Output valid JSON only.
4. Prioritize articles that directly answer the user's question.
        
INPUT LIST:
{articles_formatted}
        
Rate each article 0-10 where:
- 10 = Directly relevant to answering the user's question
- 7-9 = Highly relevant to the entities or topic
- 0 = Not relevant
        
Return ONLY a JSON array:
[
  {{"title": "Actual Title From List", "score": 10}}
]"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT2:SCORER", f"Raw response: {response[:200]}...")
            
            # Use robust extractor
            scores = extract_json_from_text(response)
            
            if isinstance(scores, dict):
                # Handle wrapped format like {"items": [...]} or {"scores": [...]}
                # Common wrapper keys: items, scores, results, articles
                for key in ["items", "scores", "results", "articles"]:
                    if key in scores and isinstance(scores[key], list):
                         debug_print("JOINT2:SCORER", f"Unwrapped list from key '{key}'")
                         scores = scores[key]
                         break
            
            if not isinstance(scores, list):
                 raise ValueError(f"Response is not a JSON array (got {type(scores).__name__})")

            # --- VALIDATION & FILTERING ---
            # Helper: Normalize a title for fuzzy matching
            def normalize_title(t: str) -> str:
                """Lowercase and remove commas/punctuation for comparison."""
                return re.sub(r'[,.:;\'\"-]+', '', t.lower()).strip()
            
            def fuzzy_match(llm_title: str, candidates: List[str]) -> Optional[str]:
                """
                Try to match llm_title to a candidate using fuzzy matching.
                Returns the original candidate title if matched, None otherwise.
                """
                norm_llm = normalize_title(llm_title)
                for candidate in candidates:
                    norm_cand = normalize_title(candidate)
                    # Exact normalized match
                    if norm_llm == norm_cand:
                        return candidate
                    # Substring match (either direction)
                    if norm_cand in norm_llm or norm_llm in norm_cand:
                        return candidate
                return None
            
            # 1. Verification Set: Only allow titles that match candidates (fuzzy)
            valid_titles = list(article_titles)
            
            # 2. Placeholder Pattern: reject titles with suspicious placeholder names
            placeholder_pattern = re.compile(r'article\s+name|title\s+\d+|example\s+article', re.IGNORECASE)
            
            scored_articles = []
            for item in scores:
                llm_title = item.get('title')
                score = float(item.get('score', 0))
                
                # Check 1: Must match original list (exact or fuzzy)
                if llm_title in valid_titles:
                    matched_title = llm_title  # Exact match
                else:
                    matched_title = fuzzy_match(llm_title, valid_titles)
                    if matched_title:
                        debug_print("JOINT2:SCORER", f"Fuzzy matched: '{llm_title}' -> '{matched_title}'")
                    else:
                        debug_print("JOINT2:SCORER", f"Filtered hallucination: '{llm_title}' (not in candidates)")
                        continue
                    
                # Check 2: Must not be a placeholder
                if placeholder_pattern.search(matched_title):
                    debug_print("JOINT2:SCORER", f"Filtered placeholder: '{matched_title}'")
                    continue
                
                scored_articles.append((matched_title, score))
            
            # Sort by score
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            
            # === MERGE EXACT MATCH OVERRIDES ===
            # Prepend exact matches to ensure they're always in top results
            # Avoid duplicates by filtering out titles already in exact_match_scores
            exact_titles = {t for t, _ in exact_match_scores}
            scored_articles = [item for item in scored_articles if item[0] not in exact_titles]
            final_results = exact_match_scores + scored_articles
            
            debug_print("JOINT2:SCORER", f"After exact match merge: {len(final_results)} total articles")
            
            elapsed = time.time() - start_time
            debug_print("JOINT2:SCORER", f"Scored {len(final_results)} valid articles in {elapsed:.2f}s")
            debug_print("JOINT2:SCORER", f"Top 5 scores: {final_results[:5]}")
            
            return final_results[:top_k]
            
        except Exception as e:
            debug_print("JOINT2:SCORER", f"Scoring failed: {type(e).__name__}: {e}")
            if 'response' in locals():
                debug_print("JOINT2:SCORER", f"FAILED RESPONSE RAW: {response}")
            
            # Fallback: return all articles with equal scores
            debug_print("JOINT2:SCORER", "Using fallback: equal scores")
            return [(title, 5.0) for title in article_titles[:top_k]]


class CoverageVerifierJoint:
    """
    Joint 2.5: Coverage Verification
    
    Checks if the selected articles cover all required entities.
    For comparison queries, ensures each entity has at least one article.
    If gaps are found, suggests targeted search terms.
    
    This joint does NOT use an LLM - it's a fast string-matching check
    that triggers secondary searches only when coverage gaps are detected.
    """
    
    def __init__(self):
        debug_print("JOINT2.5:INIT", "CoverageVerifier initialized (no LLM required)")
    
    def verify_coverage(
        self, 
        entity_info: Dict, 
        selected_articles: List[Dict]
    ) -> Dict:
        """
        Verify that selected articles cover all necessary entities.
        
        Args:
            entity_info: Output from EntityExtractorJoint
            selected_articles: List of article candidates (with 'metadata' containing 'title')
            
        Returns:
            {
                'complete': bool,          # True if all entities are covered
                'covered': List[str],      # Entities with matching articles
                'missing': List[str],      # Entities without articles
                'suggested_searches': List[str]  # Targeted search terms for missing
            }
        """
        # Extract entity names from entity_info
        entities = entity_info.get('entities', [])
        entity_names = [e.get('name', '').strip() for e in entities if e.get('name')]
        
        # Also include aliases for matching
        entity_aliases = {}
        for e in entities:
            name = e.get('name', '').strip()
            if name:
                entity_aliases[name] = [a.strip().lower() for a in e.get('aliases', []) if a]
        
        debug_print("JOINT2.5:VERIFY", f"Checking coverage for {len(entity_names)} entities: {entity_names}")
        
        # Get all article titles (normalized for matching)
        article_titles = []
        for article in selected_articles:
            title = article.get('metadata', {}).get('title', '')
            if title:
                article_titles.append(title.lower().strip())
        
        debug_print("JOINT2.5:VERIFY", f"Selected articles: {article_titles}")
        
        # Check coverage for each entity
        covered = []
        missing = []
        
        for entity_name in entity_names:
            entity_lower = entity_name.lower()
            aliases = entity_aliases.get(entity_name, [])
            
            # Check if any article title contains this entity or its aliases
            found = False
            for title in article_titles:
                # Direct match
                if entity_lower in title:
                    found = True
                    debug_print("JOINT2.5:VERIFY", f"  '{entity_name}' found in '{title}'")
                    break
                # Alias match
                for alias in aliases:
                    if alias in title:
                        found = True
                        debug_print("JOINT2.5:VERIFY", f"  '{entity_name}' (alias '{alias}') found in '{title}'")
                        break
                if found:
                    break
            
            if found:
                covered.append(entity_name)
            else:
                missing.append(entity_name)
                debug_print("JOINT2.5:VERIFY", f"  '{entity_name}' NOT FOUND in any selected article")
        
        # Generate suggested searches for missing entities
        suggested_searches = []
        for entity_name in missing:
            # Try multiple search variations
            suggested_searches.append(entity_name)  # Direct name
            # Add context-aware variations based on entity type
            for e in entities:
                if e.get('name') == entity_name:
                    entity_type = e.get('type', '').lower()
                    if entity_type in ['technology', 'concept']:
                        suggested_searches.append(f"{entity_name} (technology)")
                    elif entity_type == 'event':
                        suggested_searches.append(f"{entity_name} disaster")
                        suggested_searches.append(f"{entity_name} incident")
                    elif entity_type == 'person':
                        suggested_searches.append(f"{entity_name} biography")
                    break
        
        result = {
            'complete': len(missing) == 0,
            'covered': covered,
            'missing': missing,
            'suggested_searches': suggested_searches[:6]  # Limit to 6 suggestions
        }
        
        if missing:
            debug_print("JOINT2.5:VERIFY", f"Coverage INCOMPLETE: missing {missing}")
            debug_print("JOINT2.5:VERIFY", f"Suggested searches: {suggested_searches}")
        else:
            debug_print("JOINT2.5:VERIFY", "Coverage COMPLETE: all entities found")
        
        return result


class ChunkFilterJoint:
    """
    Joint 3: Chunk Filtering
    
    Filters retrieved chunks by relevance to the original query.
    Uses llama3.2:1b for intelligent chunk evaluation.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.FILTER_JOINT_MODEL
        self.temperature = config.FILTER_JOINT_TEMP
        debug_print("JOINT3:INIT", f"ChunkFilter initialized with {self.model}")
    
    def filter(self, query: str, chunks: List[Dict], top_k: int = 5, entity_info: Dict = None, mode: str = "FACTUAL") -> List[Dict]:
        """
        Filter chunks by query relevance.
        
        Args:
            query: Original user query
            chunks: List of chunk dicts with 'text' and 'metadata' keys
            top_k: Return top K relevant chunks
            entity_info: Optional entity info for comparison-aware filtering
            mode: Operational mode (e.g., "FACTUAL", "CODE")
        """
        if not chunks:
            debug_print("JOINT3:FILTER", "No chunks to filter")
            return []
        
        # Check if this is a comparison query
        is_comparison = entity_info.get('is_comparison', False) if entity_info else False
        entities = entity_info.get('entities', []) if entity_info else []
        entity_names = [e.get('name', '').lower() for e in entities]
        
        debug_print("JOINT3:FILTER", f"Filtering {len(chunks)} chunks for query '{query}'")
        debug_print("JOINT3:FILTER", f"Is comparison: {is_comparison}, Entities: {entity_names}")
        start_time = time.time()
        
        # For comparison queries, use diversity-aware selection as primary strategy
        # This ensures we get chunks from ALL entities
        if is_comparison and len(entity_names) >= 2:
            debug_print("JOINT3:FILTER", "Using diversity-aware selection for comparison query")
            return self._diversity_filter(chunks, entity_names, top_k)
        
        # Format chunks for prompt (truncate long chunks, limit to 20)
        chunks_formatted = []
        chunks_formatted = []
        for i, chunk in enumerate(chunks[:15]):
            text = chunk['text'][:250]  # Truncate to 250 chars
            chunks_formatted.append(f"{i+1}. {text}...")
        
        chunks_text = "\n\n".join(chunks_formatted)
        
        # Custom Prompt for CODE mode
        if mode == "CODE":
            debug_print("JOINT3:FILTER", "Using CODE-specific prompt")
            prompt = f"""Rate these text chunks for how well they provide CODE implementation for this query.

Query: {query}

Chunks:
{chunks_text}

Rate each chunk 0-10 where:
- 10 = Contains actual code blocks, function signatures, or API usage relevant to request.
- 7-9 = Technical documentation explaining the class/function.
- 0 = History/Biography/General text without code.

CRITICAL RULES:
1. PRIORITIZE SYNTAX: Chunks with `def`, `class`, `{{`, or indentation get priority.
2. IGNORE HISTORY: If the chunk talks about "history of Python" but has no code, rate it LOW (0-2).
3. We need RUNNABLE EXAMPLES.

Return ONLY a JSON array:
[
  {{"chunk_id": 1, "score": 10}},
  {{"chunk_id": 2, "score": 3}}
]
"""
        else:
            # Standard Fact/History Prompt
            prompt = f"""Rate these text chunks for how well they answer this query.

Query: {query}

Chunks:
{chunks_text}

Rate each chunk 0-10 where:
- 10 = Directly answers the query (Prioritize BIOGRAPHICAL/HISTORICAL facts for "who/what/when" queries)
- 7-9 = Highly relevant context
- 4-6 = Related information
- 1-3 = Tangentially related (or fictional/pop-culture references for historical queries)
- 0 = Not relevant

CRITICAL RULES:
1. If the query asks for HISTORICAL facts (e.g. "how did he die"), penalize chunks about Movies, Songs, or Pop Culture unless specifically asked for.
2. A snippet from a "Film" article describing a plot should get a LOW score (1-3) if the user wants real history.
3. A snippet from a "Biography" article describing real events should get a HIGH score (8-10).

Return ONLY a JSON array:
[
  {{"chunk_id": 1, "score": 10}},
  {{"chunk_id": 2, "score": 3}}
]
"""
        prompt += "\nRate ALL chunks. No explanation, only JSON."

        try:
            # Use longer timeout for chunk filtering since it processes more text
            response = local_inference(self.model, prompt, self.temperature, timeout=config.JOINT_TIMEOUT + 5, use_json_grammar=True)
            debug_print("JOINT3:FILTER", f"Raw response: {response[:200]}...")
            
            # Robust JSON Lines parsing helper
            def parse_json_lines(raw: str) -> List[Dict]:
                """
                Parse JSON that may be an array, a single dict, or multiple
                JSON objects separated by commas/newlines (JSON Lines).
                """
                # 1. Try standard parsing first
                try:
                    result = extract_json_from_text(raw)
                    if isinstance(result, list):
                        return result
                    if isinstance(result, dict):
                        # Got a single dict, try wrapping it
                        debug_print("JOINT3:FILTER", "Got single dict, attempting wrap")
                        pass  # Fall through to wrapping logic
                except (ValueError, json.JSONDecodeError):
                    pass  # Fall through to fallback
                
                # 2. Try wrapping the raw string in brackets
                try:
                    wrapped = f"[{raw.strip()}]"
                    result = json.loads(wrapped)
                    if isinstance(result, list):
                        debug_print("JOINT3:FILTER", "Parsed via bracket wrapping")
                        return result
                except json.JSONDecodeError:
                    pass  # Fall through to regex
                
                # 3. Regex fallback: find all JSON objects individually
                debug_print("JOINT3:FILTER", "Using regex fallback for JSON objects")
                objects = []
                # Match individual JSON objects (non-greedy)
                for match in re.finditer(r'\{[^{}]*\}', raw):
                    try:
                        obj = json.loads(match.group())
                        objects.append(obj)
                    except json.JSONDecodeError:
                        continue
                
                if objects:
                    debug_print("JOINT3:FILTER", f"Regex extracted {len(objects)} objects")
                    return objects
                
                raise ValueError("Could not parse any JSON from response")
            
            # Use robust JSON Lines parser
            scores = parse_json_lines(response)
            
            # Handle wrapper object {"chunks": [...]}
            if isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], dict) and "chunks" in scores[0]:
                scores = scores[0]["chunks"]
            elif isinstance(scores, dict) and "chunks" in scores:
                scores = scores["chunks"]
            
            if not isinstance(scores, list):
                 raise ValueError(f"Response is not a JSON array (got {type(scores).__name__})")
            
            # Create scored chunks list
            scored_chunks = []
            for item in scores:
                chunk_id = item.get('chunk_id')
                if chunk_id is None:
                    continue
                    
                chunk_idx = chunk_id - 1  # Convert to 0-indexed
                if 0 <= chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    chunk['relevance_score'] = float(item.get('score', 0))
                    scored_chunks.append(chunk)
            
            # Sort by score and return top-k
            scored_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            filtered = scored_chunks[:top_k]
            
            elapsed = time.time() - start_time
            debug_print("JOINT3:FILTER", f"Filtered to {len(filtered)} chunks in {elapsed:.2f}s")
            if filtered:
                avg_score = sum(c.get('relevance_score', 0) for c in filtered) / len(filtered)
                debug_print("JOINT3:FILTER", f"Average relevance score: {avg_score:.1f}/10")
            
            return filtered
            
        except Exception as e:
            debug_print("JOINT3:FILTER", f"Filtering failed: {type(e).__name__}: {e}")
            # Fallback: return original chunks (use existing scores if available)
            debug_print("JOINT3:FILTER", "Using fallback: original chunk order")
            return chunks[:top_k]

    def _diversity_filter(self, chunks: List[Dict], entity_names: List[str], top_k: int) -> List[Dict]:
        """
        Diversity-aware chunk selection that ensures coverage of all entities.
        
        For comparison queries, we need chunks from EACH entity to provide
        a balanced answer.
        """
        debug_print("JOINT3:FILTER", f"Diversity filter: finding chunks for {len(entity_names)} entities")
        
        # Group chunks by which entity they likely belong to
        entity_chunks = {name: [] for name in entity_names}
        entity_chunks['other'] = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            chunk_title = chunk.get('metadata', {}).get('title', '').lower()
            matched = False
            
            for entity_name in entity_names:
                # Check if chunk is about this entity (by text or title)
                if entity_name in chunk_text or entity_name in chunk_title:
                    entity_chunks[entity_name].append(chunk)
                    matched = True
                    break
            
            if not matched:
                entity_chunks['other'].append(chunk)
        
        # Log distribution
        for name, c_list in entity_chunks.items():
            debug_print("JOINT3:FILTER", f"  Entity '{name}': {len(c_list)} chunks")
        
        # Allocate slots proportionally, but ensure minimum per entity
        num_entities = len(entity_names)
        min_per_entity = max(1, top_k // (num_entities + 1))  # +1 for 'other'
        slots_remaining = top_k
        
        selected = []
        
        # First pass: select min_per_entity from each entity
        for entity_name in entity_names:
            entity_list = entity_chunks[entity_name]
            to_take = min(min_per_entity, len(entity_list), slots_remaining)
            selected.extend(entity_list[:to_take])
            slots_remaining -= to_take
        
        # Second pass: fill remaining slots with best remaining chunks
        if slots_remaining > 0:
            # Collect all unused chunks
            used_ids = {id(c) for c in selected}
            remaining = []
            for name, c_list in entity_chunks.items():
                for chunk in c_list:
                    if id(chunk) not in used_ids:
                        remaining.append(chunk)
            
            # Sort by RRF score if available, otherwise just take in order
            remaining.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            selected.extend(remaining[:slots_remaining])
        
        # Add relevance scores for consistency
        for i, chunk in enumerate(selected):
            if 'relevance_score' not in chunk:
                chunk['relevance_score'] = 8.0 - (i * 0.5)  # Descending scores
        
        debug_print("JOINT3:FILTER", f"Diversity filter selected {len(selected)} chunks")
        return selected


class FactRefinementJoint:
    """
    Joint 4: Fact Refinement
    
    Scans the content of the selected article to extract verifiable facts
    related to the user's query.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.FACT_JOINT_MODEL
        self.temperature = config.FACT_JOINT_TEMP
        debug_print("JOINT4:INIT", f"FactRefinement initialized with {self.model}")
    
    def refine_facts(self, query: str, text_content: str) -> List[str]:
        """
        Extract specific facts from text relevant to query.
        
        Args:
            query: User query
            text_content: Content of the top relevant article/chunks
            
        Returns:
            List of factual strings
        """
        if not text_content:
            return []
            
        debug_print("JOINT4:FACTS", f"Refining facts for query: '{query}'")
        start_time = time.time()
        
        # Truncate text to avoid context limit (approx 3000 chars)
        context_window = text_content[:3500]
        
        prompt = f"""Extract verified factual details from the text below that are relevant to this query.

Query: "{query}"

Text:
{context_window}

INSTRUCTIONS:
1. List 3-5 specific, self-contained facts found in the text.
2. Direct quotes or precise numbers are best.
3. If the text does not contain the answer, return an empty list.
4. Return ONLY a JSON list of strings.

Example:
["Tupac Shakur died on September 13, 1996.", "He was 25 years old."]

JSON Response:"""

        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT4:FACTS", f"Raw response: {response[:200]}...")
            
            # Use extract_json_from_text from the module scope
            facts = extract_json_from_text(response)
            
            if isinstance(facts, dict):
                # Convert dict values to list of facts
                debug_print("JOINT4:FACTS", "Converted dict to list values")
                facts = [f"{k}: {v}" for k, v in facts.items()]
            
            if not isinstance(facts, list):
                # Try to parse list from text lines if JSON fails
                lines = response.strip().split('\\n')
                facts = [line.strip('- ').strip() for line in lines if line.strip()]
                
            filtered_facts = [f for f in facts if isinstance(f, str) and len(f) > 10]
            
            elapsed = time.time() - start_time
            debug_print("JOINT4:FACTS", f"Extracted {len(filtered_facts)} facts in {elapsed:.2f}s")
            
            return filtered_facts
            
        except Exception as e:
            debug_print("JOINT4:FACTS", f"Fact extraction failed: {e}")
            return []

    def verify_premise(self, query: str, text_content: str) -> Dict[str, str]:
        """
        Check if the text actually supports the user's premise.
        
        Args:
            query: User query (e.g. "Was Tesla a CIA agent?")
            text_content: Content of the top relevant article
            
        Returns:
            Dict: {'status': 'SUPPORTED'|'UNSUPPORTED'|'CONTRADICTED', 'reason': '...'}
        """
        if not text_content:
            return {'status': 'UNSUPPORTED', 'reason': 'No data found'}
            
        debug_print("JOINT4:VERIFY", f"Verifying premise for: '{query}'")
        start_time = time.time()
        
        # Truncate for speed
        context_window = text_content[:2000]
        
        prompt = f"""Analyze if the text supports the premise in the user's query.

Query: "{query}"

Text:
{context_window}

INSTRUCTIONS:
1. Identify the core premise.
   - If Query is a specific claim (e.g. "Did Tesla work for CIA?"), verify that claim.
   - If Query is a TOPIC (e.g. "Synthesis of Aspirin", "Photosynthesis"), the premise is "The text contains information about [Topic]".
2. Check if the text supports it.
   - For Topic Queries: Return SUPPORTED if the text discusses the topic. Return UNSUPPORTED only if the text is completely unrelated.
3. BE SKEPTICAL BUT ACCURATE.
   - Do NOT invent claims not present in the query (e.g. do not assume "using cold water" if user didn't say it).
   - If the text contradicts a specific user claim, return CONTRADICTED.

Return JSON ONLY:
{{
  "premise": "The user is asking about [Topic]..." or "The user assumes [Claim]...",
  "status": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED",
  "reason": "The text discusses X..." or "The text explicitly refutes X..."
}}
"""
        
        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            debug_print("JOINT4:VERIFY", f"Raw response: {response[:200]}...")
            
            result = extract_json_from_text(response)
            
            if isinstance(result, dict) and 'status' in result:
                debug_print("JOINT4:VERIFY", f"Result: {result['status']} ({result.get('reason')})")
                return result
                
            return {'status': 'UNSUPPORTED', 'reason': 'Could not verify'}
            
        except Exception as e:
            debug_print("JOINT4:VERIFY", f"Verification failed: {e}")
            return {'status': 'UNSUPPORTED', 'reason': 'Error during verification'}
