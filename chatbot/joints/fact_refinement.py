
import re
import time
from typing import Dict, List, Tuple
from chatbot import config
from .base import debug_print, local_inference, extract_json_from_text


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from a query for relevance scoring."""
    # Remove common stopwords and question words
    stopwords = {
        'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were',
        'the', 'a', 'an', 'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'it', 'this', 'that', 'which', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'may',
        'about', 'between', 'their', 'them', 'they', 'its', 'his', 'her'
    }

    # Normalize and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = [w for w in words if w not in stopwords]

    return keywords


def _score_chunk(chunk: str, keywords: List[str]) -> float:
    """Score a text chunk by keyword overlap with the query."""
    chunk_lower = chunk.lower()
    score = 0.0

    for keyword in keywords:
        # Count occurrences (capped at 3 per keyword to avoid gaming)
        count = min(chunk_lower.count(keyword), 3)
        score += count

        # Bonus for exact word matches (not substrings)
        if re.search(rf'\b{re.escape(keyword)}\b', chunk_lower):
            score += 0.5

    return score


def _smart_truncate(query: str, text_content: str, max_chars: int = 2000) -> str:
    """
    Query-aware chunking: return the most relevant ~max_chars from text.

    Instead of blindly truncating at 2000 chars, we:
    1. Split text into overlapping chunks
    2. Score each chunk by keyword overlap with query
    3. Return the highest-scoring chunks up to max_chars
    """
    if len(text_content) <= max_chars:
        return text_content

    keywords = _extract_keywords(query)

    if not keywords:
        # Fallback to first max_chars if no keywords extracted
        return text_content[:max_chars]

    # Split into chunks with overlap for context continuity
    chunk_size = 400
    overlap = 100
    chunks: List[Tuple[int, str, float]] = []  # (start_idx, text, score)

    for i in range(0, len(text_content), chunk_size - overlap):
        chunk = text_content[i:i + chunk_size]
        if len(chunk) < 50:  # Skip tiny trailing chunks
            continue
        score = _score_chunk(chunk, keywords)
        chunks.append((i, chunk, score))

    if not chunks:
        return text_content[:max_chars]

    # Sort by score (descending), then by position (ascending) for ties
    chunks.sort(key=lambda x: (-x[2], x[0]))

    # Select top chunks up to max_chars
    selected_chunks: List[Tuple[int, str]] = []
    total_chars = 0

    for start_idx, chunk_text, score in chunks:
        if total_chars + len(chunk_text) > max_chars:
            # Try to fit a partial chunk
            remaining = max_chars - total_chars
            if remaining > 100:
                selected_chunks.append((start_idx, chunk_text[:remaining]))
                total_chars += remaining
            break
        selected_chunks.append((start_idx, chunk_text))
        total_chars += len(chunk_text)

    # Sort selected chunks by original position for coherent text
    selected_chunks.sort(key=lambda x: x[0])

    # Join with ellipsis if chunks are non-contiguous
    result_parts = []
    last_end = -1

    for start_idx, chunk_text in selected_chunks:
        if last_end >= 0 and start_idx > last_end + 50:
            result_parts.append("\n[...]\n")
        result_parts.append(chunk_text)
        last_end = start_idx + len(chunk_text)

    return ''.join(result_parts)


class FactRefinementJoint:
    """
    Joint 4: Fact Refinement

    Extracts verifiable facts from content with:
    - Smart truncation (query-aware chunking)
    - Batch extraction (multiple articles in one LLM call)
    - Premise verification (contradiction detection)
    """

    def __init__(self, model: str = None):
        self.model = model or getattr(config, 'REFINEMENT_JOINT_MODEL', config.FACT_JOINT_MODEL)
        debug_print("JOINT4:INIT", f"FactRefinement initialized with {self.model}")

    def refine_facts(self, query: str, text_content: str) -> List[str]:
        """
        Extract specific facts from text relevant to query.
        Uses smart truncation to select most relevant content.
        """
        # Apply smart truncation instead of blind [:2000]
        truncated_text = _smart_truncate(query, text_content, max_chars=2000)

        prompt = f"""Extract 3-5 key facts from the text that help answer the query.
Query: {query}
Text: {truncated_text}

Return ONLY a JSON list of strings.
"""
        try:
            response = local_inference(self.model, prompt, temperature=0.1, use_json_grammar=True)
            facts = extract_json_from_text(response)
            if isinstance(facts, list):
                return facts
            return []
        except:
            return []

    def refine_facts_batch(self, query: str, articles: List[Dict]) -> List[Dict]:
        """
        Extract facts from multiple articles in a SINGLE LLM call.

        Args:
            query: The user's question
            articles: List of article dicts with 'text' and 'metadata' keys

        Returns:
            List of dicts with 'title' and 'facts' keys
        """
        if not articles:
            return []

        # Limit to 3 articles max to stay within context
        articles = articles[:3]

        # Build combined prompt with smart truncation per article
        article_texts = []
        for i, article in enumerate(articles):
            title = article.get('metadata', {}).get('title', f'Article {i+1}')
            text = article.get('text', '')
            # Use smaller chunks when batching multiple articles
            truncated = _smart_truncate(query, text, max_chars=1200)
            article_texts.append(f"[ARTICLE {i+1}: {title}]\n{truncated}\n")

        combined_text = "\n---\n".join(article_texts)

        prompt = f"""Extract 2-4 key facts from EACH article that help answer the query.
Query: {query}

{combined_text}

Return a JSON array where each element has "title" and "facts" (list of strings).
Example: [{{"title": "Article Title", "facts": ["fact 1", "fact 2"]}}]
"""
        try:
            response = local_inference(self.model, prompt, temperature=0.1, use_json_grammar=True)
            results = extract_json_from_text(response)

            if isinstance(results, list):
                # Validate structure
                validated = []
                for item in results:
                    if isinstance(item, dict) and 'facts' in item:
                        validated.append({
                            'title': item.get('title', 'Unknown'),
                            'facts': item.get('facts', []) if isinstance(item.get('facts'), list) else []
                        })
                return validated
            return []
        except Exception as e:
            debug_print("JOINT4:BATCH", f"Batch extraction failed: {e}")
            # Fallback to individual extraction
            results = []
            for article in articles:
                title = article.get('metadata', {}).get('title', 'Unknown')
                facts = self.refine_facts(query, article.get('text', ''))
                if facts:
                    results.append({'title': title, 'facts': facts})
            return results

    def verify_premise(self, query: str, text_content: str) -> Dict:
        """
        Check if the text actually supports the user's premise.
        Detects contradictions in user assumptions.

        Returns:
            Dict with 'status' (SUPPORTED/CONTRADICTED/UNKNOWN) and 'reason'
        """
        # Apply smart truncation
        truncated_text = _smart_truncate(query, text_content, max_chars=1500)

        prompt = f"""Analyze whether the source text supports or contradicts the user's query/assumption.

User Query: {query}
Source Text: {truncated_text}

Determine:
1. Does the query contain an assumption or premise?
2. Does the text support, contradict, or not address that premise?

Return JSON with:
- "status": "SUPPORTED" | "CONTRADICTED" | "UNKNOWN"
- "reason": Brief explanation (1-2 sentences)
- "premise": The detected premise from the query (if any)

If the query is a simple question with no assumption, return "SUPPORTED" with reason "Simple factual query".
"""
        try:
            response = local_inference(self.model, prompt, temperature=0.0, use_json_grammar=True)
            result = extract_json_from_text(response)

            if isinstance(result, dict) and 'status' in result:
                # Validate status value
                status = result.get('status', 'UNKNOWN').upper()
                if status not in ('SUPPORTED', 'CONTRADICTED', 'UNKNOWN'):
                    status = 'UNKNOWN'

                return {
                    "status": status,
                    "reason": result.get('reason', 'Unable to determine.'),
                    "premise": result.get('premise', None)
                }

            return {"status": "UNKNOWN", "reason": "Failed to parse verification result."}

        except Exception as e:
            debug_print("JOINT4:VERIFY", f"Premise verification failed: {e}")
            return {"status": "UNKNOWN", "reason": f"Verification error: {str(e)}"}
