
import time
from typing import Dict, List, Any, Optional
from chatbot import config
from .base import debug_print, local_inference, extract_json_from_text

class MultiHopResolverJoint:
    """
    Joint 0.5: Multi-Hop Resolver
    
    Detects and resolves indirect entity references in queries.
    Example: "What university did the creator of Python attend?"
    - Detects: "creator of Python" is an indirect reference
    - Resolves: "creator of Python" → "Guido van Rossum"
    - Enables: Second-hop search for "Guido van Rossum university"
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.MULTI_HOP_JOINT_MODEL
        self.temperature = 0.1  # Low temp for precise extraction
        debug_print("JOINT0.5:INIT", f"MultiHopResolver initialized with {self.model}")
    
    def detect_indirect_pattern(self, query: str, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Detect if the query contains indirect entity references.
        
        Args:
            query: User query string
            entities: Extracted entities from EntityExtractorJoint
            
        Returns:
            Dict with pattern info if detected, None otherwise
            {
                "has_indirect": true,
                "base_entity": "Python (programming language)",
                "relationship": "creator",
                "target_type": "person"
            }
        """
        debug_print("JOINT0.5:DETECT", f"Analyzing query for indirect patterns: '{query}'")
        start_time = time.time()
        
        # Quick heuristic check first
        indirect_keywords = [
            'creator of', 'inventor of', 'founder of', 'author of',
            'capital of', 'leader of', 'president of', 'CEO of',
            'director of', 'developer of', 'designer of',
            'birthplace of', 'location of', 'home of'
        ]
        
        query_lower = query.lower()
        has_keyword = any(keyword in query_lower for keyword in indirect_keywords)
        
        if not has_keyword:
            debug_print("JOINT0.5:DETECT", "No indirect keywords detected")
            return None
        
        # Use LLM for precise pattern analysis
        prompt = f"""Analyze this query for indirect entity references (e.g., "the creator of X", "capital of Y").

Query: "{query}"

Extracted entities: {[e.get('name', '') for e in entities]}

INSTRUCTIONS:
1. Determine if the query asks about a relationship to an entity (has_indirect).
2. If yes, identify:
   - base_entity: The entity we need to look up first (e.g., "Python")
   - relationship: What relationship we're looking for (e.g., "creator", "capital", "inventor")
   - target_type: What type of entity we want to find (person, place, organization, etc.)

EXAMPLES:
Query: "What university did the creator of Python attend?"
Result: {{
  "has_indirect": true,
  "base_entity": "Python (programming language)",
  "relationship": "creator",
  "target_type": "person"
}}

Query: "What is the capital of France?"
Result: {{
  "has_indirect": true,
  "base_entity": "France",
  "relationship": "capital",
  "target_type": "city"
}}

Query: "Who invented the telephone?"
Result: {{
  "has_indirect": false,
  "reason": "Direct question, no indirect reference"
}}

Query: "Compare Python and Java"
Result: {{
  "has_indirect": false,
  "reason": "Comparison query, entities are direct"
}}

Return ONLY valid JSON. No markdown code blocks.
"""
        
        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            result = extract_json_from_text(response)
            
            if result and result.get('has_indirect'):
                debug_print("JOINT0.5:DETECT", f"Indirect pattern detected: {result.get('relationship')} of {result.get('base_entity')}")
                elapsed = time.time() - start_time
                debug_print("JOINT0.5:DETECT", f"Detection took {elapsed:.2f}s")
                return result
            else:
                debug_print("JOINT0.5:DETECT", "No indirect pattern confirmed by LLM")
                return None
                
        except Exception as e:
            debug_print("JOINT0.5:DETECT", f"Detection failed: {e}")
            return None
    
    def resolve_entity(self, base_entity: str, relationship: str, article_content: str) -> Optional[str]:
        """
        Extract the referenced entity from the base entity's article.
        
        Args:
            base_entity: The entity we retrieved (e.g., "Python (programming language)")
            relationship: What we're looking for (e.g., "creator", "capital")
            article_content: Text content of the base entity's article
            
        Returns:
            Resolved entity name (e.g., "Guido van Rossum") or None
        """
        debug_print("JOINT0.5:RESOLVE", f"Resolving '{relationship}' from {base_entity} article")
        start_time = time.time()
        
        # Truncate article to first 2000 chars to fit in context
        content_excerpt = article_content[:2000]
        
        prompt = f"""Extract the {relationship} of {base_entity} from this Wikipedia article excerpt.

Article: {base_entity}
Looking for: {relationship}
Content:
{content_excerpt}

INSTRUCTIONS:
1. Find the {relationship} mentioned in the article.
2. Return ONLY the person/entity name, formatted as a Wikipedia article title.
3. If multiple people are mentioned, return the primary/first one.
4. If not found, return null.

EXAMPLES:
Looking for: creator of Python
Content: "Python was created by Guido van Rossum in 1991..."
Result: {{"entity": "Guido van Rossum"}}

Looking for: capital of France
Content: "France is a country whose capital is Paris..."
Result: {{"entity": "Paris"}}

Looking for: inventor of telephone
Content: "The telephone was invented by Alexander Graham Bell..."
Result: {{"entity": "Alexander Graham Bell"}}

Return ONLY valid JSON: {{"entity": "Name"}} or {{"entity": null}}
"""
        
        try:
            response = local_inference(self.model, prompt, self.temperature, config.JOINT_TIMEOUT, use_json_grammar=True)
            result = extract_json_from_text(response)
            
            if result and result.get('entity'):
                resolved = result['entity']
                debug_print("JOINT0.5:RESOLVE", f"Resolved to: {resolved}")
                elapsed = time.time() - start_time
                debug_print("JOINT0.5:RESOLVE", f"Resolution took {elapsed:.2f}s")
                return resolved
            else:
                debug_print("JOINT0.5:RESOLVE", "Could not resolve entity from article")
                return None
                
        except Exception as e:
            debug_print("JOINT0.5:RESOLVE", f"Resolution failed: {e}")
            return None
    
    def process(self, query: str, entities: List[Dict[str, Any]], retrieved_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Complete multi-hop resolution pipeline.
        
        Args:
            query: Original user query
            entities: Extracted entities from EntityExtractorJoint
            retrieved_data: Already retrieved articles (should include base entity)
            
        Returns:
            Dict with resolution info:
            {
                "resolved_entity": "Guido van Rossum",
                "base_entity": "Python (programming language)",
                "relationship": "creator",
                "suggest_search": ["Guido van Rossum", "Guido_van_Rossum"]
            }
        """
        debug_print("JOINT0.5:PROCESS", f"Starting multi-hop resolution for: '{query}'")
        
        # Step 1: Detect indirect pattern
        pattern = self.detect_indirect_pattern(query, entities)
        if not pattern or not pattern.get('has_indirect'):
            debug_print("JOINT0.5:PROCESS", "No multi-hop needed")
            return None
        
        base_entity = pattern.get('base_entity')
        relationship = pattern.get('relationship')
        
        # Step 2: Find base entity article in retrieved data
        base_article = None
        for doc in retrieved_data:
            title = doc.get('metadata', {}).get('title', '')
            # Flexible matching (case-insensitive, ignore disambiguation)
            title_clean = title.lower().replace('(programming language)', '').strip()
            base_clean = base_entity.lower().replace('(programming language)', '').strip()
            
            if title_clean in base_clean or base_clean in title_clean:
                base_article = doc
                debug_print("JOINT0.5:PROCESS", f"Found base article: {title}")
                break
        
        if not base_article:
            debug_print("JOINT0.5:PROCESS", f"Base entity '{base_entity}' not found in retrieved data")
            return None
        
        # Step 3: Extract referenced entity from article
        article_text = base_article.get('text', '')
        resolved_entity = self.resolve_entity(base_entity, relationship, article_text)
        
        if not resolved_entity:
            debug_print("JOINT0.5:PROCESS", "Could not resolve entity from article")
            return None
        
        # Step 4: Generate search suggestions
        search_suggestions = [
            resolved_entity,
            resolved_entity.replace(' ', '_'),
            resolved_entity.replace(' ', '_').title()
        ]
        
        result = {
            "resolved_entity": resolved_entity,
            "base_entity": base_entity,
            "relationship": relationship,
            "suggest_search": search_suggestions
        }
        
        debug_print("JOINT0.5:PROCESS", f"✓ Multi-hop resolution complete: {resolved_entity}")
        return result
