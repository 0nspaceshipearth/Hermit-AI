
import os
import sys
import pickle
import numpy as np
import time
import glob
import string
import re
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Set

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError:
    faiss = None
    SentenceTransformer = None
    BM25Okapi = None

import libzim

from chatbot import config
from chatbot.debug_utils import debug_print
from chatbot.text_processing import TextProcessor
from chatbot.model_manager import ModelManager


def _normalize_query_for_cache(query: str) -> str:
    """Normalize query for cache lookup: lowercase, strip punctuation, collapse whitespace."""
    normalized = query.lower().strip()
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))
    normalized = ' '.join(normalized.split())
    return normalized


# Module-level LRU cache for title generation
# Using a wrapper to enable cache on instance method
_title_cache: Dict[str, List[str]] = {}
_title_cache_order: List[str] = []  # For LRU tracking


def clear_title_cache() -> None:
    """Clear in-memory title generation cache."""
    global _title_cache, _title_cache_order
    _title_cache.clear()
    _title_cache_order.clear()


class SearchModule:
    """
    Mixin for RAGSystem that handles core search functionality.
    Includes:
    - ZIM archive management
    - Title generation (LLM + Heuristic)
    - Shotgun retrieval logic
    - Index building
    """

    def _generate_candidate_titles(self, query: str) -> List[str]:
        """
        [ZERO-INDEX CORE]
        Asks the LLM to generate valid Wikipedia/ZIM article titles.
        Uses dedicated tiny model for fast title generation with LRU caching.
        """
        global _title_cache, _title_cache_order

        # Check cache first
        cache_key = _normalize_query_for_cache(query)
        if cache_key in _title_cache:
            debug_print(f"Title cache HIT for: {cache_key[:50]}...")
            # Move to end of LRU order
            _title_cache_order.remove(cache_key)
            _title_cache_order.append(cache_key)
            return _title_cache[cache_key].copy()

        # 1. HEURISTIC PROPER NOUN EXTRACTION (From Query)
        clean_q = query.strip(string.punctuation)
        words = clean_q.split()

        proper_nouns = []
        current_phrase = []
        for w in words:
            # Simple check for capitalized words that aren't common stopwords
            if w and w[0].isupper() and w.lower() not in ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 'explain']:
                current_phrase.append(w)
            else:
                if current_phrase:
                    proper_nouns.append("_".join(current_phrase))
                    current_phrase = []
        if current_phrase:
            proper_nouns.append("_".join(current_phrase))

        # Try to find quoted strings as well
        quotes = re.findall(r'"([^"]*)"', query)
        for q in quotes:
            proper_nouns.append(q.replace(' ', '_'))

        # 2. LLM GENERATION
        # Use dedicated tiny model for title generation (fast, minimal VRAM)
        # Falls back to default model if tiny model unavailable
        llm = None
        title_gen_model = getattr(config, 'TITLE_GEN_MODEL', None)
        models_to_try = [title_gen_model, config.MODEL_QWEN_1_5B, config.DEFAULT_MODEL]
        models_to_try = [m for m in models_to_try if m]  # Filter None

        for model_name in models_to_try:
            try:
                llm = ModelManager.get_model(model_name)
                if llm:
                    debug_print(f"Title generation using model: {model_name}")
                    break
            except Exception as e:
                debug_print(f"Model {model_name} failed: {e}")
        
        if not llm:
            debug_print("All models failed, using heuristics only")
            return [query.replace(" ", "_"), query.title().replace(" ", "_")] + proper_nouns
        
        system_msg = (
            "You are a Wikipedia title generator. Given a user question, output 8-10 Wikipedia article titles "
            "that would contain the answer. Follow these rules:\n"
            "1. Output exact titles using snake_case (e.g. 'United_States_declaration').\n"
            "2. If the term is ambiguous, ALWAYS include the '(disambiguation)' page.\n"
            "3. If the query is technical (e.g. Linux, Kernel, CPU), include specific technical article titles.\n"
            "4. IMPORTANT: Do NOT include question phrases (e.g. NO 'Who_was_X', NO 'What_is_X').\n"
            "5. IMPORTANT: Only use Wikipedia-style article names.\n"
            "6. Output one title per line, nothing else."
        )
        user_msg = f"Question: {query}"
        
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=200,
                temperature=0.3
            )
            raw_content = response['choices'][0]['message']['content']
            
            # 3. ROBUST PARSING & VALIDATION
            titles = []
            for line in raw_content.split('\n'):
                # Strip bullets, numbers, asterisks, etc.
                t = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                t = t.strip('"').strip("'").strip('_')
                if not t: continue
                
                t_lower = t.lower()
                
                # VALIDATION FILTERS
                # Skip question patterns
                if t_lower.startswith(('who_', 'what_', 'how_', 'why_', 'where_', 'when_', 'is_', 'can_', 'will_', 'explain_', 'which_')):
                    continue
                # Skip obvious query-like natural language phrasing
                if any(x in t_lower for x in ['_did_', '_was_', '_is_', '_the_creator_', '_attend_', '_born_in_']):
                    continue
                # Skip overly long titles
                if len(t) > 80:
                    continue
                # Filter for Wikipedia-style (no spaces, basic punctuation only)
                if ' ' in t:
                    continue
                    
                titles.append(t)
            
            # ENTITY PREFIX EXTRACTION: From "Tupac_Shakur_Murder_Case", also try "Tupac_Shakur"
            prefix_candidates = []
            for t in titles:
                parts = t.split('_')
                if len(parts) >= 2:
                    prefix_candidates.append('_'.join(parts[:2]))
                if len(parts) >= 1:
                    prefix_candidates.append(parts[0])
            
            # 4. FINAL COMBINATION & DEDUP
            heuristic = [
                query.replace(" ", "_"),
                query.title().replace(" ", "_"),
            ] + proper_nouns

            # Domain-Specific Expansion (Heuristic Suffixes)
            q_lower = query.lower()
            clean_query = query.translate(str.maketrans('', '', string.punctuation))

            # Tech/Linux domain
            if any(k in q_lower for k in ['linux', 'kernel', 'os', 'operating system', 'cpu', 'memory']):
                query_words = [w for w in clean_query.split() if len(w) > 4 and w.lower() not in ['what', 'where', 'when', 'which', 'about', 'between']]

                for base in list(set(query_words) | set(titles[:2])):
                    base_cap = base.capitalize()
                    heuristic.append(f"{base_cap}_kernel")
                    heuristic.append(f"{base_cap}_architecture")
                    heuristic.append(f"{base_cap}_(operating_system)")

            # Superlative queries: "largest X", "biggest Y", "longest Z", etc.
            # Extract the noun and try direct articles + list articles
            superlative_match = re.search(
                r'\b(largest|biggest|smallest|longest|shortest|tallest|deepest|highest|fastest|oldest|youngest|richest|poorest)\s+(\w+)',
                q_lower
            )
            if superlative_match:
                superlative = superlative_match.group(1)
                noun = superlative_match.group(2)
                noun_cap = noun.capitalize()
                # Handle common irregular plurals
                irregular_plurals = {
                    'city': 'cities', 'country': 'countries', 'company': 'companies',
                    'army': 'armies', 'baby': 'babies', 'family': 'families',
                    'man': 'men', 'woman': 'women', 'child': 'children',
                    'person': 'people', 'foot': 'feet', 'tooth': 'teeth',
                }
                if noun in irregular_plurals:
                    noun_plural = irregular_plurals[noun]
                elif noun.endswith('y') and len(noun) > 1 and noun[-2] not in 'aeiou':
                    noun_plural = noun[:-1] + 'ies'
                elif noun.endswith('s'):
                    noun_plural = noun
                else:
                    noun_plural = noun + 's'

                # Direct noun articles
                heuristic.append(noun_cap)
                heuristic.append(noun_cap + 's')

                # List articles (Wikipedia often has "List of X by Y")
                size_mappings = {
                    'largest': 'size', 'biggest': 'size', 'smallest': 'size',
                    'longest': 'length', 'shortest': 'length',
                    'tallest': 'height', 'deepest': 'depth', 'highest': 'elevation',
                    'fastest': 'speed', 'oldest': 'age', 'youngest': 'age',
                }
                metric = size_mappings.get(superlative, 'size')
                heuristic.append(f"List_of_{noun_plural}_by_{metric}")

                # For geography: "largest ocean" -> try specific ones
                if noun in ['ocean', 'sea', 'lake', 'river', 'mountain', 'country', 'city', 'island', 'desert']:
                    heuristic.append(f"List_of_{noun_plural}")
                    heuristic.append(f"List_of_{noun_plural}_by_area")
                    heuristic.append(f"List_of_{noun_plural}_by_population")

            # Legal domain expansion
            if any(k in q_lower for k in ['contract', 'law', 'legal', 'court', 'rights', 'liability', 'tort', 'statute']):
                legal_terms = ['contract', 'tort', 'liability', 'rights', 'court', 'statute', 'law']
                for term in legal_terms:
                    if term in q_lower:
                        term_cap = term.capitalize()
                        heuristic.append(term_cap)
                        heuristic.append(f"{term_cap}_law")
                        heuristic.append(f"{term_cap}_(law)")

                # Specific legal concept patterns
                if 'elements' in q_lower and 'contract' in q_lower:
                    heuristic.append("Contract")
                    heuristic.append("Contract_law")
                    heuristic.append("Elements_of_a_contract")

            # "What is X" / "What are X" - extract X directly
            whatis_match = re.search(r'\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:a\s+)?(.+?)(?:\?|$)', q_lower)
            if whatis_match:
                subject = whatis_match.group(1).strip()
                # Clean up and convert to title
                subject_clean = subject.translate(str.maketrans('', '', string.punctuation))
                subject_title = subject_clean.replace(' ', '_').title()
                heuristic.append(subject_title)
                # Also try without "the"
                subject_no_the = re.sub(r'^the\s+', '', subject_clean)
                if subject_no_the != subject_clean:
                    heuristic.append(subject_no_the.replace(' ', '_').title())
            
            all_candidates = titles + prefix_candidates + heuristic
            
            final_titles = []
            seen = set()
            for t in all_candidates:
                if t and t not in seen and len(t) >= 3:
                    final_titles.append(t)
                    seen.add(t)

            debug_print(f"Title Candidates: {final_titles}")

            # Store in cache
            self._cache_titles(cache_key, final_titles)

            return final_titles

        except Exception as e:
            debug_print(f"Title Generation Failed: {e}")
            fallback = [query.replace(" ", "_"), query.title().replace(" ", "_")] + proper_nouns
            # Cache fallback results too
            self._cache_titles(cache_key, fallback)
            return fallback

    def _cache_titles(self, cache_key: str, titles: List[str]) -> None:
        """Store titles in LRU cache, evicting oldest if at capacity."""
        global _title_cache, _title_cache_order

        cache_size = getattr(config, 'TITLE_CACHE_SIZE', 1000)

        # Evict oldest entries if at capacity
        while len(_title_cache_order) >= cache_size:
            oldest_key = _title_cache_order.pop(0)
            _title_cache.pop(oldest_key, None)

        _title_cache[cache_key] = titles.copy()
        _title_cache_order.append(cache_key)


    def get_zim_archive(self, zim_path: str):
        """
        Get ZIM archive handle with lazy loading, caching, and connection pooling.
        Prevents repeated archive opens (~100-500ms each) and limits file descriptors.
        """
        if not zim_path:
            return None

        abs_path = os.path.abspath(zim_path)

        # Track access order for LRU eviction
        if not hasattr(self, '_zim_access_order'):
            self._zim_access_order = []

        # If already open, update access order and return
        if abs_path in self.zim_archives:
            # Move to end of access order (most recently used)
            if abs_path in self._zim_access_order:
                self._zim_access_order.remove(abs_path)
            self._zim_access_order.append(abs_path)
            return self.zim_archives[abs_path]

        # Need to open new archive - check pool limit first
        max_pool_size = getattr(config, 'ZIM_POOL_MAX_SIZE', 5)

        while len(self.zim_archives) >= max_pool_size and self._zim_access_order:
            # Evict least recently used archive
            lru_path = self._zim_access_order.pop(0)
            if lru_path in self.zim_archives:
                debug_print(f"ZIM Pool: Evicting LRU archive: {os.path.basename(lru_path)}")
                # libzim.Archive doesn't have explicit close, but removing reference allows GC
                del self.zim_archives[lru_path]

        # Open the requested archive
        debug_print(f"Opening ZIM archive (pooled, {len(self.zim_archives)+1}/{max_pool_size}): {os.path.basename(abs_path)}")
        try:
            self.zim_archives[abs_path] = libzim.Archive(abs_path)
            self._zim_access_order.append(abs_path)
        except Exception as e:
            print(f"Failed to open ZIM: {abs_path}: {e}")
            return None

        return self.zim_archives[abs_path]

    def build_index(self, zim_path: str = None, zim_paths: List[str] = None, limit: int = None, batch_size: int = 1000):
        """
        Build UNIFIED Semantic Title Index from one or more ZIM files.
        Each title entry stores its source_zim for retrieval routing.
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if not self.encoder:
            try:
                self.encoder = SentenceTransformer(self.model_name, device=device)
            except Exception as e:
                print(f"Failed to load encoder for indexing: {e}")
                return
        
        # Determine which ZIMs to index
        paths_to_index = []
        if zim_paths:
            paths_to_index = [os.path.abspath(p) for p in zim_paths]
        elif zim_path:
            paths_to_index = [os.path.abspath(zim_path)]
        else:
            paths_to_index = self.zim_paths  # Use discovered ZIMs
        
        if not paths_to_index:
            print("Error: No ZIM files to index.")
            return
        
        print(f"\n{'='*60}")
        print(f"BUILDING UNIFIED TITLE INDEX FOR {len(paths_to_index)} ZIM FILE(S)")
        print(f"{'='*60}")
            
        # Initialize UNIFIED Title FAISS
        self.title_faiss_index = faiss.IndexFlatIP(384)
        self.title_metadata = []
        
        total_indexed = 0
        
        for zim_file in paths_to_index:
            zim_name = os.path.basename(zim_file)
            print(f"\nScanning: {zim_name}")
            
            try:
                zim = libzim.Archive(zim_file)
            except Exception as e:
                print(f"  ERROR: Failed to open {zim_name}: {e}")
                continue
            
            total_entries = zim.entry_count
            print(f"  Total entries: {total_entries}")
            
            zim_limit = limit if limit else total_entries
            
            titles = []
            paths = []
            source_zims = []
            
            count = 0
            for i in range(total_entries):
                if count >= zim_limit:
                    break
                    
                entry = zim.get_entry_by_index(i)
                # Filter for articles in namespace 'A'
                if entry.path.startswith("A/"):
                    titles.append(entry.title)
                    paths.append(entry.path)
                    source_zims.append(zim_file)  # Tag with source!
                    
                    # Batch processing
                    if len(titles) >= batch_size:
                        embeddings = self.encoder.encode(titles)
                        faiss.normalize_L2(embeddings)
                        self.title_faiss_index.add(embeddings)
                        
                        # Store meta WITH source_zim
                        for j, title in enumerate(titles):
                             self.title_metadata.append({
                                 'title': title,
                                 'path': paths[j],
                                 'source_zim': source_zims[j]  # Critical for multi-ZIM!
                             })
                             
                        titles = []
                        paths = []
                        source_zims = []
                        print(f"  Indexed {len(self.title_metadata)} titles...")
                        
                    count += 1
            
            # Final batch for this ZIM
            if titles:
                embeddings = self.encoder.encode(titles)
                faiss.normalize_L2(embeddings)
                self.title_faiss_index.add(embeddings)
                for j, title in enumerate(titles):
                     self.title_metadata.append({
                         'title': title,
                         'path': paths[j],
                         'source_zim': source_zims[j]
                     })
            
            zim_count = count
            total_indexed += zim_count
            print(f"  Completed: {zim_count} titles from {zim_name}")
        
        print(f"\n{'='*60}")
        print(f"UNIFIED INDEX COMPLETE: {total_indexed} titles from {len(paths_to_index)} ZIM(s)")
        print(f"{'='*60}")
        
        # Save unified indices
        print("Saving unified indices...")
        try:
            faiss.write_index(self.title_faiss_index, self.title_faiss_path)
            with open(self.title_meta_path, 'wb') as f:
                pickle.dump(self.title_metadata, f)
            print("Done.")
        except Exception as e:
            print(f"Error saving index: {e}")

    def retrieve(self, query: str, top_k: int = 5, mode: str = "FACTUAL", rebound_depth: int = 0, extra_terms: List[str] = None) -> List[Dict]:
        """
        Main retrieval entry point.
        
        If USE_ORCHESTRATION is enabled, delegates to retrieve_with_orchestration()
        for signal-based dynamic processing. Otherwise uses traditional linear pipeline.
        """
        # Check if orchestration is enabled
        # Mixin assumes retrieve_with_orchestration is available (via OrchestrationModule)
        if config.USE_ORCHESTRATION and not extra_terms and rebound_depth == 0:
            debug_print("🧠 Using ORCHESTRATED retrieval")
            if hasattr(self, 'retrieve_with_orchestration'):
                return self.retrieve_with_orchestration(query, top_k)
            else:
                 debug_print("⚠ Orchestration module missing, falling back to legacy")
        
        # Otherwise, use traditional zero-index retrieval
        debug_print("📚 Using TRADITIONAL retrieval")
        
        debug_print("-" * 70)
        debug_print(f"ZERO-INDEX RETRIEVAL: '{query}'")
        
        # 1. Generate Candidates
        candidates = self._generate_candidate_titles(query)
        if extra_terms:
            candidates.extend(extra_terms)
            
        final_results = []
        seen_titles = set()
        
        # 2. Shotgun Search across all ZIMs
        for title_guess in candidates:
            # Normalize title for display check (simple dedup)
            simple_title = title_guess.replace('_', ' ')
            if simple_title in seen_titles:
                continue
            
            # Try variations to find a hit (modern ZIMs often omit A/ prefix)
            # Optimized: reduced from 7 to 3 most effective variations
            base_title = title_guess.replace(' ', '_')
            variations = [
                base_title.capitalize(),                # Most common: Photosynthesis
                base_title,                             # As-is: photosynthesis
                base_title.title(),                     # Title case: Photo_Synthesis
            ]
            
            found_hit = False
            for zim_path in self.zim_paths:
                zim = self.get_zim_archive(zim_path)
                if not zim: continue
                
                for path_var in variations:
                    try:
                        entry = zim.get_entry_by_path(path_var)
                        if entry:
                            # Resolve Redirects
                            if entry.is_redirect:
                                try:
                                    entry = entry.get_redirect_entry()
                                    if not entry:
                                        continue
                                    debug_print(f"    Resolved redirect to: {entry.path}")
                                except Exception as e:
                                    debug_print(f"    Failed to resolve redirect: {e}")
                                    continue

                            # Process Resolved Entry
                            if not entry.is_redirect:
                                item = entry.get_item()
                                debug_print(f"    Mimetype: {item.mimetype}")
                                if item.mimetype == 'text/html':
                                    content = item.content.tobytes().decode('utf-8', errors='ignore')
                                    text_content = TextProcessor.clean_text(content)
                                    
                                    final_results.append({
                                        'text': text_content[:6000],
                                        'metadata': {
                                            'title': entry.title,
                                            'path': path_var,
                                            'source_zim': zim_path
                                        },
                                        'score': 10.0,
                                        'search_context': {'entities': candidates}
                                    })
                                    
                                    seen_titles.add(simple_title)
                                    debug_print(f"  HIT: '{entry.title}' in {os.path.basename(zim_path)}")
                                    found_hit = True
                                    break
                    except Exception as e:
                        # Only log for the first few ZIMs to avoid spam
                        pass
                
                # Fallback: Try get_entry_by_title if path lookup failed
                if not found_hit and 'wikipedia' in os.path.basename(zim_path).lower():
                    try:
                        entry = zim.get_entry_by_title(title_guess)
                        
                        # Resolve Redirects (Title lookup)
                        if entry and entry.is_redirect:
                             try:
                                 entry = entry.get_redirect_entry()
                             except:
                                 pass

                        if entry and not entry.is_redirect:
                            item = entry.get_item()
                            if item.mimetype == 'text/html':
                                content = item.content.tobytes().decode('utf-8', errors='ignore')
                                text_content = TextProcessor.clean_text(content)
                                
                                final_results.append({
                                    'text': text_content[:6000],
                                    'metadata': {
                                        'title': entry.title,
                                        'path': entry.path,
                                        'source_zim': zim_path
                                    },
                                    'score': 10.0,
                                    'search_context': {'entities': candidates}
                                })
                                
                                seen_titles.add(simple_title)
                                debug_print(f"  HIT (by title): '{entry.title}' in {os.path.basename(zim_path)}")
                                found_hit = True
                    except:
                        pass
                        
                if found_hit: break
        
        # 3. Sort by relevance order (LLM order + heuristic order) is implicit
        # We assume the first LLM guesses are best.
        
        debug_print(f"Zero-Index Search found {len(final_results)} direct matches.")
        
        # 4. Joint Processing (Refinement)
        # If we have joints enabled, run BATCH FactRefinement (1 LLM call instead of 3)
        if self.use_joints and getattr(self, 'fact_joint', None) and final_results:
            debug_print(f"[JOINT 4 INPUT] Batch refining facts for {min(len(final_results), 3)} results...")
            try:
                # Use batch extraction - single LLM call for all articles
                batch_results = self.fact_joint.refine_facts_batch(query, final_results[:3])

                # Map results back to articles
                for batch_item in batch_results:
                    title = batch_item.get('title', '')
                    facts = batch_item.get('facts', [])
                    if not facts:
                        continue

                    # Find matching result by title
                    for res in final_results[:3]:
                        if res.get('metadata', {}).get('title') == title:
                            res['extracted_facts'] = facts
                            debug_print(f"[JOINT 4 OUTPUT] Extracted {len(facts)} facts from {title}")
                            # Append facts to text for visibility
                            facts_str = "\n".join([f"- {f}" for f in facts])
                            res['text'] = f"*** VERIFIED FACTS ***\n{facts_str}\n\n*** SOURCE CONTENT ***\n{res['text']}"
                            break

            except Exception as e:
                debug_print(f"Joint 4 batch failed, falling back to individual: {e}")
                # Fallback to individual extraction
                for res in final_results[:3]:
                    try:
                        facts = self.fact_joint.refine_facts(query, res['text'])
                        if facts:
                            res['extracted_facts'] = facts
                            facts_str = "\n".join([f"- {f}" for f in facts])
                            res['text'] = f"*** VERIFIED FACTS ***\n{facts_str}\n\n*** SOURCE CONTENT ***\n{res['text']}"
                    except Exception as e2:
                        debug_print(f"Joint 4 individual failed: {e2}")

        return final_results[:top_k]

    def search_by_title(self, query: str, zim_path: str = None, full_text: bool = False) -> List[Dict]:
        """
        Search for articles by title using UNIFIED Semantic Title Index.
        Multi-ZIM aware: fetches content from the correct source ZIM.
        """
        results = []
        
        try:
            # 1. Semantic Title Search (Preferred - uses UNIFIED index)
            if self.title_faiss_index and self.title_metadata and self.encoder:
                 q_emb = self.encoder.encode([query])
                 faiss.normalize_L2(q_emb)
                 D, I = self.title_faiss_index.search(q_emb, 20)
                 
                 for i, idx in enumerate(I[0]):
                     if idx != -1 and idx < len(self.title_metadata):
                         meta = self.title_metadata[int(idx)]
                         
                         # Get the correct ZIM archive for this result
                         source_zim = meta.get('source_zim')
                         if not source_zim:
                             # Legacy index without source_zim, fall back
                             source_zim = self.zim_path or (self.zim_paths[0] if self.zim_paths else None)
                         
                         if not source_zim:
                             continue
                         
                         zim = self.get_zim_archive(source_zim)
                         if not zim:
                             continue
                         
                         try:
                             entry = zim.get_entry_by_path(meta['path'])
                             item = entry.get_item()
                             content = item.content.tobytes().decode('utf-8', errors='ignore')
                             results.append({
                                 'text': content,
                                 'metadata': {
                                     'title': meta['title'],
                                     'path': meta['path'],
                                     'source_zim': source_zim
                                 },
                                 'score': float(D[0][i])
                             })
                         except Exception:
                             continue
                 return results
            
            # 2. Heuristic Path Fallback (searches ALL ZIMs)
            debug_print(f"No title index, using heuristic fallback across {len(self.zim_paths)} ZIM(s)")
            guess_title = query.replace(" ", "_")
            paths_to_try = [
                f"A/{guess_title}", 
                f"A/{guess_title.title()}",
                f"A/{query}"
            ]
            
            for search_zim in self.zim_paths:
                zim = self.get_zim_archive(search_zim)
                if not zim:
                    continue
                
                for p in paths_to_try:
                    try:
                        entry = zim.get_entry_by_path(p)
                        if entry:
                            item = entry.get_item()
                            content = item.content.tobytes().decode('utf-8', errors='ignore')
                            results.append({
                                 'text': content,
                                 'metadata': {
                                     'title': entry.title,
                                     'path': p,
                                     'source_zim': search_zim
                                 },
                                 'score': 100.0
                            })
                            # Found in this ZIM, move to next path
                            break
                    except Exception:
                        pass
            
            return results

        except Exception as e:
            print(f"Search failed: {e}")
            return []
