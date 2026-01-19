
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RAG System Implementation.
Handles indexing, retrieval, and context generation for the chatbot.
"""

import os
import sys
import pickle
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Set

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError:
    # Zero-Index mode doesn't strictly require these if we aren't using them
    # but we keep imports for compatibility or future re-enablement
    faiss = None
    SentenceTransformer = None
    BM25Okapi = None

import libzim

from chatbot import config
from chatbot.debug_utils import debug_print
from chatbot.text_processing import TextProcessor

class RAGSystem:
    def __init__(self, index_dir: str = "data/indices", zim_path: str = None, zim_paths: List[str] = None, load_existing: bool = True):
        self.index_dir = index_dir
        self.encoder = None
        self.model_name = 'all-MiniLM-L6-v2'
        
        # === MULTI-ZIM SUPPORT ===
        # Discover all ZIM files and maintain lazy-loaded archive cache
        import glob
        self.zim_paths: List[str] = []
        self.zim_archives: Dict[str, any] = {}  # Lazy cache: {path: Archive}
        
        # Priority: explicit zim_paths > explicit zim_path > auto-discover
        if zim_paths:
            self.zim_paths = [os.path.abspath(p) for p in zim_paths]
        elif zim_path:
            self.zim_paths = [os.path.abspath(zim_path)]
        else:
            # Auto-discover all ZIM files in current directory
            discovered = glob.glob("*.zim")
            self.zim_paths = [os.path.abspath(p) for p in discovered]
        
        if self.zim_paths:
            print(f"Multi-ZIM Mode: Found {len(self.zim_paths)} ZIM file(s)")
            for zp in self.zim_paths:
                print(f"  - {os.path.basename(zp)}")
        else:
            print("Warning: No ZIM files found.")
        
        # Legacy compatibility
        self.zim_path = self.zim_paths[0] if self.zim_paths else None
        self.zim_archive = None  # Deprecated, use get_zim_archive()
        
        # In-memory storage
        self.faiss_index = None # JIT Index (Vectors)
        self.documents = []     # Metadata
        self.doc_chunks = []    # Text Chunks
        
        # State Tracking
        self.indexed_paths: Set[str] = set()
        self._next_doc_id = 0
        self._chunk_id = 0     # Global chunk ID counter
        
        self.bm25 = None
        self.tokenized_corpus = [] # For BM25
        
        # Title Indices (Pre-computed, UNIFIED across all ZIMs)
        self.title_faiss_index = None
        self.title_metadata = None  # List[{title, path, source_zim}]
        
        # Paths
        os.makedirs(index_dir, exist_ok=True)
        self.faiss_path = os.path.join(index_dir, "content_index.faiss")
        self.meta_path = os.path.join(index_dir, "content_meta.pkl")
        self.bm25_path = os.path.join(index_dir, "content_bm25.pkl")
        
        self.title_faiss_path = os.path.join(index_dir, "title_index.faiss")
        self.title_meta_path = os.path.join(index_dir, "title_meta.pkl")

        # Initialize SentenceTransformer early (lazy load usually, but we need it for everything)
        try:
            # Move encoder to CPU to save VRAM for the main LLM.
            # This encoder is still used for reranking or other semantic tasks, not for primary indexing in Zero-Index mode.
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")

        # Multi-Joint System Configuration
        self.use_joints = config.USE_JOINTS
        # We will reuse the EntityExtractor logic but specifically prompt for TITLES
        
        # Load Existing Content Indices - DEPRECATED / DISABLED for Zero-Index Mode
        # The whole point is to NOT need these.
        print("Zero-Index Mode: Skipping index loading.")
        
        # Initialize Joint System (if enabled)
        if self.use_joints:
            debug_print("Initializing multi-joint RAG system...")
            try:
                from chatbot.joints import EntityExtractorJoint, ArticleScorerJoint, CoverageVerifierJoint, ChunkFilterJoint, FactRefinementJoint, ComparisonJoint, MultiHopResolverJoint
                self.entity_joint = EntityExtractorJoint()
                self.resolver_joint = MultiHopResolverJoint(model=config.MULTI_HOP_JOINT_MODEL)  # Joint 0.5: Multi-Hop Resolver (Smart 7B)
                self.scorer_joint = ArticleScorerJoint()
                self.coverage_joint = CoverageVerifierJoint()  # Joint 2.5: Coverage Verification
                self.comparison_joint = ComparisonJoint(model=config.COMPARISON_JOINT_MODEL)    # Joint 3.5: Comparison Synthesis (Smart 7B)
                self.filter_joint = ChunkFilterJoint()
                self.fact_joint = FactRefinementJoint()
                debug_print("Joint system initialized successfully (including Multi-Hop Resolver)")
            except Exception as e:
                debug_print(f"Failed to initialize joints: {e}")
                debug_print("Falling back to semantic search")
                self.use_joints = False

    def _generate_candidate_titles(self, query: str) -> List[str]:
        """
        [ZERO-INDEX CORE]
        Asks the LLM to generate valid Wikipedia/ZIM article titles.
        Tries smart model first, falls back to fast model if loading fails.
        """
        from chatbot import config
        from chatbot.model_manager import ModelManager
        
        # Try smart model first, fall back to fast model
        llm = None
        for model_name in [config.DEFAULT_MODEL, config.ENTITY_JOINT_MODEL]:
            try:
                llm = ModelManager.get_model(model_name)
                if llm:
                    break
            except Exception as e:
                debug_print(f"Model {model_name} failed: {e}")
        
        if not llm:
            debug_print("All models failed, using heuristics only")
            return [query.replace(" ", "_"), query.title().replace(" ", "_")]
        
        system_msg = (
            "You are a Wikipedia title generator. Given a user question, output 8-10 Wikipedia article titles "
            "that would contain the answer. Follow these rules:\n"
            "1. Output exact titles using snake_case (e.g. 'United_States_declaraton').\n"
            "2. If the term is ambiguous, ALWAYS include the '(disambiguation)' page.\n"
            "3. If the query is technical (e.g. Linux, Kernel, CPU), include specific technical article titles (e.g. 'Linux_kernel', 'Process_management').\n"
            "4. Output one title per line, nothing else."
        )
        user_msg = f"Question: {query}"
        
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=200,
                temperature=0.3  # Slightly creative to get variations
            )
            raw_content = response['choices'][0]['message']['content']
            
            # Robust parsing: strip bullets, numbers, asterisks, parenthetical notes
            import re
            titles = []
            for line in raw_content.split('\n'):
                # Remove common prefixes: "1.", "- ", "* ", "*   ", etc.
                t = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                # Remove quotes and leading/trailing underscores
                t = t.strip('"').strip("'").strip('_')
                # Skip garbage (starts with punctuation, too short, etc.)
                if not t or len(t) < 3 or t.startswith('(') or t.startswith('['):
                    continue
                # Skip question-like titles (Who_is, What_is, etc.)
                if t.lower().startswith(('who_', 'what_', 'how_', 'why_', 'where_', 'when_')):
                    continue
                # Accept valid-looking titles
                if ' ' not in t and len(t) < 100:
                    titles.append(t)
            
            # ENTITY PREFIX EXTRACTION: From "Tupac_Shakur_Murder_Case", also try "Tupac_Shakur" and "Tupac"
            prefix_candidates = []
            for t in titles:
                parts = t.split('_')
                if len(parts) >= 2:
                    # Try first 2 words (common for person names)
                    prefix_candidates.append('_'.join(parts[:2]))
                if len(parts) >= 1:
                    # Try first word alone
                    prefix_candidates.append(parts[0])
            
            # HEURISTIC ENTITY EXTRACTION FROM QUERY (Fallback for weak LLM title gen)
            # Extract proper nouns / capitalized phrases from query
            import string
            clean_q = query.strip(string.punctuation)
            words = clean_q.split()
            cap_phrases = []
            current_phrase = []
            
            for w in words:
                if w[0].isupper() and w.lower() not in ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'a', 'an']:
                    current_phrase.append(w)
                else:
                    if current_phrase:
                        cap_phrases.append("_".join(current_phrase))
                        current_phrase = []
            if current_phrase:
                cap_phrases.append("_".join(current_phrase))
            
            # Add heuristic fallbacks
            heuristic = [
                query.replace(" ", "_"),
                query.title().replace(" ", "_"),
            ] + cap_phrases

            # Domain-Specific Expansion (Heuristic Suffixes)
            # Use QUERY terms for expansion to catch things the LLM might miss
            q_lower = query.lower()
            if any(k in q_lower for k in ['linux', 'kernel', 'os', 'operating system', 'cpu', 'memory']):
                # Extract potential tech keywords from query (words > 4 chars)
                import string
                clean_query = query.translate(str.maketrans('', '', string.punctuation))
                query_words = [w for w in clean_query.split() if len(w) > 4 and w.lower() not in ['what', 'where', 'when', 'which', 'about', 'between']]
                
                # Also consider base terms from LLM titles
                base_terms = set(query_words)
                for t in titles:
                    if len(t.split('_')) == 1:
                        base_terms.add(t)

                for base in base_terms:
                    # Title-case the base for Wikipedia compatibility
                    base_cap = base.capitalize()
                    
                    heuristic.append(f"{base_cap}_kernel")
                    heuristic.append(f"{base_cap}_architecture")
                    heuristic.append(f"{base_cap}_system")
                    heuristic.append(f"{base_cap}_(operating_system)")
            
            # Combine: LLM titles first, then prefixes, then heuristics
            all_candidates = titles + prefix_candidates + heuristic
            
            # Dedupe while preserving order
            final_titles = []
            seen = set()
            for t in all_candidates:
                if t and t not in seen and len(t) >= 3:
                    final_titles.append(t)
                    seen.add(t)
                    
            debug_print(f"Title Candidates: {final_titles}")
            return final_titles
            
        except Exception as e:
            debug_print(f"Title Generation Failed: {e}")
            return [query.replace(" ", "_"), query.title().replace(" ", "_")]

    def get_zim_archive(self, zim_path: str):
        """
        Get ZIM archive handle with lazy loading and caching.
        This prevents repeated archive opens which are expensive (~100-500ms each).
        """
        if not zim_path:
            return None
        
        abs_path = os.path.abspath(zim_path)
        
        if abs_path not in self.zim_archives:
            debug_print(f"Opening ZIM archive (cached): {os.path.basename(abs_path)}")
            try:
                self.zim_archives[abs_path] = libzim.Archive(abs_path)
            except Exception as e:
                print(f"Failed to open ZIM: {abs_path}: {e}")
                return None
        
        return self.zim_archives[abs_path]

    def build_index(self, zim_path: str = None, zim_paths: List[str] = None, limit: int = None, batch_size: int = 1000):
        """
        Build UNIFIED Semantic Title Index from one or more ZIM files.
        Each title entry stores its source_zim for retrieval routing.
        
        Args:
            zim_path: Single ZIM path (legacy, for backwards compat)
            zim_paths: List of ZIM paths (preferred for multi-ZIM)
            limit: Max titles per ZIM (None = all)
            batch_size: Embedding batch size
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, device=device)
        
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
        faiss.write_index(self.title_faiss_index, self.title_faiss_path)
        with open(self.title_meta_path, 'wb') as f:
            pickle.dump(self.title_metadata, f)
        print("Done.")

    def retrieve(self, query: str, top_k: int = 5, mode: str = "FACTUAL", rebound_depth: int = 0, extra_terms: List[str] = None) -> List[Dict]:
        """
        Zero-Index Retrieval Pipeline.
        1. Ask LLM for probable titles.
        2. 'Shotgun' check these titles against ALL ZIM files.
        3. Return hits.
        """
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
            base_title = title_guess.replace(' ', '_')
            variations = [
                base_title,                             # As-is: photosynthesis
                base_title.capitalize(),                # Cap: Photosynthesis
                base_title.title(),                     # Title: Photosynthesis
                f"A/{base_title}",                      
                f"A/{base_title.capitalize()}",
                title_guess,
                f"A/{title_guess}",
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
                                        'score': 100.0,
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
                                    'score': 100.0,
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
        # If we have joints enabled, run FactRefinement on the top results
        if self.use_joints and self.fact_joint and final_results:
             debug_print(f"[JOINT 4 INPUT] Refining facts for {len(final_results)} results...")
             for res in final_results[:3]: # Only refine top 3 to save time
                 try:
                     facts = self.fact_joint.refine_facts(query, res['text'])
                     if facts:
                         res['extracted_facts'] = facts
                         debug_print(f"[JOINT 4 OUTPUT] Extracted {len(facts)} facts from {res['metadata']['title']}")
                         # Append facts to text for visibility
                         facts_str = "\n".join([f"- {f}" for f in facts])
                         res['text'] = f"*** VERIFIED FACTS ***\n{facts_str}\n\n*** SOURCE CONTENT ***\n{res['text']}"
                 except Exception as e:
                     debug_print(f"Joint 4 failed: {e}")

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

if __name__ == "__main__":
    pass