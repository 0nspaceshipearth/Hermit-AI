import os
import sys
import shutil
import pickle
from typing import List, Dict, Optional, Tuple, Iterator
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
try:
    import libzim
except ImportError:
    print("WARNING: libzim not available - ZIM file functionality disabled")
    libzim = None
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Import config module for runtime DEBUG check
from chatbot import config

def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:RAG] {msg}", file=sys.stderr)


class TextProcessor:
    """Handles text extraction and chunking."""
    
    @staticmethod
    def extract_text(html_content: bytes) -> str:
        """Extract clean text from HTML."""
        try:
            if isinstance(html_content, str):
                text_content = html_content
            else:
                text_content = bytes(html_content).decode('utf-8', errors='ignore')
        except Exception:
            text_content = ""
            
        if not text_content:
            return ""

        try:
            soup = BeautifulSoup(text_content, 'html.parser')
            # Remove script and style elements
            for tag in ["script", "style", "header", "footer", "nav"]:
                for element in soup.find_all(tag):
                    element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            # print(f"BS4 Error: {e}")
            return ""
    @staticmethod
    def extract_renderable_text(html_content):
        """Extract text with markdown-like markers for headers and lists."""
        try:
            if isinstance(html_content, str):
                text_content = html_content
            else:
                text_content = bytes(html_content).decode('utf-8', errors='ignore')
            
            soup = BeautifulSoup(text_content, 'html.parser')
            
            # Kill clutter
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "meta"]):
                tag.extract()
            
            # Process headers
            for i in range(1, 7):
                for h in soup.find_all(f'h{i}'):
                    # Prefix with #, ##, etc.
                    prefix = '#' * i + ' '
                    if h.string:
                        h.string = prefix + h.get_text().strip() + '\n'
            
            # Process lists
            for li in soup.find_all('li'):
                if li.string:
                    li.string = '• ' + li.get_text().strip()
                else:
                    li.insert(0, '• ')

            # Process block elements to ensure newlines
            for tag in soup.find_all(['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                tag.append('\n')

            # Get text with separator
            text = soup.get_text(separator='', strip=True)
            
            # Post-processing cleanup: ensure max 2 newlines
            import re
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text
        except Exception as e:
            print(f"Error extracting renderable text: {e}")
            return f"Error rendering content: {e}"
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if not words:
            return []
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

class RAGSystem:
    """Core RAG System handling indexing and retrieval."""
    
    def __init__(self, index_dir: str = "data/index", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.encoder = None
        
        # Content Indices (for documents/chunks)
        self.faiss_index = None
        self.bm25 = None
        self.documents = [] # Metadata storage
        self.doc_chunks = [] # Actual text chunks
        self.indexed_paths = set() # Track what we have indexed
        
        # Title Indices (for JIT discovery)
        self.title_faiss_index = None
        self.title_metadata = None
        
        # Paths
        self.faiss_path = os.path.join(index_dir, "faiss.index")
        self.bm25_path = os.path.join(index_dir, "bm25.pkl")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")
        
        # Title Paths
        self.title_faiss_path = os.path.join(index_dir, "titles.faiss")
        self.title_meta_path = os.path.join(index_dir, "titles.pkl")
        
        # Joint System (initialized in load_resources if enabled)
        self.use_joints = config.USE_JOINTS
        self.entity_joint = None
        self.scorer_joint = None
        self.filter_joint = None
        
        # JIT Cache: {(zim_path, article_path): (chunks, embeddings)}
        self.jit_cache = {}
        
        # EXPERIMENTAL: Online Mode
        self.web_retriever = None
        self.online_mode = False

    def toggle_online_mode(self, enabled: bool):
        """Enable or disable experimental online mode."""
        self.online_mode = enabled
        if enabled and not self.web_retriever:
            try:
                from chatbot.web_rag import WebRetriever
                self.web_retriever = WebRetriever()
                print("Online mode ENABLED: WebRetriever initialized.")
            except Exception as e:
                print(f"Failed to initialize WebRetriever: {e}")
                self.online_mode = False
        else:
             print(f"Online mode set to: {enabled}")
        
    def load_resources(self):
        """Load models and indices if they exist."""
        print(f"Loading encoder: {self.model_name}...")
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                debug_print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                debug_print("CUDA not available, falling back to CPU")
                print("WARNING: CUDA not available for RAG. Embeddings will be slow.")
            
            print(f"Using device: {device}")
            self.encoder = SentenceTransformer(self.model_name, device=device)
        except Exception as e:
             debug_print(f"Failed to load encoder with explicit device: {e}")
             self.encoder = SentenceTransformer(self.model_name)
        
        # Load Content Indices
        if os.path.exists(self.faiss_path) and os.path.exists(self.bm25_path):
            print("Loading existing content indices...")
            self.faiss_index = faiss.read_index(self.faiss_path)
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_chunks = data['chunks']
                # Rebuild indexed set
                self.indexed_paths = {doc.get('path') for doc in self.documents if doc.get('path') is not None}
        else:
            print("No existing content indices found.")

        # Load Title Indices (Optional)
        if os.path.exists(self.title_faiss_path) and os.path.exists(self.title_meta_path):
            print("Loading Semantic Title Index...")
            try:
                self.title_faiss_index = faiss.read_index(self.title_faiss_path)
                with open(self.title_meta_path, 'rb') as f:
                    self.title_metadata = pickle.load(f)
                print(f"Loaded {len(self.title_metadata)} titles.")
            except Exception as e:
                print(f"Failed to load title index: {e}")
        
        # Initialize Joint System (if enabled)
        if self.use_joints:
            debug_print("Initializing multi-joint RAG system...")
            try:
                from chatbot.joints import EntityExtractorJoint, ArticleScorerJoint, ChunkFilterJoint, FactRefinementJoint
                self.entity_joint = EntityExtractorJoint()
                self.scorer_joint = ArticleScorerJoint()
                self.filter_joint = ChunkFilterJoint()
                self.fact_joint = FactRefinementJoint()
                debug_print("Joint system initialized successfully")
            except Exception as e:
                debug_print(f"Failed to initialize joints: {e}")
                debug_print("Falling back to semantic search")
                self.use_joints = False

    def build_index(self, zim_path: str, limit: int = None, batch_size: int = 1000):
        """Build FAISS and BM25 indices from ZIM file using batch processing."""
        # ... (Existing build_index logic - truncated for brevity but functionality preserved by keeping same structure if overwrite?)
        # WAIT - write_to_file overwrites. I must include the full logic.
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if not self.encoder:
            self.encoder = SentenceTransformer(self.model_name, device=device)
            
        os.makedirs(self.index_dir, exist_ok=True)
        
        print(f"Opening ZIM file: {zim_path}")
        zim = libzim.Archive(zim_path)
        
        # Initialize indices
        self.doc_chunks = []
        self.documents = []
        
        # Approximate dimension from model
        sample_emb = self.encoder.encode(["test"], device=device)
        dimension = sample_emb.shape[1]
        
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        count = 0
        total = zim.entry_count
        
        current_batch_chunks = []
        current_batch_meta = []
        
        print("Extracting and indexing documents in batches...")
        for i in tqdm(range(total)):
            try:
                entry = zim._get_entry_by_id(i)
                if entry.is_redirect:
                    continue
                    
                item = entry.get_item()
                if item.mimetype != 'text/html':
                    continue
                    
                text = TextProcessor.extract_text(item.content)
                if not text:
                    continue
                    
                doc_chunks = TextProcessor.chunk_text(text)
                for chunk in doc_chunks:
                    current_batch_chunks.append(chunk)
                    current_batch_meta.append({
                        'title': entry.title,
                        'path': entry.path,
                        'zim_index': i
                    })
                    
                count += 1
                
                # Process batch
                if len(current_batch_chunks) >= batch_size:
                    self._process_batch(current_batch_chunks, current_batch_meta)
                    current_batch_chunks = []
                    current_batch_meta = []
                    
                if limit and count >= limit:
                    break
            except Exception as e:
                pass

        # Process remaining
        if current_batch_chunks:
            self._process_batch(current_batch_chunks, current_batch_meta)

        # Save indices
        faiss.write_index(self.faiss_index, self.faiss_path)
        
        print("Building BM25 index...")
        tokenized_corpus = [chunk.split(" ") for chunk in self.doc_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
            
        with open(self.meta_path, 'wb') as f:
            pickle.dump({'documents': self.documents, 'chunks': self.doc_chunks}, f)
            
        print("Indexing complete.")

    def _process_batch(self, chunks: List[str], meta: List[Dict]):
        """Encode and index a batch of chunks."""
        if not chunks:
            return
            
        # Add to storage
        self.doc_chunks.extend(chunks)
        self.documents.extend(meta)
        
        # Encode
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = self.encoder.encode(chunks, batch_size=32, device=device, show_progress_bar=False, convert_to_numpy=True)
        
        # Add to FAISS
        self.faiss_index.add(embeddings.astype('float32'))

    def retrieve(self, query: str, top_k: int = 5, rebound_depth: int = 0, extra_terms: List[str] = None, mode: str = "FACTUAL") -> List[Dict]:
        """Hybrid retrieval with Just-In-Time indexing and Reranking.
        
        EPHEMERAL INDEXING: Each call creates a fresh JIT index to prevent
        state bleed between queries. Only articles relevant to THIS query
        are indexed and searched.
        
        Args:
            query: User query
            top_k: Number of results to return
            rebound_depth: Recursion depth for adaptive RAG (0 = initial, 1 = rebound)
            extra_terms: Additional search terms to force into the pipeline (used in rebound)
            mode: Operational mode (e.g., "FACTUAL", "CODE") to guide joint reasoning
        """
        if self.online_mode and self.web_retriever:
            return self.retrieve_web(query, top_k)

        debug_print("=" * 70)
        debug_print(f"RETRIEVE CALLED: query='{query}', top_k={top_k}")
        
        # === EPHEMERAL INDEX RESET ===
        # Reset JIT-specific state to ensure query isolation.
        # This prevents vectors from previous queries contaminating results.
        # SKIPS RESET if this is a rebound pass (we want to ADD to the index).
        if rebound_depth == 0:
            debug_print("EPHEMERAL RESET: Clearing JIT index state...")
            self.faiss_index = None  # Will be re-created if articles are found
            self.doc_chunks = []     # Fresh chunks for this query only
            self.documents = []      # Fresh metadata for this query only
            self.indexed_paths = set()  # Reset path tracking
            self._chunk_id = 0       # Reset chunk ID counter for metadata alignment
            debug_print("EPHEMERAL RESET COMPLETE: faiss_index=None, doc_chunks=[], documents=[], indexed_paths={}")
        else:
            debug_print(f"EPHEMERAL RESET SKIPPED (Rebound Depth: {rebound_depth}) - Maintaining index state.")
        
        # Note: BM25 is NOT reset - it's a pre-built corpus index (if loaded).
        # For JIT-only mode, BM25 will also be empty/None.
        debug_print(f"BM25 status (pre-built corpus): {self.bm25 is not None}")
        
        if not self.bm25:
            debug_print("Note: No pre-built BM25 index available (JIT-only mode)")
            pass 
        
        print(f"\nProcessing Query: '{query}'")
        
        # 0. JIT Indexing step
        debug_print("-" * 70)
        debug_print("PHASE 0: JUST-IN-TIME INDEXING")
        try:
            # 0a. Entity Extraction (Joint 1) - Now supports multiple entities
            entity_info = None
            search_terms = [query]
            is_comparison = False
            
            if self.use_joints and self.entity_joint:
                debug_print("0a. Entity extraction with Joint 1...")
                entity_info = self.entity_joint.extract(query)
                
                # Extract search terms from ALL entities (new multi-entity format)
                search_terms = []
                entities = entity_info.get('entities', [])
                is_comparison = entity_info.get('is_comparison', False)
                
                for entity in entities:
                    entity_name = entity.get('name', '')
                    if entity_name:
                        search_terms.append(entity_name)
                    # Also add aliases
                    search_terms.extend(entity.get('aliases', []))
                
                # Fallback if entities list is empty
                if not search_terms:
                    search_terms = [query]
                
                # REBOUND: Add extra terms if provided
                if extra_terms:
                    debug_print(f"Adding extra terms from rebound: {extra_terms}")
                    search_terms.extend(extra_terms)
                
                debug_print(f"Is comparison query: {is_comparison}")
                debug_print(f"Extracted {len(entities)} entities: {[e.get('name', '') for e in entities]}")
                debug_print(f"Search terms expanded to: {search_terms}")
            else:
                debug_print("0a. Skipping entity extraction (joints disabled)")
            
            # 0b. Title Search with expanded terms
            candidates = []
            seen_titles = set()
            
            # A. Keyword Search (using expanded terms if available)
            # For comparison queries, search more terms to ensure coverage of all entities
            debug_print("0b. Performing keyword-based title search...")
            debug_print(f"    search_terms: {search_terms}")
            max_terms = 6 if is_comparison else 3  # More terms for comparison queries
            for term in search_terms[:max_terms]:
                keyword_candidates = self.search_by_title(term, full_text=True)
                for c in keyword_candidates:
                    t = c['metadata']['title']
                    if t not in seen_titles:
                        candidates.append(c)
                        seen_titles.add(t)
                        if len(candidates) <= 10:  # Log first 10
                            debug_print(f"  - Added keyword candidate: '{t}'")
            
            debug_print(f"Keyword search returned {len(candidates)} total candidates")
            
            # B. Semantic Title Search
            debug_print("0c. Performing semantic title search...")
            semantic_candidates = self.search_by_embedding(query, top_k=5, full_text=True)
            debug_print(f"Semantic search returned {len(semantic_candidates)} candidates")
            for c in semantic_candidates:
                t = c['metadata']['title']
                if t not in seen_titles:
                    candidates.append(c)
                    seen_titles.add(t)
                    debug_print(f"  - Added semantic candidate: '{t}'")
                    print(f" [Semantic Title Match] Found: '{t}'")
            
            # 0c. Article Scoring (Joint 2)
            top_articles = []
            if self.use_joints and self.scorer_joint and entity_info and candidates:
                debug_print("0d. Article scoring with Joint 2...")
                candidate_titles = [c['metadata']['title'] for c in candidates]
                # For comparison queries, increase top_k to ensure we get articles for all entities
                scorer_top_k = 10 if is_comparison else 7
                debug_print(f"Using scorer top_k={scorer_top_k} (is_comparison={is_comparison})")
                scored_titles = self.scorer_joint.score(entity_info, candidate_titles, top_k=scorer_top_k)
                
                # Convert scored titles back to full candidate objects
                title_to_candidate = {c['metadata']['title']: c for c in candidates}
                for title, score in scored_titles:
                    if title in title_to_candidate:
                        top_articles.append(title_to_candidate[title])
                
                debug_print(f"Joint 2 selected top {len(top_articles)} articles")
            else:
                # Fallback: use all candidates (up to 10 for comparison, 7 otherwise)
                debug_print("0d. Skipping article scoring (joints disabled or no candidates)")
                fallback_k = 10 if is_comparison else 7
                top_articles = candidates[:fallback_k]

            if top_articles:
                article_titles = [a['metadata']['title'] for a in top_articles]
                debug_print(f"Articles selected for indexing: {article_titles}")
                print(f"Found Title Candidates: {article_titles}")
            else:
                debug_print("No article candidates found")
                print("No direct title matches found.")
            
            # 0e. JIT Indexing for top articles
            debug_print("Checking top articles for JIT indexing...")
            for idx, cand in enumerate(top_articles):
                path = cand['metadata'].get('path')
                title = cand['metadata'].get('title')
                debug_print(f"Article {idx+1}: path='{path}', already_indexed={path in self.indexed_paths if path else 'N/A'}")
                
                # Check if already indexed
                if path is not None and path not in self.indexed_paths:
                    debug_print(f"JIT INDEXING START for '{title}'")
                    
                    # Check Cache First
                    zim_path = cand['metadata'].get('source_zim', '')
                    cache_key = (zim_path, path)
                    debug_print(f"Checking cache for key: {cache_key}")
                    debug_print(f"Current cache keys: {list(self.jit_cache.keys())}")
                    
                    chunks = []
                    embeddings = None
                    
                    if cache_key in self.jit_cache:
                        print(f"JIT Indexing: '{title}' (From Cache)...")
                        debug_print(f"CACHE HIT for {cache_key}")
                        chunks, embeddings = self.jit_cache[cache_key]
                    else:
                        print(f"JIT Indexing: '{title}' (New Topic)...")
                        text = cand['text'] # Full text
                        debug_print(f"Full text length: {len(text)} chars")
                        chunks = TextProcessor.chunk_text(text)
                        debug_print(f"Chunked into {len(chunks)} chunks")
                        
                        if chunks:
                            import torch
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            debug_print(f"Using device: {device}")
                            if not self.encoder:
                                debug_print("Encoder not loaded, loading now...")
                                self.encoder = SentenceTransformer(self.model_name, device=device)
                            
                            debug_print("Encoding chunks...")
                            embeddings = self.encoder.encode(chunks, device=device, show_progress_bar=False)
                            debug_print(f"Embeddings shape: {embeddings.shape}")
                            
                            # Valid embeddings?
                            if embeddings is not None and len(embeddings) > 0:
                                self.jit_cache[cache_key] = (chunks, embeddings)
                    
                    if chunks and embeddings is not None:
                        if self.faiss_index is None:
                            dimension = embeddings.shape[1]
                            debug_print(f"Creating new FAISS index with dimension={dimension}")
                            self.faiss_index = faiss.IndexFlatL2(dimension)
                        
                        debug_print(f"Adding {len(embeddings)} embeddings to FAISS index...")
                        self.faiss_index.add(embeddings.astype('float32'))
                        
                        self.doc_chunks.extend(chunks)
                        for _ in chunks:
                            self.documents.append(cand['metadata'])
                        
                        self.indexed_paths.add(path)
                        debug_print(f"JIT INDEXING COMPLETE. Total indexed chunks now: {len(self.doc_chunks)}")
                        print(f"Indexed {len(chunks)} chunks.")
                else:
                    if path:
                        debug_print(f"Skipping '{title}' - already indexed")
        except Exception as e:
            print(f"JIT Error: {e}")
            debug_print(f"JIT EXCEPTION: {type(e).__name__}: {e}")

        # === EPHEMERAL GUARD: Early Return if No Articles Indexed ===
        # If no articles were JIT-indexed for this query, return empty immediately.
        # This prevents searching a stale index from previous queries.
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            debug_print("=" * 70)
            debug_print("EPHEMERAL GUARD: No articles indexed for this query.")
            
            # ADAPTIVE RAG: Trigger rebound if no articles found
            if rebound_depth == 0 and self.use_joints and self.entity_joint:
                print("No direct articles found. Expanding research...")
                new_terms = self.entity_joint.suggest_expansion(query, search_terms)
                if new_terms:
                    print(f" Expanding Research: {new_terms}")
                    return self.retrieve(query, top_k, rebound_depth=1, extra_terms=new_terms)
            
            debug_print("Returning empty results to prevent state bleed.")
            debug_print("=" * 70)
            print("No relevant articles found for this query. Cannot retrieve context.")
            return []  # Return empty - no context flag

        debug_print(f"JIT indexing complete. FAISS now contains {self.faiss_index.ntotal} vectors from {len(self.indexed_paths)} articles.")

        # 1. Dense Retrieval
        debug_print("-" * 70)
        debug_print("PHASE 1: DENSE RETRIEVAL (FAISS)")
        print("Performing Dense Retrieval...")
        dense_hits = {}
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                debug_print(f"FAISS index total vectors: {self.faiss_index.ntotal}")
                debug_print(f"Encoding query for dense search...")
                q_emb = self.encoder.encode([query]).astype('float32')
                debug_print(f"Query embedding shape: {q_emb.shape}")
                k_search = min(top_k * 4, self.faiss_index.ntotal) # Fetch more for reranking
                debug_print(f"Searching for top {k_search} candidates...")
                D, I = self.faiss_index.search(q_emb, k_search)
                debug_print(f"FAISS search distances (top 5): {D[0][:5]}")
                debug_print(f"FAISS search indices (top 5): {I[0][:5]}")
                dense_hits = {idx: rank for rank, idx in enumerate(I[0])}
                debug_print(f"Dense retrieval found {len(dense_hits)} hits")
                print(f"Dense retrieval found {len(dense_hits)} hits.")
            except Exception as e: 
                print(f"Dense search error: {e}")
                debug_print(f"Dense search exception: {type(e).__name__}: {e}")
                pass
        else:
            debug_print("FAISS index not available or empty")
            print("FAISS index not available or empty for dense retrieval.")
        
        # 2. Sparse Retrieval
        debug_print("-" * 70)
        debug_print("PHASE 2: SPARSE RETRIEVAL (BM25)")
        print("Performing Sparse Retrieval (BM25)...")
        sparse_hits = {}
        if self.bm25:
            try:
                tokenized_query = query.split(" ")
                debug_print(f"Tokenized query: {tokenized_query}")
                debug_print("Computing BM25 scores...")
                sparse_scores = self.bm25.get_scores(tokenized_query)
                debug_print(f"BM25 scores shape: {sparse_scores.shape}, max score: {sparse_scores.max():.4f}")
                sparse_indices = np.argsort(sparse_scores)[::-1][:top_k * 4] # Fetch more for reranking
                debug_print(f"Top 5 BM25 scores: {[sparse_scores[i] for i in sparse_indices[:5]]}")
                debug_print(f"Top 5 BM25 indices: {sparse_indices[:5].tolist()}")
                sparse_hits = {idx: rank for rank, idx in enumerate(sparse_indices)}
                debug_print(f"Sparse retrieval found {len(sparse_hits)} hits")
                print(f"Sparse retrieval found {len(sparse_hits)} hits.")
            except Exception as e: 
                print(f"Sparse search error: {e}")
                debug_print(f"Sparse search exception: {type(e).__name__}: {e}")
                pass
        else:
            debug_print("BM25 index not available")
            print("BM25 index not available for sparse retrieval.")
            
        # 3. Reciprocal Rank Fusion (Initial Filter)
        debug_print("-" * 70)
        debug_print("PHASE 3: RECIPROCAL RANK FUSION (RRF)")
        print("Combining results with RRF...")
        fused_scores = {}
        all_indices = set(dense_hits.keys()) | set(sparse_hits.keys())
        debug_print(f"Total unique indices from both retrievers: {len(all_indices)}")
        k = 60
        debug_print(f"RRF constant k={k}")
        
        for idx in all_indices:
            if idx == -1: continue 
            score = 0
            contributions = []
            if idx in dense_hits:
                dense_contrib = 1 / (k + dense_hits[idx])
                score += dense_contrib
                contributions.append(f"dense={dense_contrib:.4f}")
            if idx in sparse_hits:
                sparse_contrib = 1 / (k + sparse_hits[idx])
                score += sparse_contrib
                contributions.append(f"sparse={sparse_contrib:.4f}")
            fused_scores[idx] = score
            if len(fused_scores) <= 5:  # Log first 5 for brevity
                debug_print(f"  idx={idx}, score={score:.4f} ({', '.join(contributions)})")
            
        # Get top 20 candidates for Reranking
        top_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        debug_print(f"Selected top 20 candidates for reranking")
        debug_print(f"Top 5 RRF scores: {[(idx, score) for idx, score in top_candidates[:5]]}")
        
        # 4. Chunk Filtering / Neural Reranking
        debug_print("-" * 70)
        debug_print("PHASE 4: CHUNK FILTERING / RERANKING")
        reranked_results = []
        
        # Prepare chunks for filtering
        chunks_to_filter = []
        for idx, score in top_candidates:
            if idx < len(self.doc_chunks):
                chunks_to_filter.append({
                    'idx': idx,
                    'text': self.doc_chunks[idx],
                    'rrf_score': score,
                    'metadata': self.documents[idx] if idx < len(self.documents) else {}
                })
        
        # Joint 3: Chunk Filtering (preferred)
        if self.use_joints and self.filter_joint:
            debug_print("Using Joint 3 for chunk filtering...")
            print("Filtering chunks with Joint 3...")
            try:
                # Pass entity_info for comparison-aware filtering
                filtered_chunks = self.filter_joint.filter(query, chunks_to_filter, top_k, entity_info=entity_info, mode=mode)
                debug_print(f"Joint 3 returned {len(filtered_chunks)} filtered chunks")
                
                # Convert back to (idx, score) format
                for chunk in filtered_chunks:
                    idx = chunk['idx']
                    relevance_score = chunk.get('relevance_score', chunk.get('rrf_score', 0))
                    reranked_results.append((idx, relevance_score))
                
                if reranked_results:
                    avg_score = sum(score for _, score in reranked_results) / len(reranked_results)
                    debug_print(f"Chunk filtering complete. Avg relevance: {avg_score:.2f}/10")
                    print(f"Filtering complete. {len(reranked_results)} chunks selected.")
                
            except Exception as e:
                debug_print(f"Joint 3 filtering failed: {type(e).__name__}: {e}")
                debug_print("Falling back to neural reranking")
                self.use_joints = False  # Temporary fallback
        
        # Fallback: Neural Reranking
        if not reranked_results:
            debug_print("Using neural reranking...")
            print("Reranking candidates...")
            try:
                from sentence_transformers import CrossEncoder
                # Load small efficient cross-encoder
                debug_print("Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
                reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
                
                pairs = []
                candidate_indices = []
                
                debug_print(f"Building query-document pairs for reranking...")
                for idx, _ in top_candidates:
                    if idx < len(self.doc_chunks):
                        text = self.doc_chunks[idx]
                        pairs.append([query, text])
                        candidate_indices.append(idx)
                
                debug_print(f"Created {len(pairs)} pairs for reranking")
                
                if pairs:
                    debug_print("Running cross-encoder prediction...")
                    scores = reranker.predict(pairs)
                    debug_print(f"Reranker scores (top 5): {scores[:5]}")
                    
                    for i, score in enumerate(scores):
                        idx = candidate_indices[i]
                        reranked_results.append((idx, float(score)))
                    
                    # Sort by new scores
                    reranked_results.sort(key=lambda x: x[1], reverse=True)
                    reranked_results = reranked_results[:top_k]
                    debug_print(f"Reranking complete. Kept top {len(reranked_results)} results")
                    debug_print(f"Top reranked scores: {[score for _, score in reranked_results]}")
                    print(f"Reranking complete. Top score: {reranked_results[0][1]:.4f}")
                else:
                    debug_print("No pairs to rerank, falling back to RRF")
                    reranked_results = top_candidates[:top_k]
                     
            except Exception as e:
                print(f"Reranker failed (or not installed), falling back to RRF: {e}")
                debug_print(f"Reranker exception: {type(e).__name__}: {e}")
                debug_print("Falling back to RRF scores")
                reranked_results = top_candidates[:top_k]

        # === ADAPTIVE RAG: QUALITY CHECK & REBOUND ===
        # If results are poor, trigger a second pass with expanded terms
        if rebound_depth == 0 and self.use_joints and self.entity_joint:
            top_score = reranked_results[0][1] if reranked_results else 0.0
            
            # Condition: No results OR Max score is weak
            if len(reranked_results) == 0 or top_score < config.ADAPTIVE_THRESHOLD:
                debug_print(f"[ADAPTIVE] Quality Check Failed: top_score={top_score}, count={len(reranked_results)}")
                print(f"Initial results weak (Score: {top_score:.2f}). Thinking deeper...")
                
                # Ask Joint 1 for help
                new_terms = self.entity_joint.suggest_expansion(query, search_terms)
                
                if new_terms:
                    print(f" Expanding Research: {new_terms}")
                    # Recursive Call - Add new terms to existing index
                    return self.retrieve(query, top_k, rebound_depth=1, extra_terms=new_terms)
                else:
                    debug_print("[ADAPTIVE] No expansion terms generated. Accepting fate.")

        results = []
        # 5. Fact Refinement (Joint 4)
        debug_print("-" * 70)
        debug_print("PHASE 5: FACT REFINEMENT (JOINT 4)")
        
        extracted_facts = []
        if self.use_joints and self.fact_joint and reranked_results and mode != "CODE":
            debug_print("Using Joint 4 for fact refinement...")
            try:
                # Loop through top 3 candidates to find supporting evidence
                candidates_to_check = reranked_results[:3]
                found_support = False
                verification_failures = []
                
                for i, (top_idx, top_score) in enumerate(candidates_to_check):
                    if top_idx >= len(self.documents):
                        continue
                        
                    top_doc = self.documents[top_idx]
                    path = top_doc.get('path')
                    zim_path = top_doc.get('source_zim')
                    title = top_doc.get('title', 'Unknown')
                    
                    if not (path and zim_path):
                        continue
                        
                    debug_print(f"Refining facts from candidate {i+1}: '{title}' ({path})")
                    print(f"Scanning '{title}' for specific details...")
                    
                    # Open ZIM and get full text
                    zim = libzim.Archive(zim_path)
                    entry = zim.get_entry_by_path(path)
                    item = entry.get_item()
                    full_text = TextProcessor.extract_text(item.content)
                    
                    # PREMISE VERIFICATION (Systems Fix)
                    verification = self.fact_joint.verify_premise(query, full_text)
                    
                    if verification['status'] == 'SUPPORTED':
                        debug_print(f"Premise SUPPORTED by candidate {i+1}")
                        found_support = True
                        
                        # Extract facts from this supporting article
                        facts = self.fact_joint.refine_facts(query, full_text)
                        if facts:
                            print(f"Found {len(facts)} verified details.")
                            extracted_facts.extend(facts)
                        break # Stop checking once we find support
                        
                    elif verification['status'] == 'CONTRADICTED':
                        debug_print(f"Candidate {i+1} result: {verification['status']} ({verification['reason']})")
                        verification_failures.append(verification['reason'])
                        
                    else:
                        # UNSUPPORTED (Irrelevant text)
                        # Do not treat as a failure/blocker, just skip facts
                        debug_print(f"Candidate {i+1} result: {verification['status']} (Skipping facts, no alert)")
                
                # Only alert if we found NO support AND we found explicit contradictions
                if not found_support and verification_failures:
                    debug_print(f"[SYSTEM ALERT] Premise explicitly contradicted in {len(verification_failures)} articles.")
                    primary_reason = verification_failures[0]
                    print(f"Wait... {primary_reason}")
                    # Inject override as a fact
                    extracted_facts.insert(0, f"[SYSTEM ALERT - PREMISE INCORRECT]: {primary_reason}")

            except Exception as e:
                debug_print(f"Joint 4 failed: {type(e).__name__}: {e}")

        debug_print("-" * 70)
        debug_print("PHASE 6: RESULTS ASSEMBLY")
        print("\nSemantic Search Results:")
        for i, (idx, score) in enumerate(reranked_results):
            if idx < len(self.doc_chunks):
                doc = self.documents[idx]
                title = doc['title']
                chunk_text = self.doc_chunks[idx]
                debug_print(f"Result {i+1}: idx={idx}, title='{title}', score={score:.4f}, chunk_len={len(chunk_text)}")
                print(f"   - {doc['title']} (Score: {score:.4f})")
                results.append({
                    'text': self.doc_chunks[idx],
                    'metadata': doc,
                    'score': score,
                    'search_context': {'entities': search_terms, 'facts': extracted_facts}
                })
        
        debug_print(f"RETRIEVE COMPLETE. Returning {len(results)} results")
        debug_print("=" * 70)
        debug_print(f"RETRIEVE COMPLETE. Returning {len(results)} results")
        debug_print("=" * 70)
        return results

    def retrieve_web(self, query: str, top_k: int = 5) -> List[Dict]:
        """Experimental Online Retrieval Pipeline."""
        print(f" ONLINE MODE: Searching the web for '{query}'...")
        
        # 1. Search
        search_results = self.web_retriever.search(query, max_results=5)
        if not search_results:
            return []
            
        print(f"Found {len(search_results)} links. Scraping content safely...")
        
        # 2. Scrape & Chunk
        self.jit_cache = {}
        self.faiss_index = None # Reset index
        self.doc_chunks = []
        self.documents = []
        self.indexed_paths = set()
        
        chunk_count = 0
        
        for res in search_results:
            url = res.get('href')
            title = res.get('title')
            
            try:
                # Scrape safely
                text = self.web_retriever.scrape_url_sandboxed(url)
                if not text or len(text) < 100:
                    continue
                    
                chunks = TextProcessor.chunk_text(text)
                if chunks:
                    # Index immediately (JIT style)
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if not self.encoder:
                        self.encoder = SentenceTransformer(self.model_name, device=device)
                        
                    embeddings = self.encoder.encode(chunks, device=device, show_progress_bar=False, convert_to_numpy=True)
                    
                    if self.faiss_index is None:
                         dimension = embeddings.shape[1]
                         self.faiss_index = faiss.IndexFlatL2(dimension)
                    
                    self.faiss_index.add(embeddings.astype('float32'))
                    self.doc_chunks.extend(chunks)
                    for _ in chunks:
                        self.documents.append({
                            'title': title,
                            'path': url,
                            'source_zim': 'WEB'
                        })
                    
                    chunk_count += len(chunks)
                    print(f"  Scraped & Indexed: {title[:30]}... ({len(chunks)} chunks)")
                    
            except Exception as e:
                debug_print(f"Failed to process {url}: {e}")
                
        if chunk_count == 0:
            return []
            
        # 3. Dense Retrieval on the new web index
        print(f"Retrieving relevant chunks from {chunk_count} web segments...")
        q_emb = self.encoder.encode([query]).astype('float32')
        D, I = self.faiss_index.search(q_emb, top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.doc_chunks):
                results.append({
                    'text': self.doc_chunks[idx],
                    'metadata': self.documents[idx],
                    'score': float(D[0][i])
                })
                
        return results

    def search_by_title(self, query: str, zim_path: str = None, full_text: bool = False) -> List[Dict]:
        """Fast fallback: Search by title using ZIM's internal index (supports multiple ZIMs)."""
        debug_print(f"search_by_title called: query='{query}', full_text={full_text}")
        
        zim_files_to_search = []
        if zim_path:
            zim_files_to_search = [zim_path]
        else:
            # Search all ZIM files in current directory
            zim_files_to_search = [f for f in os.listdir('.') if f.endswith('.zim')]
            debug_print(f"  ZIM files to search in '.': {zim_files_to_search}")
            
        if not zim_files_to_search:
            debug_print("No ZIM files found")
            return []

        all_results = []
        # We need a shared hits map if we want global deduplication, or we can just collect lists
        # But score logic is local to each ZIM search.
        
        debug_print(f"Searching across {len(zim_files_to_search)} ZIM files: {zim_files_to_search}")

        for current_zim_path in zim_files_to_search:
            try:
                debug_print(f"Opening ZIM archive: {current_zim_path}")
                zim = libzim.Archive(current_zim_path)
                searcher = libzim.SuggestionSearcher(zim)
                
                clean_query = query.replace("?", "").replace(".", "").replace("!", "").replace("_", " ")
                tokens = clean_query.split()
                
                # Extended stop words: includes instructional/generic terms
                stopwords = {
                    # Question words
                    "how", "what", "when", "who", "where", "why", "which",
                    # Common verbs
                    "is", "are", "was", "were", "do", "did", "does", "can", "could", "would", "should", "has", "have", "had",
                    # Articles & conjunctions
                    "the", "a", "an", "and", "or", "but", "if", "then",
                    # Prepositions
                    "in", "on", "at", "to", "for", "of", "with", "by", "about", "from", "into",
                    # Instructional words (common false positive triggers)
                    "tell", "me", "explain", "describe", "define", "show", "give", "find", "get",
                    "introduced", "features", "list", "main", "what", "overview", "details", "information",
                    "best", "way", "good", "bad", "better", "worse", "how", "to", "secure", "protect", # NEW: qualitative words
                    # Comparison words
                    "comparing", "compare", "difference", "different", "between", "versus", "vs", "summary",
                    # Format words
                    "markdown", "table", "graph", "chart", "json", "html", "code", "format",
                    # Common filler
                    "some", "any", "all", "many", "most", "other", "such", "very", "just", "also", "only"
                }
                
                # === WEIGHTED KEYWORD SCORER ===
                scored_keywords = []
                for i, w in enumerate(tokens):
                    if w.lower() in stopwords: continue
                    
                    score = 1.0
                    if len(w) > 0 and w[0].isupper() and i > 0: score *= 3.0
                    if len(w) < 4: score *= 0.5
                    if w.isupper() and len(w) >= 2: score = max(score, 3.0)
                    
                    scored_keywords.append((w, score))

                scored_keywords.sort(key=lambda x: x[1], reverse=True)
                keywords = [w for w, s in scored_keywords[:3]]
                debug_print(f"  Extracted keywords for title search: {keywords}")
                
                hits_map = {} 
                
                # Strategy 1: Individual keywords (Try subjects first) with Singular/Plural Expansion
                sorted_keywords = sorted(keywords, key=len, reverse=True)
                for kw in sorted_keywords[:4]:
                    search_terms = {kw}
                    if kw.endswith('s') and len(kw) > 3:
                        search_terms.add(kw[:-1]) # Try singular
                    
                    for term in search_terms:
                        debug_print(f"RAG:   Trying keyword title search: '{term}'")
                        results = searcher.suggest(term)
                        if results.getEstimatedMatches() > 0:
                             self._collect_hits(zim, results, hits_map, full_text, source=current_zim_path)
                             if len(hits_map) >= 5: break
                    if len(hits_map) >= 5: break

                # Strategy 2: Full phrase (Less reliable for prefix search)
                if len(hits_map) < 3 and keywords:
                    full_term = " ".join(keywords)
                    debug_print(f"RAG:   Trying full phrase title search: '{full_term}'")
                    results = searcher.suggest(full_term)
                    if results.getEstimatedMatches() > 0:
                         self._collect_hits(zim, results, hits_map, full_text, source=current_zim_path)
                             
                # Strategy 3: Original query
                if not hits_map:
                    results = searcher.suggest(query)
                    if results.getEstimatedMatches() > 0:
                        self._collect_hits(zim, results, hits_map, full_text, source=current_zim_path)

                # Strategy 4: Fallback Linear Scan (Robustness for custom/unindexed small ZIMs)
                # If libzim indexing failed (common with some bindings/custom ZIMs), we scan titles linearly.
                if not hits_map and zim.entry_count < 10000:
                    # Only try this for reasonably small ZIMs to avoid performance hit
                    debug_print(f"  Fallback: Linear scan for small ZIM '{current_zim_path}' ({zim.entry_count} entries)...")
                    for i in range(zim.entry_count):
                        try:
                            entry = zim._get_entry_by_id(i)
                            if entry.is_redirect: continue
                            
                            # Basic string matching
                            # We check if query terms are in the title
                            if query.lower() in entry.title.lower():
                                debug_print(f"    - Fallback matched: '{entry.title}'")
                                
                                content_snippet = ""
                                if full_text:
                                    try:
                                        item = entry.get_item()
                                        if item.mimetype == "text/html":
                                            raw_text = TextProcessor.extract_text(item.content)
                                            content_snippet = raw_text[:500] if raw_text else ""
                                    except: pass
                                
                                hits_map[entry.path] = {
                                    'metadata': {
                                        'title': entry.title,
                                        'path': entry.path,
                                        'zim_index': i,
                                        'source_zim': current_zim_path,
                                        'snippet': content_snippet
                                    },
                                    'score': 100.0 # High priority for direct title match
                                }
                        except Exception as e:
                            continue
                    
                    if hits_map:
                        debug_print(f"  Fallback found {len(hits_map)} matches.")

                all_results.extend(list(hits_map.values()))

            except Exception as e:
                debug_print(f"Error searching ZIM '{current_zim_path}': {type(e).__name__}: {e}")
                continue
        
        # Deduplicate results across ZIMs (unlikely overlap but possible with titles)
        # We'll just return top 25 total to ensure diversity from multiple ZIMs
        debug_print(f"Total results across all ZIMs: {len(all_results)}")
        
        # Fix 2: Prioritize Wikipedia ZIMs
        # Sort results so that hits from ZIMs containing 'wikipedia' come first
        def get_priority(hit):
            src = hit['metadata'].get('source_zim', '').lower()
            if 'wikipedia' in src:
                return 2
            if 'stackexchange' in src or 'serverfault' in src or 'superuser' in src:
                return 1
            return 0
            
        all_results.sort(key=lambda x: (get_priority(x), x.get('score', 0)), reverse=True)
        
        return all_results[:25]

    def search_by_embedding(self, query: str, top_k: int = 5, zim_path: str = None, full_text: bool = False) -> List[Dict]:
        """Search titles using vector embeddings."""
        debug_print(f"search_by_embedding called: query='{query}', top_k={top_k}, full_text={full_text}")
        if not self.title_faiss_index or not self.title_metadata:
            debug_print("Title FAISS index or metadata not available")
            return []
            
        if not zim_path:
            files = [f for f in os.listdir('.') if f.endswith('.zim')]
            if files:
                zim_path = files[0]
            else:
                return []

        try:
            zim = libzim.Archive(zim_path)
            
            # Embed Query
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            q_emb = self.encoder.encode([query], device=device).astype('float32')
            
            # Search
            D, I = self.title_faiss_index.search(q_emb, top_k)
            print(f"DEBUG: FAISS Indices: {I[0]}")
            print(f"DEBUG: FAISS Distances: {D[0]}")

            results = []
            for idx in I[0]:
                if idx == -1 or idx >= len(self.title_metadata):
                    continue
                
                meta = self.title_metadata[idx]
                hit_path = meta['path']
                print(f"DEBUG: Checking path: {hit_path}")
                
                # Fetch content from ZIM
                try:
                    entry = zim.get_entry_by_path(hit_path)
                    item = entry.get_item()
                    if item.mimetype == 'text/html':
                        text = TextProcessor.extract_text(item.content)
                        display_text = text if full_text else text[:2000]
                        
                        results.append({
                            'text': display_text,
                            'metadata': {'title': entry.title, 'path': entry.path},
                            'score': 1.0 # Placeholder
                        })
                except:
                    continue
            
            return results

        except Exception as e:
            print(f"Semantic Search Error: {e}")
            return []

    def _collect_hits(self, zim, results, hits_map: Dict, full_text: bool, source: str = None):
        """Helper to collect and process hits."""
        try:
            s_hits = results.getResults(0, 5) # Top 5 per strategy
            debug_print(f"    _collect_hits: Processing {len(s_hits) if hasattr(s_hits, '__len__') else 'unknown'} hits from source {source}")
            
            for hit_path in s_hits:
                if hit_path in hits_map:
                    continue
                try:
                    # In libzim 3.6.0+, hit_path is the string path
                    entry = None
                    try:
                        entry = zim.get_entry_by_path(hit_path)
                    except:
                        try:
                            # Fallback if it's somehow a title or other identifier
                            entry = zim.get_entry_by_title(hit_path)
                        except Exception as e:
                            debug_print(f"      Failed to get entry for '{hit_path}': {e}")
                            continue
                    
                    if not entry:
                         continue

                    item = entry.get_item()
                    if item.mimetype == 'text/html':
                        text = TextProcessor.extract_text(item.content)
                        display_text = text if full_text else text[:2000]
                        
                        hits_map[hit_path] = {
                            'text': display_text, 
                            'metadata': {
                                'title': entry.title, 
                                'path': entry.path,
                                'source_zim': source
                            },
                            'score': 1.0
                        }
                        
                        # Fix 1: Hard Filter for "User" profiles (StackExchange pollution)
                        if entry.title.startswith("User "):
                            debug_print(f"      Skipping User profile: {entry.title}")
                            del hits_map[hit_path]
                            continue
                            
                        debug_print(f"      Added hit: {entry.title}")
                except Exception as inner_e:
                    debug_print(f"      Error processing hit {hit_path}: {inner_e}")
                    continue
        except Exception as e:
            debug_print(f"    _collect_hits top-level error: {e}")

if __name__ == "__main__":
    pass
