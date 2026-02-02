
"""
RAG System Package.
Handles indexing, retrieval, and context generation for the chatbot.
"""

import os
import sys
import pickle
import numpy as np
import time
import glob
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

# Import Mixins
from .search import SearchModule
from .orchestrator import OrchestrationModule

class RAGSystem(SearchModule, OrchestrationModule):
    def __init__(self, index_dir: str = "data/indices", zim_path: str = None, zim_paths: List[str] = None, load_existing: bool = True):
        self.index_dir = index_dir
        self.encoder = None
        self.model_name = 'all-MiniLM-L6-v2'
        
        # === MULTI-ZIM SUPPORT ===
        # Discover all ZIM files and maintain lazy-loaded archive cache
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
            # Check for local offline model
            # We assume we are in chatbot/rag/__init__.py, so up 2 levels is root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # NOTE: Original was os.path.dirname(os.path.dirname(os.path.abspath(__file__))) from chatbot/rag.py
            # rag.py was in chatbot/. so up 2 levels = root.
            # now we are in chatbot/rag/__init__.py. so up 3 levels = root.
            
            # Wait, let's verify path.
            # chatbot/rag.py -> dirname = chatbot. dirname = root.
            # chatbot/rag/__init__.py -> dirname = rag. dirname = chatbot. dirname = root.
            # So yes, 3 dirnames.
            
            local_embed_path = os.path.join(root_dir, "shared_models", "embedding")
            
            if os.path.exists(local_embed_path):
                debug_print(f"Loading local embedding model from: {local_embed_path}")
                self.model_name = local_embed_path
            
            # Move encoder to CPU to save VRAM for the main LLM.
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")

        # Multi-Joint System Configuration
        self.use_joints = config.USE_JOINTS

        # Load Existing Content Indices - DEPRECATED / DISABLED for Zero-Index Mode
        print("Zero-Index Mode: Skipping index loading.")

        # === LAZY JOINT INITIALIZATION ===
        # Joints are initialized on first use, not at startup
        # This saves memory for queries that don't need all joints
        self._joint_classes = None  # Will hold class references
        self._entity_joint = None
        self._resolver_joint = None
        self._scorer_joint = None
        self._coverage_joint = None
        self._comparison_joint = None
        self._filter_joint = None
        self._fact_joint = None
        self._pioneer_joint = None

        if self.use_joints:
            debug_print("Joint system configured for LAZY initialization")

    def _ensure_joint_classes(self):
        """Import joint classes once when first needed."""
        if self._joint_classes is not None:
            return
        try:
            from chatbot.joints import (
                EntityExtractorJoint, ArticleScorerJoint, CoverageVerifierJoint,
                ChunkFilterJoint, FactRefinementJoint, ComparisonJoint, MultiHopResolverJoint,
                PioneerJoint
            )
            self._joint_classes = {
                'entity': EntityExtractorJoint,
                'resolver': MultiHopResolverJoint,
                'scorer': ArticleScorerJoint,
                'coverage': CoverageVerifierJoint,
                'comparison': ComparisonJoint,
                'filter': ChunkFilterJoint,
                'fact': FactRefinementJoint,
                'pioneer': PioneerJoint
            }
            debug_print("Joint classes imported successfully")
        except Exception as e:
            debug_print(f"Failed to import joint classes: {e}")
            self._joint_classes = {}
            self.use_joints = False

    @property
    def entity_joint(self):
        """Lazy initialization of EntityExtractorJoint."""
        if self._entity_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'entity' in self._joint_classes:
                debug_print("Initializing EntityExtractorJoint (lazy)")
                self._entity_joint = self._joint_classes['entity']()
        return self._entity_joint

    @property
    def resolver_joint(self):
        """Lazy initialization of MultiHopResolverJoint."""
        if self._resolver_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'resolver' in self._joint_classes:
                debug_print("Initializing MultiHopResolverJoint (lazy)")
                self._resolver_joint = self._joint_classes['resolver'](model=config.MULTI_HOP_JOINT_MODEL)
        return self._resolver_joint

    @property
    def scorer_joint(self):
        """Lazy initialization of ArticleScorerJoint."""
        if self._scorer_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'scorer' in self._joint_classes:
                debug_print("Initializing ArticleScorerJoint (lazy)")
                self._scorer_joint = self._joint_classes['scorer']()
        return self._scorer_joint

    @property
    def coverage_joint(self):
        """Lazy initialization of CoverageVerifierJoint."""
        if self._coverage_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'coverage' in self._joint_classes:
                debug_print("Initializing CoverageVerifierJoint (lazy)")
                self._coverage_joint = self._joint_classes['coverage']()
        return self._coverage_joint

    @property
    def comparison_joint(self):
        """Lazy initialization of ComparisonJoint."""
        if self._comparison_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'comparison' in self._joint_classes:
                debug_print("Initializing ComparisonJoint (lazy)")
                self._comparison_joint = self._joint_classes['comparison'](model=config.COMPARISON_JOINT_MODEL)
        return self._comparison_joint

    @property
    def filter_joint(self):
        """Lazy initialization of ChunkFilterJoint."""
        if self._filter_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'filter' in self._joint_classes:
                debug_print("Initializing ChunkFilterJoint (lazy)")
                self._filter_joint = self._joint_classes['filter']()
        return self._filter_joint

    @property
    def fact_joint(self):
        """Lazy initialization of FactRefinementJoint."""
        if self._fact_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'fact' in self._joint_classes:
                debug_print("Initializing FactRefinementJoint (lazy)")
                self._fact_joint = self._joint_classes['fact']()
        return self._fact_joint
    @property
    def pioneer_joint(self):
        """Lazy initialization of PioneerJoint."""
        if self._pioneer_joint is None and self.use_joints:
            self._ensure_joint_classes()
            if 'pioneer' in self._joint_classes and self._joint_classes['pioneer'] is not None:
                debug_print("Initializing PioneerJoint (lazy)")
                self._pioneer_joint = self._joint_classes['pioneer']()
        return self._pioneer_joint
