
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
import os
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

"""Configuration constants."""

HF_TOKEN = os.getenv("HF_TOKEN")

OLLAMA_CHAT_URL = "N/A" # Legacy/Deprecated
# Local Model Repositories
MODEL_QWEN_3B = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # "Fast" Model
MODEL_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct-GGUF"      # "Smart" Model
MODEL_NVIDIA_8B = "bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"
MODEL_PIONEER = "0nspaceshipearth/pioneer-translator-v4"

# === PERFORMANCE OPTIMIZATION ===
# Tiered options: 1.5B (~1.3GB) < 3B (~2GB) < 8B (~6GB)
DEFAULT_MODEL = MODEL_QWEN_3B   # Default for public release
# DEFAULT_MODEL = MODEL_QWEN_1_5B  # Fastest, lowest VRAM
# DEFAULT_MODEL = MODEL_NVIDIA_8B  # Best quality, needs 6GB free VRAM
STRICT_RAG_MODE = False
MIN_ARTICLE_SCORE = 2.5
DEBUG = True

# === TITLE GENERATION OPTIMIZATION ===
# Tiny model for fast title generation (0.5B-1.5B recommended)
# This avoids loading the 14B model just to guess Wikipedia titles
MODEL_QWEN_0_5B = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"  # Tiny model for title gen
TITLE_GEN_MODEL = MODEL_QWEN_0_5B  # Fast title generation (~1-3s -> ~0.3s)
# TITLE_GEN_MODEL = MODEL_QWEN_1_5B  # Alternative: slightly smarter
TITLE_CACHE_SIZE = 1000  # LRU cache entries for title generation

# === ZIM ARCHIVE POOLING ===
# Limit concurrent open ZIM handles to reduce file descriptor usage
ZIM_POOL_MAX_SIZE = 5  # Maximum concurrent open ZIM archives

# API / External Model Configuration
API_MODE = False  # If True, use external API instead of local GGUF
API_BASE_URL = "http://localhost:1234/v1"  # Default (LM Studio / Ollama)
API_KEY = "lm-studio"  # Often ignored by local servers but required by spec
API_MODEL_NAME = "local-model"  # Passed in API request

# Multi-Joint RAG System Configuration
USE_JOINTS = True
USE_PIONEER_JOINT = False # Enable if pioneer_sdk binary is present


# === TIERED MODEL ARCHITECTURE ===
# Fast Models (1.5B) for high-volume, low-complexity tasks
# Smart Models (8B) for reasoning, logic, and synthesis

# Unified Architecture: Use the same model for everything to prevent VRAM swapping
JOINT_MODEL = DEFAULT_MODEL 

ENTITY_JOINT_MODEL = JOINT_MODEL
SCORER_JOINT_MODEL = JOINT_MODEL
FILTER_JOINT_MODEL = JOINT_MODEL

# Reasoning Joints (8B)
# NOTE: Using 1.5B for fact joint to eliminate model swapping overhead
# This provides 3-5x faster orchestration (5-10s vs 20-30s per query)
# with minimal quality impact. Re-enable 8B if precision is critical.
FACT_JOINT_MODEL = JOINT_MODEL       # Fast mode: 3-5x speedup
# FACT_JOINT_MODEL = MODEL_NVIDIA_8B     # Uncomment for higher precision
MULTI_HOP_JOINT_MODEL = JOINT_MODEL  # Using 1.5B (8B needs 6GB VRAM)
# MULTI_HOP_JOINT_MODEL = MODEL_NVIDIA_8B  # Uncomment if VRAM available
COMPARISON_JOINT_MODEL = JOINT_MODEL # Fast mode (was 8B)

# Joint Temperatures
ENTITY_JOINT_TEMP = 0.1
SCORER_JOINT_TEMP = 0.0
FILTER_JOINT_TEMP = 0.1
FACT_JOINT_TEMP = 0.0

# Joint Timeout (not used for local inference but kept for compat)
JOINT_TIMEOUT = 30 # Increased for 7B model generation

# Adaptive RAG Configuration
ADAPTIVE_THRESHOLD = 3.0  # Lowered to trigger fewer expansions when data is present

# Global Context Window Configuration
DEFAULT_CONTEXT_SIZE = 8192

SYSTEM_PROMPT = (
    "You are a helpful, thorough AI assistant. When provided with context, "
    "you carefully read ALL of it to find the most accurate and complete answer. "
    "You synthesize information from multiple sources when relevant and always verify "
    "that your answer directly addresses what was asked."
)

# === DYNAMIC ORCHESTRATION CONFIGURATION ===
# Enable signal-based "conscious" decision-making in RAG pipeline
USE_ORCHESTRATION = True

# Maximum orchestration loop iterations (safety limit)
MAX_ORCHESTRATION_STEPS = 10

# Signal Thresholds for Gear-Shifting
MIN_SOURCE_SCORE_THRESHOLD = 6.0   # Below this, trigger query expansion
MIN_COVERAGE_THRESHOLD = 1.0        # Below this, trigger targeted entity search
HIGH_AMBIGUITY_THRESHOLD = 0.7     # Above this, enable multi-hop resolution

# === REFINEMENT CONFIGURATION ===
# Early Termination Thresholds
HIGH_QUALITY_THRESHOLD = 8.0          # Score (0-10) above which we can exit early
MIN_RESULTS_FOR_EARLY_EXIT = 3       # Minimum results before considering early exit
MAX_EXPANSION_ITERATIONS = 2          # Limit expansion loops

# Multi-Hop Resolution
ENABLE_MULTI_HOP_RESOLUTION = True    # Toggle for multi-hop resolver
MULTI_HOP_AMBIGUITY_THRESHOLD = 0.6   # Ambiguity level to trigger resolution

