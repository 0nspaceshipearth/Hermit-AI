
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

"""Configuration constants."""

OLLAMA_CHAT_URL = "N/A" # Legacy/Deprecated
# Local Model Repositories
MODEL_ALETHEIA_3B = "Ishaanlol/Aletheia-Llama-3.2-3B" 
MODEL_QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # "Fast" Model
MODEL_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct-GGUF"      # "Smart" Model
MODEL_NVIDIA_8B = "bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"

# === PERFORMANCE OPTIMIZATION ===
# Tiered options: 1.5B (~1.3GB) < 3B (~2GB) < 8B (~6GB)
DEFAULT_MODEL = MODEL_ALETHEIA_3B   # Good balance of quality vs VRAM
# DEFAULT_MODEL = MODEL_QWEN_1_5B  # Fastest, lowest VRAM
# DEFAULT_MODEL = MODEL_NVIDIA_8B  # Best quality, needs 6GB free VRAM
STRICT_RAG_MODE = False
MIN_ARTICLE_SCORE = 2.5
DEBUG = True

# API / External Model Configuration
API_MODE = False  # If True, use external API instead of local GGUF
API_BASE_URL = "http://localhost:1234/v1"  # Default (LM Studio / Ollama)
API_KEY = "lm-studio"  # Often ignored by local servers but required by spec
API_MODEL_NAME = "local-model"  # Passed in API request

# Multi-Joint RAG System Configuration
USE_JOINTS = True

# === TIERED MODEL ARCHITECTURE ===
# Fast Models (1.5B) for high-volume, low-complexity tasks
# Smart Models (8B) for reasoning, logic, and synthesis

ENTITY_JOINT_MODEL = MODEL_QWEN_1_5B
SCORER_JOINT_MODEL = MODEL_QWEN_1_5B
FILTER_JOINT_MODEL = MODEL_QWEN_1_5B

# Reasoning Joints (8B)
# NOTE: Using 1.5B for fact joint to eliminate model swapping overhead
# This provides 3-5x faster orchestration (5-10s vs 20-30s per query)
# with minimal quality impact. Re-enable 8B if precision is critical.
FACT_JOINT_MODEL = MODEL_QWEN_1_5B       # Fast mode: 3-5x speedup
# FACT_JOINT_MODEL = MODEL_NVIDIA_8B     # Uncomment for higher precision
MULTI_HOP_JOINT_MODEL = MODEL_QWEN_1_5B  # Using 1.5B (8B needs 6GB VRAM)
# MULTI_HOP_JOINT_MODEL = MODEL_NVIDIA_8B  # Uncomment if VRAM available
COMPARISON_JOINT_MODEL = MODEL_QWEN_1_5B # Fast mode (was 8B)

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

