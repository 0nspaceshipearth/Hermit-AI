
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
MODEL_QWEN_3B = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # "Fast" Model
MODEL_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct-GGUF"      # "Smart" Model
MODEL_NVIDIA_8B = "bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF"

# === PERFORMANCE OPTIMIZATION ===
# Tiered options: 1.5B (~1.3GB) < 3B (~2GB) < 8B (~6GB)
DEFAULT_MODEL = MODEL_QWEN_3B   # Default for public release
# DEFAULT_MODEL = MODEL_QWEN_1_5B  # Fastest, lowest VRAM
# DEFAULT_MODEL = MODEL_NVIDIA_8B  # Best quality, needs 6GB free VRAM
STRICT_RAG_MODE = False
MIN_ARTICLE_SCORE = 2.5
DEBUG = False

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
API_ACCOUNT_ID = ""  # Optional ChatGPT account id for Codex OAuth flows

# Codex/OpenAI cloud auth / provider menu
CODEX_CLOUD_PROVIDER = "openai-codex"
CODEX_CLOUD_LOGIN_COMMAND = "openclaw models auth login --provider openai-codex"
CODEX_CLOUD_DEFAULT_MODEL = "gpt-5.3-codex"
CODEX_CLOUD_MODELS = ["gpt-5.3-codex", "gpt-5.4", "gpt-4o"]

# Multi-Joint RAG System Configuration
USE_JOINTS = True


# === TIERED MODEL ARCHITECTURE ===
# Public branch keeps the fast tiered architecture, but avoids experimental features.
#   - Tier 1 (0.5B): simple structured extraction/scoring/filtering
#   - Tier 2 (1.5B): fact extraction / refinement
#   - Tier 3 (3B): multi-hop + comparison reasoning
#   - Final generation: DEFAULT_MODEL

# Runtime architecture mode:
#   classic = aggressive unload/reload between model tiers (stable default)
#   wave    = keep a single selected model hot and simulate resets at the prompt/state level
RUNTIME_MODE = "classic"
# In wave mode, pin a larger context so the same resident model survives context growth.
WAVE_PIN_CONTEXT_SIZE = 12288

# Optional aux-task reuse: if the default model is already loaded, some helper tasks
# may reuse it to avoid load/unload thrash. Large defaults stay blocked by default.
PREFER_DEFAULT_MODEL_FOR_AUX = True
BLOCK_LARGE_DEFAULT_REUSE_FOR_AUX = True

# llama.cpp runtime tuning knobs for model loading.
LLAMA_LARGE_MODEL_N_BATCH = 768
LLAMA_SMALL_MODEL_N_BATCH = 2048
LLAMA_LARGE_MODEL_N_UBATCH = 768
LLAMA_SMALL_MODEL_N_UBATCH = 1024
LLAMA_THREADS = 0
LLAMA_THREADS_BATCH = 0

# Tiered joints
ENTITY_JOINT_MODEL = MODEL_QWEN_0_5B
SCORER_JOINT_MODEL = MODEL_QWEN_0_5B
FILTER_JOINT_MODEL = MODEL_QWEN_0_5B
FACT_JOINT_MODEL = MODEL_QWEN_1_5B
REFINEMENT_JOINT_MODEL = MODEL_QWEN_1_5B
MULTI_HOP_JOINT_MODEL = MODEL_QWEN_3B
COMPARISON_JOINT_MODEL = MODEL_QWEN_3B

# Legacy alias for compatibility with older codepaths
JOINT_MODEL = MODEL_QWEN_1_5B

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

# === ADAPTIVE FINAL GENERATION ===
# Generation lane: auto | sprint | cruise | beast
FINAL_GENERATION_MODE = "auto"

# Per-lane context targets (used for final answer path)
FINAL_SPRINT_CONTEXT_SIZE = 4096
FINAL_CRUISE_CONTEXT_SIZE = 8192
FINAL_BEAST_CONTEXT_SIZE = 12288

# Per-lane max token budgets
FINAL_SPRINT_MAX_TOKENS = 160
FINAL_CRUISE_MAX_TOKENS = 320
FINAL_BEAST_MAX_TOKENS = 640

# Prompt budgeting: when facts exist, prefer compact fact-first context
FINAL_FACTS_FIRST = True
FINAL_MAX_SOURCE_CHARS_SPRINT = 2500
FINAL_MAX_SOURCE_CHARS_CRUISE = 7000
FINAL_MAX_SOURCE_CHARS_BEAST = 12000

# === RETRIEVAL CONSTANTS ===
# Maximum characters of RAG context to inject into the system prompt.
# ~2500 tokens, leaves headroom in an 8K context window.
MAX_CONTEXT_CHARS = 10000

# Maximum characters per individual article chunk before truncation.
MAX_CHUNK_CHARS = 4000

# Default number of top-k results for RAG retrieval.
DEFAULT_TOP_K = 3

# Maximum links returned in link mode.
MAX_LINK_RESULTS = 10

# Maximum characters of article text stored per search result.
MAX_ARTICLE_TEXT_CHARS = 6000

# === ADVANCED CLI / CHAMBER WORKSPACE ===
# Root directory for local filesystem excursions from the Hermit CLI.
CLI_WORKSPACE_ROOT = "."
CLI_WRITE_ENABLED = True
CLI_SHELL_ENABLED = True
CLI_SHELL_TIMEOUT = 20
CLI_MAX_FILE_READ_CHARS = 12000
CLI_MAX_FILE_WRITE_CHARS = 20000

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
