"""Configuration constants."""

OLLAMA_CHAT_URL = "N/A" # Deprecated
# Local Model Repositories
MODEL_ALETHEIA_3B = "Ishaanlol/Aletheia-Llama-3.2-3B" 
MODEL_DARKIDOL_8B = "DavidAU/Llama-3.1-DeepSeek-8B-DarkIdol-Instruct-1.2-Uncensored-GGUF"

DEFAULT_MODEL = MODEL_ALETHEIA_3B
STRICT_RAG_MODE = True
DEBUG = False

# Multi-Joint RAG System Configuration
USE_JOINTS = True

# Joint Models - All use the fast Aletheia 3B model
ENTITY_JOINT_MODEL = MODEL_ALETHEIA_3B
SCORER_JOINT_MODEL = MODEL_ALETHEIA_3B
FILTER_JOINT_MODEL = MODEL_ALETHEIA_3B
FACT_JOINT_MODEL = MODEL_ALETHEIA_3B

# Joint Temperatures
ENTITY_JOINT_TEMP = 0.1
SCORER_JOINT_TEMP = 0.0
FILTER_JOINT_TEMP = 0.1
FACT_JOINT_TEMP = 0.0

# Joint Timeout (not used for local inference but kept for compat)
JOINT_TIMEOUT = 10

# Adaptive RAG Configuration
ADAPTIVE_THRESHOLD = 4.0  # If max score is below this, trigger expansion

# Global Context Window Configuration
DEFAULT_CONTEXT_SIZE = 16384



