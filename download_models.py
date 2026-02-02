import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import config
from chatbot.model_manager import ModelManager

if __name__ == "__main__":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed. Run setup.sh first.")
        sys.exit(1)

    print("=== Downloading Hermit AI Models ===")
    print("Pre-fetching models for offline use...")
    
    # 1. Download Embedding Model
    print("\n[Checking] Embedding Model (all-MiniLM-L6-v2)...")
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        embed_path = os.path.join(root_dir, "shared_models", "embedding")
        
        # This will download to cache, then we save to local dir
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(embed_path)
        print(f"✓ Ready: {embed_path}")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")
        sys.exit(1)

    # 2. Download LLM GGUFs
    # Identify models from config
    models_to_download = [
        config.MODEL_QWEN_0_5B, # Tiny Title Generation Model (0.5B)
        config.MODEL_QWEN_1_5B, # Fast Joint Model
        config.MODEL_NVIDIA_8B, # Smart/Reasoning Model (Nvidia 8B)
        config.DEFAULT_MODEL    # Main Chat Model (Qwen 3B)
    ]

    # Add Pioneer model if enabled (requires HF_TOKEN)
    if getattr(config, 'USE_PIONEER_JOINT', False):
        models_to_download.append(config.MODEL_PIONEER)
        if not config.HF_TOKEN:
            print("\n[!] WARNING: Pioneer Joint is enabled but HF_TOKEN is not set.")
            print("    Accessing private models requires a Hugging Face token.")

    # De-duplicate
    models_to_download = list(set(models_to_download))

    for model_id in models_to_download:
        print(f"\n[Checking] {model_id}...")
        try:
            # ensure_model_path handles checking local cache/file and downloading if needed
            path = ModelManager.ensure_model_path(model_id)
            print(f"✓ Ready: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}")
            sys.exit(1)

    print("\n✓ All models downloaded and ready for offline use.")
