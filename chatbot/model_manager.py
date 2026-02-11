
"""
Model Manager for Local Inference.
Handles downloading GGUF models from Hugging Face and loading them via llama-cpp-python.
"""

import os
import sys
import glob
from typing import Optional, Dict, List, Callable
import requests

# Force stable HTTP download path.
# Some environments fail with Xet transport ("xet_get") even on valid repos.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files, try_to_load_from_cache
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print("WARNING: llama-cpp-python not installed. Local inference will fail.")

from chatbot import config

# Global progress callback for GUI integration
# Signature: callback(status: str, progress: float, total_size: str)
# - status: "downloading", "loading", "ready", "error"
# - progress: 0.0 to 1.0 (or -1 for indeterminate)
# - total_size: human-readable size string like "2.1 GB"
_download_callback: Optional[Callable[[str, float, str], None]] = None


def set_download_callback(callback: Optional[Callable[[str, float, str], None]]) -> None:
    """Set a callback function to receive download progress updates.
    
    Args:
        callback: Function taking (status, progress, total_size) or None to clear.
    """
    global _download_callback
    _download_callback = callback


def _notify_progress(status: str, progress: float = -1, total_size: str = "") -> None:
    """Internal helper to notify the callback if set."""
    if _download_callback:
        try:
            _download_callback(status, progress, total_size)
        except Exception:
            pass  # Don't let callback errors break downloads


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _is_xet_transport_error(exc: Exception) -> bool:
    """Detect known Hugging Face transport/plugin failures."""
    text = str(exc).lower()
    needles = (
        "xet_get",
        "http_get",
        "hf_xet",
        "cas service error",
        "huggingface_hub.xet_get",
        "huggingface_hub.http_get",
        "unknown argument(s)",
    )
    return any(n in text for n in needles)


def _manual_http_download(repo_id: str, filename: str, local_dir: str) -> str:
    """
    Fallback direct download path that bypasses huggingface_hub transport plugins.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    target_path = os.path.join(local_dir, filename)
    tmp_path = f"{target_path}.part"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    timeout = (10, 120)
    downloaded = 0
    total_bytes = 0
    detail = ProgressTqdm._default_desc

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", "0") or 0)
            if total_bytes > 0:
                _notify_progress("downloading", ProgressTqdm._progress_offset, detail)

            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes > 0:
                        shard_progress = min(1.0, downloaded / total_bytes)
                        progress = ProgressTqdm._progress_offset + (shard_progress * ProgressTqdm._progress_scale)
                        _notify_progress("downloading", min(1.0, progress), detail)

        os.replace(tmp_path, target_path)
        return target_path
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


class ProgressTqdm:
    """A tqdm-compatible wrapper that notifies the global callback."""
    _progress_offset: float = 0.0
    _progress_scale: float = 1.0
    _default_desc: str = "Downloading"

    @classmethod
    def configure(cls, offset: float = 0.0, scale: float = 1.0, detail: str = "Downloading") -> None:
        """Configure progress mapping for the next hf_hub_download call."""
        cls._progress_offset = max(0.0, min(1.0, offset))
        cls._progress_scale = max(0.0, min(1.0, scale))
        cls._default_desc = detail or "Downloading"

    def __init__(self, *args, **kwargs):
        self._total = kwargs.get('total', 0)
        self._n = 0
        self._desc = kwargs.get('desc', self._default_desc)
        
        # Internal tqdm for terminal output
        if tqdm:
            self._tqdm = tqdm(*args, **kwargs)
        else:
            self._tqdm = None

    def update(self, n=1):
        self._n += n
        if self._tqdm:
            self._tqdm.update(n)
        
        if self._total and self._total > 0:
            shard_progress = min(1.0, self._n / self._total)
            progress = self._progress_offset + (shard_progress * self._progress_scale)
            _notify_progress("downloading", min(1.0, progress), self._desc)

    def set_description(self, desc, refresh=True):
        self._desc = desc
        if self._tqdm:
            self._tqdm.set_description(desc, refresh)

    def close(self):
        # Emit a final update for this shard/file based on actual bytes seen.
        if self._total and self._total > 0:
            shard_progress = min(1.0, self._n / self._total)
            final_progress = self._progress_offset + (shard_progress * self._progress_scale)
            _notify_progress("downloading", min(1.0, final_progress), self._desc)
        if self._tqdm:
            self._tqdm.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ModelManager:
    """Singleton manager for local LLM models."""
    
    _instances: Dict[str, 'Llama'] = {}
    
    @staticmethod
    def ensure_model_path(repo_id: str) -> str:
        """
        Ensure the model exists locally. varying quantization support.
        Downloads the best available GGUF if not found.
        """
        # Determine path relative to this file (chatbot/model_manager.py -> project_root/shared_models)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_dir = os.path.join(project_root, "shared_models")
        os.makedirs(model_dir, exist_ok=True)
        print(f"DEBUG: Initializing model directory at: {model_dir}")
        
        # 1. Check if we already have a suitable file for this repo
        # We store them as "RepoName-Quant.gguf" or just rely on huggingface cache
        # Ideally, we copy/symlink to data/models for clarity, or just use the cache path.
        # Using cache path is safer for updates.
        
        print(f"Checking model availability for: {repo_id}")
        
        # Search strategy: Q5_K_M > Q4_K_M > Q8_0 > Q4_0
        preferences = ["Q5_K_M", "Q4_K_M", "Q8_0", "Q4_0", "F16"]
        
        # 0. DIRECT FILE CHECK (Fast Path for manually downloaded models)
        # If repo_id looks like a filename (ends in .gguf) and exists, just use it.
        direct_path = os.path.join(model_dir, repo_id)
        if repo_id.lower().endswith(".gguf") and os.path.exists(direct_path):
             print(f"Loading local model directly: {direct_path}")
             return direct_path
        
        # 0. Fast Path: Check if we have a matching GGUF in the local dir
        # We search for files containing the repo name (or part of it) and the quant
        existing_files = glob.glob(os.path.join(model_dir, "*.gguf"))
        
        if existing_files:
            # Try to match based on preferences
            for quant in preferences:
                # Find files that look like they belong to this model (heuristic: usually filename has quant)
                # Matches if quant is in filename
                matches = [f for f in existing_files if quant.lower() in f.lower()]
                if matches:
                    # Determine if it matches the repo roughly?
                    # Since we centralized, we might have multiple models.
                    # Simple heuristic: Just pick the first match if we assume we only keep what we want?
                    # Better: Check if the filename roughly matches the repo name's last part
                    repo_name_part = repo_id.split('/')[-1]
                    
                    # Heuristic: check if significant part of repo name is in filename
                    # For DarkIdol: look for "DarkIdol"
                    # For Qwen: look for "Qwen"
                    
                    match_found = False
                    candidate_file = None

                    for candidate in matches:
                        if "DarkIdol" in repo_name_part and "DarkIdol" in candidate:
                            match_found = True
                            candidate_file = candidate
                            break
                        elif "Qwen2.5-3B" in repo_id and "qwen2.5-3b-instruct" in candidate.lower():
                             match_found = True
                             candidate_file = candidate
                             break
                        elif "Llama-3.1" in repo_name_part and "Llama-3.1" in candidate:
                             match_found = True
                             candidate_file = candidate
                             break
                        elif "Qwen2.5-7B" in repo_id:
                             if "qwen2.5-7b-instruct" in candidate.lower():
                                 if "00001-of-" in candidate or "-00001" in candidate:
                                     match_found = True
                                     candidate_file = candidate
                                     break
                                 elif not any("00001-of-" in c for c in matches):
                                     match_found = True
                                     candidate_file = candidate
                                     break
                        elif "Qwen2.5-1.5B" in repo_id:
                             if "qwen2.5-1.5b-instruct" in candidate.lower():
                                 match_found = True
                                 candidate_file = candidate
                             if "qwen2.5-1.5b-instruct" in candidate.lower():
                                 match_found = True
                                 candidate_file = candidate
                                 break
                    if match_found and candidate_file:
                        # [FIX] Verify all shards if it's a split file
                        import re
                        is_valid = True
                        split_check = re.search(r'(.*)-00001-of-(\d{5})\.gguf$', candidate_file)
                        if split_check:
                             base = split_check.group(1)
                             total = int(split_check.group(2))
                             print(f"Verifying {total} shards for {os.path.basename(candidate_file)}...")
                             for i in range(1, total + 1):
                                 shard = f"{base}-{i:05d}-of-{total:05d}.gguf"
                                 if not os.path.exists(shard):
                                     print(f"Missing shard: {os.path.basename(shard)}")
                                     is_valid = False
                                     break
                        
                        if is_valid:
                            print(f"Found local cached model: {candidate_file}")
                            return candidate_file
                        else:
                            print(f"Incomplete split model found. Re-triggering download logic.")
                            # Fall through to download logic
                    else:
                        print(f"Skipping ambiguous local file(s) for {repo_id}")
                    
                    if matches and len(existing_files) < 10: 
                         pass 


        # List files in repo (to find best quantization)
        try:
            # Note: Don't notify "checking" here - it causes dialog flash for cached models
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {repo_id}")
            
            selected_file = None
            for quant in preferences:
                matches = [f for f in gguf_files if quant.lower() in f.lower()]
                if matches:
                    selected_file = matches[0]
                    # Prefer "uncensored" in name if duplicates exist? 
                    # Usually repo is specific enough.
                    break
            
            if not selected_file:
                # Fallback to the smallest/first
                selected_file = gguf_files[0]
                
            print(f"Selected model file: {selected_file}")
            
            # Check if this file is already in HuggingFace cache (avoid dialog flash)
            cached_path = try_to_load_from_cache(repo_id, selected_file)
            if cached_path is not None and not isinstance(cached_path, type):
                # File is already cached - return silently without showing dialog
                print(f"Model already cached: {cached_path}")
                return cached_path
            
            # Not cached - need to download. Get file info for progress display
            try:
                from huggingface_hub import hf_hub_url, get_hf_file_metadata
                url = hf_hub_url(repo_id=repo_id, filename=selected_file)
                metadata = get_hf_file_metadata(url)
                file_size = metadata.size if metadata.size else 0
                size_str = _format_size(file_size) if file_size else "unknown size"
            except Exception:
                size_str = "unknown size"
            
            # Notify GUI that download is starting (only shown for actual downloads)
            model_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
            _notify_progress("downloading", 0.0, f"{model_name} ({size_str})")
            print(f"Downloading {model_name} ({size_str})...")

            def _download_file(filename: str) -> str:
                """Download one file with progress callback support when available."""
                def _hub_download() -> str:
                    try:
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=model_dir,
                            tqdm_class=ProgressTqdm
                        )
                    except TypeError:
                        # Older huggingface_hub may not support tqdm_class.
                        return hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=model_dir
                        )

                try:
                    return _hub_download()
                except Exception as e:
                    # Retry path for known Xet transport failures.
                    # This keeps downloads functional across mixed hf-xet environments.
                    if _is_xet_transport_error(e):
                        print("⚠️ Xet transport failed; retrying with HTTP-only mode...")
                        os.environ["HF_HUB_DISABLE_XET"] = "1"
                        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                        try:
                            return _hub_download()
                        except Exception as retry_error:
                            if _is_xet_transport_error(retry_error):
                                print("⚠️ huggingface_hub transport unavailable; falling back to direct HTTP download...")
                                return _manual_http_download(repo_id=repo_id, filename=filename, local_dir=model_dir)
                            raise
                    raise
            
            # Download the model (handle potential splits)
            import re
            split_match = re.search(r'(.*)-00001-of-(\d{5})\.gguf$', selected_file)
            
            if split_match:
                base_name = split_match.group(1)
                total_parts = int(split_match.group(2))
                print(f"Detected split GGUF ({total_parts} parts). Downloading all shards...")
                
                final_path = None
                for i in range(1, total_parts + 1):
                    shard_name = f"{base_name}-{i:05d}-of-{total_parts:05d}.gguf"
                    offset = (i - 1) / total_parts
                    scale = 1.0 / total_parts
                    detail = f"Shard {i}/{total_parts}: {model_name}"
                    ProgressTqdm.configure(offset=offset, scale=scale, detail=detail)
                    _notify_progress("downloading", offset, detail)
                    
                    path = _download_file(shard_name)
                    if i == 1:
                        final_path = path
                
                _notify_progress("ready", 1.0, size_str)
                return final_path
            else:
                # Single file download
                ProgressTqdm.configure(offset=0.0, scale=1.0, detail=f"{model_name} ({size_str})")
                path = _download_file(selected_file)
                
                _notify_progress("ready", 1.0, size_str)
                print(f"Model downloaded to: {path}")
                return path
            
        except Exception as e:
            _notify_progress("error", -1, str(e))
            print(f"Error resolving model {repo_id}: {e}")
            # Final Fallback: Check if ANY file exists in model_dir
            if existing_files:
                print(f"Network error, falling back to local file: {existing_files[0]}")
                return existing_files[0]
            raise

    @classmethod
    def get_model(cls, repo_id: str, n_ctx: int = 8192, n_gpu_layers: int = -1) -> 'Llama':
        """
        Get or load a Llama model instance.
        Enforces single-model policy to prevent VRAM OOM.
        Uses 8192 context by default to accommodate RAG content.
        """
        if repo_id in cls._instances:
            return cls._instances[repo_id]
            
        # Unload ALL other models to free VRAM before loading new one
        if cls._instances:
            print(f"Unloading {len(cls._instances)} active models to free VRAM for {repo_id}...")
            import gc
            for key in list(cls._instances.keys()):
                print(f"Unloading model: {key}")
                # Explicitly delete the Llama object
                model_instance = cls._instances[key]
                del model_instance
                del cls._instances[key]
            
            # Force garbage collection to ensure VRAM is released
            cls._instances.clear()
            gc.collect()
            
            # Force CUDA to release memory (critical for OOM prevention)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("CUDA memory cache cleared.")
            except Exception:
                pass  # torch might not be available
            
        model_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
        print(f"Loading model: {repo_id}...")
        _notify_progress("loading", -1, f"Loading {model_name} into GPU...")
        
        try:
            # API MODE CHECK
            if config.API_MODE:
                print(f"DEBUG: API Mode Enabled. Connecting to {config.API_BASE_URL}...")
                from chatbot.api_client import OpenAIClientWrapper
                
                # Use configured API details
                client = OpenAIClientWrapper(
                    base_url=config.API_BASE_URL,
                    api_key=config.API_KEY,
                    model_name=config.API_MODEL_NAME
                )
                
                # Cache it so we don't re-init (though it's cheap)
                cls._instances[repo_id] = client
                _notify_progress("ready", 1.0, f"API: {config.API_MODEL_NAME}")
                return client

            # Local model loading requires llama-cpp-python
            if Llama is None:
                raise ImportError("llama-cpp-python is missing. Cannot load local model.")

            model_path = cls.ensure_model_path(repo_id)
            
            # Load with GPU offload
            # n_gpu_layers = -1 means 'all layers' (good for 3060 12GB)
            # n_ctx depends on usage (8192 for darkidol, 2048 for joints)
            
            # Check model file size for multi-GPU splitting logic
            file_stats = os.stat(model_path)
            file_size_gb = file_stats.st_size / (1024 * 1024 * 1024)
            
            # Default params
            split_mode = 1 # LLAMA_SPLIT_MODE_LAYER
            tensor_split = None
            
            # Auto-detect multi-GPU/large model scenario
            # If model is > 16GB (e.g. 32B model) and we have multiple GPUs
            try:
                import torch
                gpu_count = torch.cuda.device_count()
            except ImportError:
                gpu_count = 0
            
            # Default loading strategy
            load_candidates = [{"n_gpu_layers": n_gpu_layers, "use_mmap": True}]

            print(f"DEBUG: File Size: {file_size_gb:.2f}GB, GPUs: {gpu_count}")
            if file_size_gb > 16.0 and gpu_count > 1:
                print(f"⚠️ Large model detected ({file_size_gb:.1f} GB) with {gpu_count} GPUs.")
                print("   -> Activating Multi-GPU Tensor Split Mode (ROW SPLIT).")
                
                # Robust Dynamic VRAM Strategy
                # Calculates exact maximal layer count based on model size and VRAM
                try:
                    import torch
                    
                    if torch.cuda.is_available():
                        # 1. Get Free VRAM
                        # Use the minimum free memory across cards to avoid bottlenecking one
                        min_free_vram_gb = min([torch.cuda.mem_get_info(i)[0] for i in range(gpu_count)]) / (1024**3)
                        print(f"🔍 Dynamic Load: Avail VRAM per card: {min_free_vram_gb:.2f} GB")

                        # 2. Check Model Size
                        # If using the simplified ID, we might need to resolve it, but here we assume model_id is a path if it ends with .gguf
                        if model_path.endswith(".gguf") and os.path.exists(model_path):
                             model_size_gb = os.path.getsize(model_path) / (1024**3)
                        else:
                             # Fallback relative to hardcoded known sizes
                             model_size_gb = 20.0 
                        
                        print(f"   Model Size: {model_size_gb:.2f} GB")

                        # 3. Calculate Layers
                        # Qwen 32B has roughly 64 transformer layers
                        # We reserve VRAM for KV cache (context) and overhead
                        # 8192 context ~ 2GB KV cache (1GB per card) + 1GB buffer = 2GB safety per card
                        
                        # Estimate layer count from model size
                        # ~0.3GB per layer for 32B Q4, ~0.2GB for 14B, ~0.5GB for 70B
                        TOTAL_LAYERS = max(32, int(model_size_gb / 0.32))
                        LAYER_OVERHEAD = 0.5 # Extra space for output heads/norms
                        
                        # Size per layer (approx)
                        gb_per_layer = model_size_gb / (TOTAL_LAYERS + LAYER_OVERHEAD)
                        
                        # Safety Buffer (KV Cache + Desktop Overhead)
                        SAFETY_BUFFER_GB = 1.5 
                        
                        usable_vram_per_card = max(0, min_free_vram_gb - SAFETY_BUFFER_GB)
                        total_usable_vram = usable_vram_per_card * gpu_count
                        
                        # Max layers fitting in usable VRAM
                        calculated_layers = int(total_usable_vram / gb_per_layer)
                        
                        # Clamp
                        calculated_layers = max(0, min(TOTAL_LAYERS, calculated_layers))
                        
                        print(f"   Calculated Safe Layers: {calculated_layers} (Based on {total_usable_vram:.2f}GB usable VRAM)")

                        # Generate ONE optimal candidate + fallbacks
                        load_candidates = [
                            {"n_gpu_layers": calculated_layers, "use_mmap": False, "desc": f"Auto-Optimized ({calculated_layers} layers)"},
                            {"n_gpu_layers": max(0, calculated_layers - 8), "use_mmap": False, "desc": f"Safe Fallback ({max(0, calculated_layers - 8)} layers)"},
                            {"n_gpu_layers": 0, "use_mmap": True, "desc": "CPU Only", "hide_gpu": True}
                        ]
                    else:
                        print("⚠️ No CUDA, falling back to CPU candidates.")

                except Exception as e:
                     import traceback
                     traceback.print_exc()
                     print(f"⚠️ Dynamic sizing failed: {e}. Using conservative defaults.")
                     # Fallback to safe table
                     load_candidates = [
                        {"n_gpu_layers": 32, "use_mmap": False, "desc": "Fallback (32 layers)"},
                        {"n_gpu_layers": 16, "use_mmap": False, "desc": "Fallback (16 layers)"},
                        {"n_gpu_layers": 0,  "use_mmap": True,  "desc": "CPU Only"}
                     ]
            
            # Iterative Loading Loop - Strict
            last_error = None
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

            for load_config in load_candidates:
                current_layers = load_config.get("n_gpu_layers", n_gpu_layers)
                current_mmap = load_config.get("use_mmap", True)
                desc = load_config.get("desc", f"Standard ({current_layers} layers)")
                hide_gpu = load_config.get("hide_gpu", False)
                
                print(f"🔄 Attempting load: {desc}...")
                _notify_progress("loading", -1, f"Loading {model_name} ({desc})...")

                try:
                    # Clean up before attempt
                    try:
                        import gc
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as cleanup_err:
                        print(f"⚠️ Cleanup Warning: {cleanup_err}")

                    # Handle GPU hiding for CPU mode
                    if hide_gpu:
                        print("   -> Hiding GPUs to prevent CUDA OOM...")
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    elif original_cuda_visible is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                    elif "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]

                    llm = Llama(
                        model_path=model_path,
                        n_gpu_layers=current_layers,
                        n_ctx=n_ctx,
                        split_mode=split_mode,
                        tensor_split=tensor_split,
                        use_mmap=current_mmap,
                        verbose=True 
                    )
                    
                    # Restore env if changed
                    if hide_gpu:
                        if original_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        else:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

                    cls._instances[repo_id] = llm
                    _notify_progress("ready", 1.0, f"{model_name} ready")
                    print(f"✅ Model {repo_id} loaded successfully using: {desc}")
                    return llm

                except Exception as e:
                    print(f"⚠️ Load Failed ({desc}): {e}")
                    last_error = e
                    # Restore env for next attempt (e.g. going from CPU back to GPU? Unlikely order, but good practice)
                    if hide_gpu:
                         if original_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                         elif "CUDA_VISIBLE_DEVICES" in os.environ:
                            del os.environ["CUDA_VISIBLE_DEVICES"]
                    # Continue to next candidate
            
            # If we get here, all attempts failed
            _notify_progress("error", -1, f"Failed to load: {last_error}")
            print(f"❌ All load attempts failed for {repo_id}. Last error: {last_error}")
            raise last_error

        except Exception as e:
            # Catch-all for the outer try block (API mode, pathing, etc)
            print(f"❌ Model Manager Error: {e}")
            _notify_progress("error", -1, str(e))
            raise e

    @classmethod
    def close_all(cls):
        """Free memory."""
        cls._instances.clear()
        # Python GC should handle the rest if no refs remain
