
"""
Model Manager for Local Inference.
Handles downloading GGUF models from Hugging Face and loading them via llama-cpp-python.
"""

import glob
import os
import struct
import subprocess
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    import llama_cpp
    from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER, llama_supports_gpu_offload
except ImportError:
    llama_cpp = None
    Llama = None
    LLAMA_SPLIT_MODE_LAYER = 1
    llama_supports_gpu_offload = None
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

    timeout = (10, 60)
    max_retries = 3
    detail = ProgressTqdm._default_desc

    for attempt in range(1, max_retries + 1):
        downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
        write_mode = "ab" if downloaded > 0 else "wb"
        total_bytes = 0

        try:
            with requests.get(
                url,
                stream=True,
                allow_redirects=True,
                timeout=timeout,
                headers=headers
            ) as response:
                if downloaded > 0 and response.status_code == 200:
                    # Server ignored Range; restart from scratch.
                    downloaded = 0
                    write_mode = "wb"
                response.raise_for_status()

                content_range = response.headers.get("content-range")
                content_length = int(response.headers.get("content-length", "0") or 0)
                if content_range and "/" in content_range:
                    try:
                        total_bytes = int(content_range.rsplit("/", 1)[-1])
                    except ValueError:
                        total_bytes = 0
                elif content_length > 0:
                    total_bytes = downloaded + content_length

                if total_bytes > 0:
                    _notify_progress("downloading", ProgressTqdm._progress_offset, detail)
                    print(
                        f"  -> HTTP fallback download: {_format_size(downloaded)}"
                        f"/{_format_size(total_bytes)}"
                    )
                else:
                    print(f"  -> HTTP fallback download: {_format_size(downloaded)} downloaded")

                last_log = time.time()
                last_bytes = downloaded

                with open(tmp_path, write_mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes > 0:
                            shard_progress = min(1.0, downloaded / total_bytes)
                            progress = ProgressTqdm._progress_offset + (shard_progress * ProgressTqdm._progress_scale)
                            _notify_progress("downloading", min(1.0, progress), detail)

                        now = time.time()
                        should_log = (now - last_log >= 10) or (downloaded - last_bytes >= 256 * 1024 * 1024)
                        if should_log:
                            if total_bytes > 0:
                                pct = (downloaded / total_bytes) * 100.0
                                print(
                                    f"  -> HTTP fallback progress: {pct:.1f}% "
                                    f"({_format_size(downloaded)}/{_format_size(total_bytes)})"
                                )
                            else:
                                print(f"  -> HTTP fallback progress: {_format_size(downloaded)} downloaded")
                            last_log = now
                            last_bytes = downloaded

            if total_bytes > 0 and downloaded < total_bytes:
                raise IOError(
                    f"Incomplete download for {filename}: got {downloaded}/{total_bytes} bytes"
                )

            os.replace(tmp_path, target_path)
            print(
                f"  -> HTTP fallback complete: {_format_size(downloaded)} "
                f"saved to {target_path}"
            )
            return target_path

        except Exception as e:
            if attempt == max_retries:
                if os.path.exists(tmp_path):
                    print(
                        f"⚠️ HTTP fallback stopped with partial file kept for resume: {tmp_path}"
                    )
                raise
            print(
                f"⚠️ HTTP fallback interrupted (attempt {attempt}/{max_retries}): {e}. "
                "Retrying with resume..."
            )
            time.sleep(min(5 * attempt, 15))

    raise RuntimeError(f"Failed to download {filename} after retries")


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


FAST_QUANT_PREFERENCES = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "F16",
]

_GGUF_SCALAR_FORMATS = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}

_KNOWN_UNSUPPORTED_ARCH_HINTS = {
    "qwen35moe": (
        "This GGUF uses the qwen35moe architecture. The bundled llama-cpp-python "
        "runtime cannot load it in embedded mode. Use Hermit's API mode with an "
        "external backend that supports Qwen 3.5 MoE, or rebuild llama-cpp-python "
        "against a newer llama.cpp."
    ),
}


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of GGUF file")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_gguf_string(handle) -> str:
    length = _read_u64(handle)
    if length == 0:
        return ""
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_gguf_value(handle, value_type: int) -> None:
    if value_type == 8:
        length = _read_u64(handle)
        handle.seek(length, os.SEEK_CUR)
        return

    if value_type == 9:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if element_type in _GGUF_SCALAR_FORMATS:
            handle.seek(struct.calcsize(_GGUF_SCALAR_FORMATS[element_type]) * count, os.SEEK_CUR)
            return
        for _ in range(count):
            _skip_gguf_value(handle, element_type)
        return

    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")
    handle.seek(struct.calcsize(fmt), os.SEEK_CUR)


def _read_gguf_value(handle, value_type: int) -> Any:
    if value_type == 8:
        return _read_gguf_string(handle)

    if value_type == 9:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if element_type in _GGUF_SCALAR_FORMATS:
            size = struct.calcsize(_GGUF_SCALAR_FORMATS[element_type])
            handle.seek(size * count, os.SEEK_CUR)
            return None
        values = []
        for _ in range(count):
            values.append(_read_gguf_value(handle, element_type))
        return values

    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


@lru_cache(maxsize=64)
def _inspect_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """Read a few useful GGUF metadata keys without loading the full model."""
    wanted = {"general.architecture", "general.name", "general.size_label"}
    metadata: Dict[str, Any] = {}

    with open(model_path, "rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {model_path}")

        version = _read_u32(handle)
        if version < 2:
            raise ValueError(f"Unsupported GGUF version {version} in {model_path}")

        _ = _read_u64(handle)  # tensor_count
        kv_count = _read_u64(handle)
        architecture: Optional[str] = None

        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            value_type = _read_u32(handle)

            if architecture and key in {f"{architecture}.block_count", f"{architecture}.context_length"}:
                metadata[key] = _read_gguf_value(handle, value_type)
            elif key in wanted:
                value = _read_gguf_value(handle, value_type)
                metadata[key] = value
                if key == "general.architecture" and isinstance(value, str):
                    architecture = value
                    wanted.update({f"{architecture}.block_count", f"{architecture}.context_length"})
            else:
                _skip_gguf_value(handle, value_type)

            if wanted.issubset(metadata.keys()):
                break

    return metadata


def _detect_gpu_inventory() -> List[Dict[str, Any]]:
    """Return visible NVIDIA GPUs with current free VRAM when available."""
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        gpus: List[Dict[str, Any]] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) != 4:
                continue
            index_str, total_str, free_str, name = parts
            gpus.append(
                {
                    "index": int(index_str),
                    "memory_total_gb": float(total_str) / 1024.0,
                    "memory_free_gb": float(free_str) / 1024.0,
                    "name": name,
                }
            )
        if gpus:
            return gpus
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            gpus = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                gpus.append(
                    {
                        "index": index,
                        "memory_total_gb": total_bytes / (1024 ** 3),
                        "memory_free_gb": free_bytes / (1024 ** 3),
                        "name": props.name,
                    }
                )
            return gpus
    except Exception:
        pass

    return []


@lru_cache(maxsize=1)
def _llama_gpu_offload_enabled() -> bool:
    if llama_supports_gpu_offload is None:
        return False
    try:
        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def _recommended_contexts(requested_n_ctx: int, file_size_gb: float, trained_ctx: int) -> List[int]:
    contexts = [requested_n_ctx]
    if file_size_gb >= 18.0:
        contexts.extend([6144, 4096, 3072, 2048])
    elif file_size_gb >= 10.0:
        contexts.extend([6144, 4096])

    deduped: List[int] = []
    seen = set()
    for ctx in contexts:
        effective_ctx = ctx
        if trained_ctx > 0:
            effective_ctx = min(effective_ctx, trained_ctx)
        if effective_ctx < 512:
            continue
        if effective_ctx not in seen:
            seen.add(effective_ctx)
            deduped.append(effective_ctx)
    return deduped or [requested_n_ctx]


def _reserve_vram_per_gpu(n_ctx: int) -> float:
    return 1.0 + min(2.0, (n_ctx / 4096.0) * 0.6)


def _compute_tensor_split(gpus: List[Dict[str, Any]], n_ctx: int) -> Optional[List[float]]:
    if len(gpus) < 2:
        return None

    reserve = _reserve_vram_per_gpu(n_ctx)
    usable = [max(0.0, gpu["memory_free_gb"] - reserve) for gpu in gpus]
    total_usable = sum(usable)
    if total_usable <= 0:
        return None
    return [value / total_usable for value in usable]


def _recommend_gpu_layers(
    file_size_gb: float,
    total_layers: int,
    gpus: List[Dict[str, Any]],
    n_ctx: int,
    requested_n_gpu_layers: int,
) -> int:
    if requested_n_gpu_layers >= 0:
        return requested_n_gpu_layers

    if not gpus:
        return 0

    reserve = _reserve_vram_per_gpu(n_ctx)
    usable_total = sum(max(0.0, gpu["memory_free_gb"] - reserve) for gpu in gpus)
    if usable_total <= 0:
        return 0

    if usable_total >= (file_size_gb + 0.5):
        return -1

    gb_per_layer = file_size_gb / max(total_layers, 1)
    candidate_layers = int(usable_total / max(gb_per_layer, 0.01))
    return max(0, min(total_layers, candidate_layers))


def _recommended_batch_sizes(file_size_gb: float, n_ctx: int) -> Tuple[int, int]:
    if file_size_gb >= 12.0:
        target_n_batch = int(getattr(config, "LLAMA_LARGE_MODEL_N_BATCH", 768))
        target_n_ubatch = int(getattr(config, "LLAMA_LARGE_MODEL_N_UBATCH", target_n_batch))
    else:
        target_n_batch = int(getattr(config, "LLAMA_SMALL_MODEL_N_BATCH", 2048))
        target_n_ubatch = int(getattr(config, "LLAMA_SMALL_MODEL_N_UBATCH", 1024))

    tuned_n_batch = max(64, min(int(n_ctx), target_n_batch))
    tuned_n_ubatch = max(64, min(tuned_n_batch, target_n_ubatch))
    return tuned_n_batch, tuned_n_ubatch


def _build_load_candidates(
    file_size_gb: float,
    total_layers: int,
    gpus: List[Dict[str, Any]],
    requested_n_ctx: int,
    requested_n_gpu_layers: int,
    trained_ctx: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    gpu_enabled = _llama_gpu_offload_enabled() and bool(gpus)
    contexts = _recommended_contexts(requested_n_ctx, file_size_gb, trained_ctx)

    if not gpu_enabled:
        cpu_ctx = contexts[0]
        cpu_n_batch, cpu_n_ubatch = _recommended_batch_sizes(8.0, cpu_ctx)
        return [
            {
                "n_ctx": cpu_ctx,
                "n_gpu_layers": 0,
                "n_batch": max(64, min(cpu_n_batch, 512)),
                "n_ubatch": max(64, min(cpu_n_ubatch, 256)),
                "flash_attn": False,
                "offload_kqv": False,
                "op_offload": False,
                "split_mode": LLAMA_SPLIT_MODE_LAYER,
                "tensor_split": None,
                "use_mmap": True,
                "desc": f"{cpu_ctx} ctx / CPU only",
                "main_gpu": 0,
                "gpu_summary": "CPU only",
                "hide_gpu": True,
            }
        ]

    for index, ctx in enumerate(contexts):
        n_batch, n_ubatch = _recommended_batch_sizes(file_size_gb, ctx)
        tensor_split = _compute_tensor_split(gpus, ctx) if gpu_enabled else None
        candidate_layers = _recommend_gpu_layers(
            file_size_gb=file_size_gb,
            total_layers=total_layers,
            gpus=gpus if gpu_enabled else [],
            n_ctx=ctx,
            requested_n_gpu_layers=requested_n_gpu_layers,
        )

        desc = f"{ctx} ctx / {'all' if candidate_layers == -1 else candidate_layers} GPU layers"
        candidates.append(
            {
                "n_ctx": ctx,
                "n_gpu_layers": candidate_layers if gpu_enabled else 0,
                "n_batch": n_batch,
                "n_ubatch": n_ubatch,
                "flash_attn": gpu_enabled,
                "offload_kqv": gpu_enabled,
                "op_offload": gpu_enabled,
                "split_mode": LLAMA_SPLIT_MODE_LAYER,
                "tensor_split": tensor_split,
                "use_mmap": True,
                "desc": desc,
                "main_gpu": max(gpus, key=lambda item: item["memory_free_gb"])["index"] if gpu_enabled else 0,
                "gpu_summary": ", ".join(
                    f"GPU{gpu['index']} {gpu['memory_free_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB"
                    for gpu in gpus
                ) if gpu_enabled else "CPU only",
            }
        )

        if gpu_enabled and index == 0 and candidate_layers not in (-1, 0):
            safe_layers = max(0, candidate_layers - 6)
            candidates.append(
                {
                    "n_ctx": ctx,
                    "n_gpu_layers": safe_layers,
                    "n_batch": max(64, min(n_batch, 512)),
                    "n_ubatch": max(64, min(n_ubatch, 256)),
                    "flash_attn": True,
                    "offload_kqv": True,
                    "op_offload": True,
                    "split_mode": LLAMA_SPLIT_MODE_LAYER,
                    "tensor_split": tensor_split,
                    "use_mmap": index == 0,
                    "desc": f"{ctx} ctx / safe fallback ({safe_layers} GPU layers)",
                    "main_gpu": max(gpus, key=lambda item: item["memory_free_gb"])["index"],
                    "gpu_summary": ", ".join(
                        f"GPU{gpu['index']} {gpu['memory_free_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB"
                        for gpu in gpus
                    ),
                }
            )

        if len(candidates) >= 4:
            break

    cpu_ctx = min(contexts[-1], requested_n_ctx)
    cpu_n_batch, cpu_n_ubatch = _recommended_batch_sizes(8.0, cpu_ctx)
    candidates.append(
        {
            "n_ctx": cpu_ctx,
            "n_gpu_layers": 0,
            "n_batch": max(64, min(cpu_n_batch, 512)),
            "n_ubatch": max(64, min(cpu_n_ubatch, 256)),
            "flash_attn": False,
            "offload_kqv": False,
            "op_offload": False,
            "split_mode": LLAMA_SPLIT_MODE_LAYER,
            "tensor_split": None,
            "use_mmap": True,
            "desc": f"{cpu_ctx} ctx / CPU only",
            "main_gpu": 0,
            "gpu_summary": "CPU only",
            "hide_gpu": True,
        }
    )
    return candidates

class ModelManager:
    """Singleton manager for local LLM models."""

    _instances: Dict[str, Any] = {}
    _active_model: Optional[Dict[str, Any]] = None

    @staticmethod
    def _is_large_model_id(model_id: str) -> bool:
        lower = (model_id or "").lower()
        return any(marker in lower for marker in ("32b", "34b", "70b", "72b", "405b"))
    
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
        
        # Speed-first quant preference. Large models are much more usable at Q4 than Q5.
        preferences = FAST_QUANT_PREFERENCES
        
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
    def get_model(cls, repo_id: str, n_ctx: int = 8192, n_gpu_layers: int = -1, prefer_default: bool = False) -> 'Llama':
        """
        Get or load a Llama model instance.
        Enforces single-model policy to prevent VRAM OOM.
        Uses 8192 context by default to accommodate RAG content.
        """
        model_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id

        try:
            # API MODE CHECK
            if config.API_MODE:
                api_cache_key = f"api::{config.API_BASE_URL.rstrip('/')}::{config.API_MODEL_NAME}"
                if cls._active_model and cls._active_model.get("cache_key") == api_cache_key:
                    active_instance = cls._instances.get(api_cache_key)
                    if active_instance is not None:
                        return active_instance

                cls.close_all()
                print(f"DEBUG: API Mode Enabled. Connecting to {config.API_BASE_URL}...")
                from chatbot.api_client import OpenAIClientWrapper

                client = OpenAIClientWrapper(
                    base_url=config.API_BASE_URL,
                    api_key=config.API_KEY,
                    model_name=config.API_MODEL_NAME
                )

                cls._instances[api_cache_key] = client
                cls._active_model = {
                    "cache_key": api_cache_key,
                    "kind": "api",
                }
                _notify_progress("ready", 1.0, f"API: {config.API_MODEL_NAME}")
                return client

            # Local model loading requires llama-cpp-python
            if Llama is None:
                raise ImportError("llama-cpp-python is missing. Cannot load local model.")

            model_path = cls.ensure_model_path(repo_id)

            metadata = {}
            architecture = "unknown"
            trained_ctx = 0
            total_layers = 0
            try:
                metadata = _inspect_gguf_metadata(model_path)
                architecture = str(metadata.get("general.architecture") or "unknown")
                if architecture != "unknown":
                    trained_ctx = int(metadata.get(f"{architecture}.context_length") or 0)
                    total_layers = int(metadata.get(f"{architecture}.block_count") or 0)
            except Exception as metadata_error:
                print(f"⚠️ GGUF metadata inspection failed for {model_path}: {metadata_error}")

            file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
            if total_layers <= 0:
                total_layers = max(24, int(round(file_size_gb / 0.32)))

            if cls._active_model:
                active = cls._active_model
                if active.get("kind") == "local" and active.get("model_path") == model_path:
                    active_instance = cls._instances.get(active.get("cache_key", ""))
                    requested_ctx = int(active.get("requested_n_ctx", 0))
                    if active_instance is not None and (
                        int(active.get("n_ctx", 0)) >= n_ctx or requested_ctx == n_ctx
                    ):
                        print(
                            f"Reusing loaded model {model_name} "
                            f"(active ctx={active.get('n_ctx')}, requested ctx={n_ctx})"
                        )
                        return active_instance

                if prefer_default:
                    default_id = getattr(config, "DEFAULT_MODEL", None)
                    active_repo_id = active.get("repo_id")
                    active_instance = cls._instances.get(active.get("cache_key", ""))
                    block_large = getattr(config, "BLOCK_LARGE_DEFAULT_REUSE_FOR_AUX", True)
                    if (
                        active_instance is not None
                        and default_id
                        and active_repo_id == default_id
                        and (not block_large or not cls._is_large_model_id(default_id))
                    ):
                        print(f"Reusing active default model for auxiliary task: {default_id}")
                        return active_instance

            cls.close_all()

            gpus = _detect_gpu_inventory()
            gpu_enabled = _llama_gpu_offload_enabled() and bool(gpus)
            load_candidates = _build_load_candidates(
                file_size_gb=file_size_gb,
                total_layers=total_layers,
                gpus=gpus,
                requested_n_ctx=n_ctx,
                requested_n_gpu_layers=n_gpu_layers,
                trained_ctx=trained_ctx,
            )

            print(f"Loading model: {repo_id}...")
            print(
                f"DEBUG: arch={architecture}, layers={total_layers}, trained_ctx={trained_ctx or 'unknown'}, "
                f"file_size={file_size_gb:.2f}GB, gpu_backend={gpu_enabled}, gpus={len(gpus)}"
            )
            _notify_progress("loading", -1, f"Loading {model_name}...")

            last_error = None
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            cpu_count = os.cpu_count() or 4
            configured_threads = int(getattr(config, "LLAMA_THREADS", 0) or 0)
            configured_threads_batch = int(getattr(config, "LLAMA_THREADS_BATCH", 0) or 0)
            n_threads = configured_threads if configured_threads > 0 else max(1, cpu_count - 1)
            n_threads_batch = configured_threads_batch if configured_threads_batch > 0 else max(1, cpu_count)

            for load_config in load_candidates:
                current_layers = load_config.get("n_gpu_layers", n_gpu_layers)
                current_ctx = load_config.get("n_ctx", n_ctx)
                current_mmap = load_config.get("use_mmap", True)
                desc = load_config.get("desc", f"Standard ({current_layers} layers)")
                hide_gpu = load_config.get("hide_gpu", False)
                current_tensor_split = load_config.get("tensor_split")

                print(
                    f"🔄 Attempting load: {desc} | batch={load_config.get('n_batch')} "
                    f"| ubatch={load_config.get('n_ubatch')} | flash={load_config.get('flash_attn')} "
                    f"| mmap={current_mmap} | {load_config.get('gpu_summary')}"
                )
                _notify_progress("loading", -1, f"Loading {model_name} ({desc})...")

                try:
                    try:
                        import gc

                        gc.collect()
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception as cleanup_err:
                        print(f"⚠️ Cleanup Warning: {cleanup_err}")

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
                        n_ctx=current_ctx,
                        n_batch=load_config.get("n_batch"),
                        n_ubatch=load_config.get("n_ubatch"),
                        n_threads=n_threads,
                        n_threads_batch=n_threads_batch,
                        split_mode=load_config.get("split_mode", LLAMA_SPLIT_MODE_LAYER),
                        tensor_split=current_tensor_split,
                        main_gpu=load_config.get("main_gpu", 0),
                        use_mmap=current_mmap,
                        flash_attn=load_config.get("flash_attn", False),
                        offload_kqv=load_config.get("offload_kqv", True),
                        op_offload=load_config.get("op_offload"),
                        verbose=True
                    )

                    if hide_gpu:
                        if original_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        else:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

                    cache_key = (
                        f"local::{model_path}::{current_ctx}::{current_layers}:"
                        f"{int(bool(load_config.get('flash_attn')))}"
                    )
                    cls._instances[cache_key] = llm
                    cls._active_model = {
                        "cache_key": cache_key,
                        "kind": "local",
                        "model_path": model_path,
                        "repo_id": repo_id,
                        "n_ctx": current_ctx,
                        "requested_n_ctx": n_ctx,
                        "n_gpu_layers": current_layers,
                        "architecture": architecture,
                    }
                    _notify_progress("ready", 1.0, f"{model_name} ready")
                    print(
                        f"✅ Model {repo_id} loaded successfully using: {desc} "
                        f"(arch={architecture}, ctx={current_ctx}, gpu_layers={current_layers})"
                    )
                    return llm

                except Exception as e:
                    print(f"⚠️ Load Failed ({desc}): {e}")
                    last_error = e
                    if hide_gpu:
                        if original_cuda_visible is not None:
                            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        elif "CUDA_VISIBLE_DEVICES" in os.environ:
                            del os.environ["CUDA_VISIBLE_DEVICES"]

            _notify_progress("error", -1, f"Failed to load: {last_error}")
            print(f"❌ All load attempts failed for {repo_id}. Last error: {last_error}")

            last_error_text = str(last_error or "").lower()
            arch_hint = _KNOWN_UNSUPPORTED_ARCH_HINTS.get(architecture)
            backend_version = getattr(llama_cpp, "__version__", "unknown")
            if arch_hint and (
                "failed to load model from file" in last_error_text
                or "unknown model architecture" in last_error_text
                or backend_version == "0.3.16"
            ):
                raise RuntimeError(
                    f"{arch_hint} Architecture: {architecture}. "
                    f"Bundled backend: llama-cpp-python {backend_version}. "
                    f"Original loader error: {last_error}"
                )

            raise last_error

        except Exception as e:
            print(f"❌ Model Manager Error: {e}")
            _notify_progress("error", -1, str(e))
            raise e

    @classmethod
    def close_all(cls):
        """Free memory."""
        if not cls._instances:
            cls._active_model = None
            return

        print(f"Unloading {len(cls._instances)} active model instance(s)...")
        import gc

        for key, model_instance in list(cls._instances.items()):
            print(f"Unloading model: {key}")
            close_fn = getattr(model_instance, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as close_error:
                    print(f"⚠️ Model close warning for {key}: {close_error}")

        cls._instances.clear()
        cls._active_model = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
