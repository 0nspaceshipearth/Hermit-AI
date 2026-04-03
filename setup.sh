#!/usr/bin/env bash

# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

echo "=== Hermit Setup Script ==="
echo "Sets up the local AI environment with GPU support."
echo ""

# Ensure we are not running as root (prevents venv permission issues)
if [[ $EUID -eq 0 ]]; then
   echo "❌ Error: Please do NOT run this script with sudo."
   echo "The script will ask for your password only when necessary (app installation)."
   exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REQUIRE_LLAMA_GPU=0
for arg in "$@"; do
    case "$arg" in
        --gpu-required)
            REQUIRE_LLAMA_GPU=1
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [--gpu-required]"
            echo "  --gpu-required   Fail setup if llama.cpp GPU offload is unavailable."
            exit 0
            ;;
    esac
done
if [[ "${HERMIT_REQUIRE_LLAMA_GPU:-0}" == "1" ]]; then
    REQUIRE_LLAMA_GPU=1
fi
if [[ "${REQUIRE_LLAMA_GPU}" == "1" ]]; then
    echo "GPU mode: strict (--gpu-required enabled)"
    echo ""
fi

has_working_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null && nvidia-smi -L >/dev/null 2>&1
}

has_nvcc() {
    command -v nvcc &>/dev/null
}

configure_cuda_toolkit_env() {
    # Arch installs CUDA under /opt/cuda; make sure non-login shells can see it.
    if [[ -x "/opt/cuda/bin/nvcc" ]]; then
        export CUDA_HOME="/opt/cuda"
        export CUDAToolkit_ROOT="/opt/cuda"
        export CUDACXX="/opt/cuda/bin/nvcc"
        if [[ ":${PATH}:" != *":/opt/cuda/bin:"* ]]; then
            export PATH="/opt/cuda/bin:${PATH}"
        fi
        if [[ -d "/opt/cuda/lib64" && ":${LD_LIBRARY_PATH:-}:" != *":/opt/cuda/lib64:"* ]]; then
            export LD_LIBRARY_PATH="/opt/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
    fi
}

export_venv_cuda_libs() {
    local lib_dirs=()
    local lib_dir
    shopt -s nullglob
    lib_dirs=(./venv/lib/python*/site-packages/nvidia/*/lib)
    shopt -u nullglob

    for lib_dir in "${lib_dirs[@]}"; do
        if [[ ":${LD_LIBRARY_PATH:-}:" != *":${lib_dir}:"* ]]; then
            export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
    done
}

llama_gpu_offload_supported() {
    ./venv/bin/python - <<'PY' >/dev/null
try:
    from llama_cpp import llama_supports_gpu_offload
    raise SystemExit(0 if llama_supports_gpu_offload() else 1)
except Exception:
    raise SystemExit(1)
PY
}

llama_cuda_cmake_args() {
    local args="-DGGML_CUDA=ON"
    if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
        args+=" -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
    fi
    if [[ -n "${CUDACXX:-}" ]]; then
        args+=" -DCMAKE_CUDA_COMPILER=${CUDACXX}"
    fi
    printf "%s" "$args"
}

install_llama_cuda_from_source() {
    local cmake_args
    cmake_args="$(llama_cuda_cmake_args)"
    CMAKE_ARGS="${cmake_args}" FORCE_CMAKE=1 ./venv/bin/python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2>&1 | tee -a setup.log
}

# Cleanup artifacts
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 1. System Packages
echo "[1/5] Checking System Prerequisites..."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_LIKE=${ID_LIKE:-""}
else
    # Fallback if os-release is not found (e.g., macOS)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    OS_LIKE=""
fi

if [[ "$OS" == "ubuntu" || "$OS" == "debian" || "$OS_LIKE" == *"debian"* || "$OS_LIKE" == *"ubuntu"* ]]; then
    echo "  -> Detected Debian/Ubuntu based OS"
    sudo apt update || echo "⚠️ Warning: apt update failed, but proceeding..."
    sudo apt install -y python3 python3-venv python3-full python3-tk python3-libzim curl cmake build-essential
elif [[ "$OS" == "arch" || "$OS" == "endeavouros" || "$OS_LIKE" == *"arch"* ]]; then
    echo "  -> Detected Arch Linux based OS"
    # Decide which Pytorch package to use
    PYTORCH_PKG="python-pytorch-opt"
    if has_working_nvidia_gpu; then
        echo "  -> NVIDIA GPU detected. Using CUDA-optimized PyTorch."
        PYTORCH_PKG="python-pytorch-opt-cuda"
    fi
    # Arch needs python-tk for tkinter and base-devel for compilation
    sudo pacman -Sy --needed --noconfirm python tk curl cmake base-devel libzim $PYTORCH_PKG
elif [[ "$OS" == "fedora" || "$OS_LIKE" == *"fedora"* ]]; then
    echo "  -> Detected Fedora based OS"
    sudo dnf install -y python3 python3-tkinter curl cmake gcc-c++ make
elif [[ "$OS" == "darwin" ]]; then
    echo "  -> Detected macOS"
    if ! command -v brew &> /dev/null; then
        echo "❌ Error: Homebrew is required on macOS. Please install it from https://brew.sh/"
        exit 1
    fi
    brew install python tcl-tk curl cmake
else
    echo "⚠️ Warning: Unknown OS '$OS'. Please assure you have Python3, Tkinter, CMake, and build tools installed."
fi

echo "✓ System packages verified"

# 2. Virtual Environment
echo "[2/5] Setting up Virtual Environment..."
if [ -d "venv" ]; then
    # Check if the venv is valid (it might have been moved or come from a different OS)
    if ! ./venv/bin/python3 --version &>/dev/null || ! ./venv/bin/python3 -m pip --version &>/dev/null; then
        echo "⚠️  Existing virtual environment is invalid (likely moved). Recreating..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    VENV_OPTS=""
    if [[ "$OS" == "arch" || "$OS" == "endeavouros" || "$OS_LIKE" == *"arch"* ]]; then
        # On Arch/Python 3.14, we use system-installed PyTorch
        echo "  -> Using --system-site-packages for system PyTorch compatibility"
        VENV_OPTS="--system-site-packages"
    fi
    python3 -m venv $VENV_OPTS venv
    echo "✓ Virtual environment created"
fi

# Upgrade pip
./venv/bin/python3 -m pip install --upgrade pip

# Trap errors
trap 'echo "❌ Error on line $LINENO. Check setup.log for details."; exit 1' ERR

# 3. GPU support (PyTorch CUDA 12.1 - pinned for stability)
echo "[3/5] Installing PyTorch with CUDA support..."

# Skip pinned pip install if PyTorch is already available (e.g. from system packages on Arch)
if python3 -c "import torch; print(torch.__version__)" &>/dev/null || ./venv/bin/python3 -c "import torch; print(torch.__version__)" &>/dev/null; then
    echo "✓ PyTorch already installed (Found $(./venv/bin/python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || python3 -c 'import torch; print(torch.__version__)'))"
else
    # Retry logic for large downloads
    max_retries=3
    count=0
    while [ $count -lt $max_retries ]; do
        if ./venv/bin/python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | tee -a setup.log; then
            break
        fi
        count=$((count+1))
        echo "⚠️ Download failed, retrying ($count/$max_retries)..."
        sleep 2
    done
    if [ $count -eq $max_retries ]; then
        echo "❌ Failed to install PyTorch after retries."
        exit 1
    fi
    echo "✓ PyTorch (CUDA) installed"
fi

# 4. Core Dependencies
echo "[4/5] Installing Project Dependencies..."
# Install everything except llama-cpp-python first to avoid unoptimized builds
grep -v "llama-cpp-python" requirements.txt > requirements_temp.txt
./venv/bin/python3 -m pip install -r requirements_temp.txt 2>&1 | tee -a setup.log
rm requirements_temp.txt

# hf-xet/hf_transfer can fail in some environments; force stable HTTP transport.
# Keep packages installed to avoid dependency churn/warnings from huggingface-hub.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "✓ Hugging Face transport configured (HTTP mode)"

echo "[4.5/5] Installing llama-cpp-python..."
if has_working_nvidia_gpu; then
    configure_cuda_toolkit_env
    echo "  -> NVIDIA GPU detected. Installing with CUDA support..."
    # Try CUDA wheels first. If unavailable for this Python/platform, fall back safely.
    if ! ./venv/bin/python3 -m pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
        --upgrade --force-reinstall --no-cache-dir 2>&1 | tee -a setup.log; then
        if has_nvcc; then
            echo "  -> CUDA wheel unavailable; trying source build with nvcc..."
            if ! install_llama_cuda_from_source; then
                echo "  -> CUDA source build failed. Falling back to CPU-only build."
                ./venv/bin/python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2>&1 | tee -a setup.log
            fi
        else
            echo "  -> CUDA wheel unavailable and nvcc not found. Installing CPU-only build."
            ./venv/bin/python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2>&1 | tee -a setup.log
        fi
    fi

    # If GPU offload still isn't available, only attempt CUDA rebuild when nvcc exists.
    if ! llama_gpu_offload_supported && has_nvcc; then
        echo "  -> Installed build does not expose GPU offload. Rebuilding from source with CUDA..."
        if ! install_llama_cuda_from_source; then
            echo "  -> CUDA rebuild failed. Continuing with CPU-only llama-cpp-python."
        fi
    fi

    if llama_gpu_offload_supported; then
        echo "✓ llama-cpp-python (CUDA) installed"
    else
        if has_nvcc; then
            echo "⚠️  llama-cpp-python installed, but GPU offload is still unavailable."
            echo "    Check NVIDIA driver health, CUDA toolkit compatibility, and setup.log."
            echo "    Rebuild manually:"
            echo "    CMAKE_ARGS='$(llama_cuda_cmake_args)' FORCE_CMAKE=1 ./venv/bin/python3 -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python"
        else
            echo "⚠️  NVIDIA driver detected but CUDA toolkit (nvcc) is missing."
            echo "    Installed CPU-only llama-cpp-python to keep setup working."
            echo "    Install CUDA toolkit if you want llama.cpp GPU offload."
        fi
    fi
else
    echo "  -> No working NVIDIA GPU runtime detected. Installing CPU-only version..."
    ./venv/bin/python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir 2>&1 | tee -a setup.log
    echo "✓ llama-cpp-python (CPU) installed"
fi

if [[ "${REQUIRE_LLAMA_GPU}" == "1" ]] && ! llama_gpu_offload_supported; then
    echo "❌ --gpu-required was set, but llama.cpp GPU offload is unavailable."
    echo "   Check setup.log and verify CUDA toolkit + driver compatibility."
    exit 1
fi


# 5. Download Models
echo "[5/6] Downloading AI Models..."
echo "This may take a while depending on your internet connection."
# Use stable Hugging Face HTTP download path.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# Ensure CUDA libraries from the venv are in the path
export_venv_cuda_libs
./venv/bin/python download_models.py
echo "✓ Models downloaded"

# 6. Check Resources
echo "[6/6] Checking Resources..."
ZIM_FILE=$(find . -maxdepth 1 -name "*.zim" | head -n 1)
if [ -z "$ZIM_FILE" ]; then
    echo "⚠️  No .zim file found. Place your Wikipedia ZIM file here."
fi

SHARED_MODELS="shared_models"
mkdir -p "$SHARED_MODELS"
echo "✓ Model directory verified: $SHARED_MODELS"

# Global Commands
echo ""
echo "Enabling global commands..."
INSTALL_SCOPE="user"
if command -v sudo >/dev/null 2>&1; then
    if sudo -v; then
        INSTALL_SCOPE="system"
    else
        echo "⚠️  Could not refresh sudo credentials."
        echo "    Falling back to user-local command/desktop install."
    fi
fi

if [[ "$INSTALL_SCOPE" == "system" ]]; then
    HERMIT_WRAPPER="/usr/local/bin/hermit"
    sudo tee "$HERMIT_WRAPPER" > /dev/null << HERMIT_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/run_chatbot.sh" "\$@"
HERMIT_EOF
    sudo chmod +x "$HERMIT_WRAPPER"

    FORGE_WRAPPER="/usr/local/bin/forge"
    sudo tee "$FORGE_WRAPPER" > /dev/null << FORGE_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/venv/bin/python" "\$INSTALL_DIR/forge.py" "\$@"
FORGE_EOF
    sudo chmod +x "$FORGE_WRAPPER"

    # Desktop Integration
    echo "Configuring Desktop Integration..."
    if [ -f "assets/icon.png" ]; then
        sudo cp "assets/icon.png" "/usr/share/pixmaps/hermit.png" || true
    fi

    HERMIT_DESKTOP="/usr/share/applications/hermit.desktop"
    sudo tee "$HERMIT_DESKTOP" > /dev/null << HERMIT_ENTRY
[Desktop Entry]
Name=Hermit AI
Comment=Offline AI Chatbot
Exec=hermit
Icon=hermit
Type=Application
Terminal=false
Categories=Education;Science;Utility;AI;
HERMIT_ENTRY

    if command -v update-desktop-database &> /dev/null; then
        sudo update-desktop-database > /dev/null 2>&1 || true
    fi
    RUN_HINT="hermit"
else
    LOCAL_BIN="${HOME}/.local/bin"
    LOCAL_APPS="${HOME}/.local/share/applications"
    LOCAL_PIXMAPS="${HOME}/.local/share/pixmaps"
    mkdir -p "$LOCAL_BIN" "$LOCAL_APPS" "$LOCAL_PIXMAPS"

    HERMIT_WRAPPER="${LOCAL_BIN}/hermit"
    tee "$HERMIT_WRAPPER" > /dev/null << HERMIT_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/run_chatbot.sh" "\$@"
HERMIT_EOF
    chmod +x "$HERMIT_WRAPPER"

    FORGE_WRAPPER="${LOCAL_BIN}/forge"
    tee "$FORGE_WRAPPER" > /dev/null << FORGE_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/venv/bin/python" "\$INSTALL_DIR/forge.py" "\$@"
FORGE_EOF
    chmod +x "$FORGE_WRAPPER"

    echo "Configuring Desktop Integration (user-local)..."
    ICON_TARGET="${LOCAL_PIXMAPS}/hermit.png"
    if [ -f "assets/icon.png" ]; then
        cp "assets/icon.png" "$ICON_TARGET" || true
    fi

    HERMIT_DESKTOP="${LOCAL_APPS}/hermit.desktop"
    tee "$HERMIT_DESKTOP" > /dev/null << HERMIT_ENTRY
[Desktop Entry]
Name=Hermit AI
Comment=Offline AI Chatbot
Exec=${HERMIT_WRAPPER}
Icon=${ICON_TARGET}
Type=Application
Terminal=false
Categories=Education;Science;Utility;AI;
HERMIT_ENTRY

    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$LOCAL_APPS" > /dev/null 2>&1 || true
    fi

    if [[ ":$PATH:" != *":${LOCAL_BIN}:"* ]]; then
        echo "⚠️  ${LOCAL_BIN} is not in PATH for this shell."
        echo "    You can run Hermit with: ${HERMIT_WRAPPER}"
    fi
    RUN_HINT="${HERMIT_WRAPPER}"
fi

echo "=== Setup Complete! ==="
echo "Run with: ${RUN_HINT}"
