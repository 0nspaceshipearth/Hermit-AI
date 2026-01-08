#!/usr/bin/env bash
set -euo pipefail

echo "=== Hermit Setup Script ==="
echo "Sets up the local AI environment with GPU support."
echo ""

# Check configured directory for models
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup artifacts (prevent stale bytecode issues)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check sudo access (for system packages)
if ! sudo -n true 2>/dev/null; then
    echo "Note: Sudo access required for system packages (python3, libzim)."
fi

# 1. System Packages
echo "[1/5] Installing System Prerequisites..."
sudo apt update -qq
# Added cmake and build-essential for compiling llama-cpp-python
sudo apt install -y python3 python3-venv python3-full python3-tk python3-libzim curl cmake build-essential > /dev/null 2>&1
echo "✓ System packages installed"

# 2. Virtual Environment
echo "[2/5] Setting up Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# 3. GPU support (PyTorch CUDA 12.1)
echo "[3/5] Installing PyTorch with CUDA support..."
# We explicitly install the CUDA version. 
# It's safe to run this even if installed; pip handles caching.
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121 >> setup.log 2>&1
echo "✓ PyTorch (CUDA) installed"

# 4. Core Dependencies
echo "[4/5] Installing Project Dependencies..."
./venv/bin/pip install -r requirements.txt >> setup.log 2>&1

echo "[4.5/5] Compiling llama-cpp-python..."
if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected. Building with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" ./venv/bin/pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir >> setup.log 2>&1
    echo "✓ llama-cpp-python (CUDA) installed"
else
    echo "  -> No NVIDIA GPU detected. Installing CPU-only version..."
    ./venv/bin/pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir >> setup.log 2>&1
    echo "✓ llama-cpp-python (CPU) installed"
fi

# 5. Check Resources
echo "[5/5] Checking Resources..."
# ZIM Check
ZIM_FILE=$(find . -maxdepth 1 -name "*.zim" | head -n 1)
if [ -n "$ZIM_FILE" ]; then
    echo "✓ Found ZIM file: $ZIM_FILE"
else
    echo "⚠️  No .zim file found. Please place your Wikipedia ZIM file here."
fi

# Model Directory Check
SHARED_MODELS="shared_models"
mkdir -p "$SHARED_MODELS"
echo "✓ Model directory verified: $SHARED_MODELS"
echo "  (Models will be automatically downloaded here on first run)"

# Enable 'hermit' command
echo ""
echo "Setting up 'hermit' command..."
HERMIT_WRAPPER="/usr/local/bin/hermit"
sudo tee "$HERMIT_WRAPPER" > /dev/null << HERMIT_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"

# Check if directory exists and give helpful error if not (common with external drives)
if [ ! -d "\$INSTALL_DIR" ]; then
    echo "❌ Error: Hermit installation directory not found at:"
    echo "   \$INSTALL_DIR"
    echo ""
    echo "👉 If this is on an external drive, please ensure it is MOUNTED."
    echo "   (Open your file manager and click on the drive to mount it)"
    echo ""
    echo "💡 If you moved the installation, re-run setup.sh from the new location."
    exit 1
fi

if [ -f "\$INSTALL_DIR/run_chatbot.sh" ]; then
    exec "\$INSTALL_DIR/run_chatbot.sh" "\$@"
else
    echo "❌ Error: run_chatbot.sh not found in \$INSTALL_DIR"
    echo "   The installation might be corrupted. Try re-running setup.sh"
    exit 1
fi
HERMIT_EOF
sudo chmod +x "$HERMIT_WRAPPER"
echo "✓ 'hermit' command installed"

# Enable 'forge' command (ZIM creator)
echo "Setting up 'forge' command..."
FORGE_WRAPPER="/usr/local/bin/forge"
sudo tee "$FORGE_WRAPPER" > /dev/null << FORGE_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"

if [ ! -d "\$INSTALL_DIR" ]; then
    echo "❌ Error: Hermit installation directory not found at:"
    echo "   \$INSTALL_DIR"
    exit 1
fi

if [ -f "\$INSTALL_DIR/forge.py" ]; then
    exec "\$INSTALL_DIR/venv/bin/python" "\$INSTALL_DIR/forge.py" "\$@"
else
    echo "❌ Error: forge.py not found in \$INSTALL_DIR"
    exit 1
fi
FORGE_EOF
sudo chmod +x "$FORGE_WRAPPER"
echo "✓ 'forge' command installed"

# Install optional document parsing dependencies for Forge
echo ""
echo "Installing optional Forge dependencies (PDF, DOCX support)..."
./venv/bin/pip install pypdf python-docx ebooklib markdown >> setup.log 2>&1 || true
echo "✓ Document parsers installed"

echo ""
echo "=== Setup Complete! ==="
echo "Run the chatbot with: hermit"
echo "Create ZIM files with: forge"
