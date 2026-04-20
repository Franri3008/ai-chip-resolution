#!/usr/bin/env bash
# Setup script for ai-chip-resolution
# Usage: bash setup.sh [--dev] [--no-vllm]
#   --dev      also install dev/test dependencies
#   --no-vllm  skip vLLM installation (use if you only want cloud providers)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="${PYTHON:-python3}"
INSTALL_DEV=false
INSTALL_VLLM=true

for arg in "$@"; do
    case $arg in
        --dev)     INSTALL_DEV=true ;;
        --no-vllm) INSTALL_VLLM=false ;;
        *)         echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo " ai-chip-resolution setup"
echo "========================================"
echo

# ── 1. Python version check ───────────────────────────────────────────────────

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found. Install Python 3.12+ first."
    exit 1
fi

PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
    echo "ERROR: Python 3.12+ required (found $PY_MAJOR.$PY_MINOR)."
    exit 1
fi

echo "Python $PY_MAJOR.$PY_MINOR — OK"

# ── 2. Virtual environment ────────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at .venv..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated: $VENV_DIR"

# ── 3. Python dependencies ────────────────────────────────────────────────────

echo
echo "Installing dependencies..."
pip install --upgrade pip --quiet

pip install -r "$REPO_DIR/requirements.txt"

if $INSTALL_DEV; then
    echo
    echo "Installing dev dependencies..."
    pip install -r "$REPO_DIR/requirements-dev.txt"
fi

# ── 4. vLLM + Gemma 4 E2B ────────────────────────────────────────────────────

if $INSTALL_VLLM; then
    echo
    echo "Installing vLLM..."
    pip install vllm
    echo "  vLLM installed."

    echo
    echo "Downloading google/gemma-4-e2b-it from HuggingFace..."
    echo "(Requires your HF token to have accepted Gemma 4 terms at"
    echo " https://huggingface.co/google/gemma-4-e2b-it)"

    HF_TOKEN_FILE="$REPO_DIR/keys/.hf_token"
    if [ -f "$HF_TOKEN_FILE" ] && [ -s "$HF_TOKEN_FILE" ]; then
        HF_TOKEN=$(cat "$HF_TOKEN_FILE")
        huggingface-cli download google/gemma-4-e2b-it \
            --token "$HF_TOKEN" \
            --local-dir-use-symlinks False
        echo "  Model downloaded."
    else
        echo "  No HF token at keys/.hf_token yet — skipping model download."
        echo "  After setting your token, run:"
        echo "    huggingface-cli download google/gemma-4-e2b-it"
    fi
fi

# ── 5. API token files ────────────────────────────────────────────────────────

echo
echo "Setting up keys/..."
mkdir -p "$REPO_DIR/keys"

_write_token() {
    local label="$1" filename="$2" example="$3" required="${4:-optional}"
    local path="$REPO_DIR/keys/$filename"

    if [ -f "$path" ] && [ -s "$path" ]; then
        echo "  $label: already set"
        return
    fi

    local prompt="  $label ($example)"
    if [ "$required" = "required" ]; then
        prompt="$prompt [required]"
    else
        prompt="$prompt [optional, Enter to skip]"
    fi

    read -rp "$prompt: " tok </dev/tty
    if [ -n "$tok" ]; then
        echo "$tok" > "$path"
        chmod 600 "$path"
        echo "  Saved to $path"
    else
        echo "  Skipped — set later:  echo 'TOKEN' > $path"
    fi
}

_write_token "HuggingFace token" ".hf_token"    "hf_..."    "required"
_write_token "GitHub token"       ".gh_token"    "ghp_..."   "required"
_write_token "OpenAI key"         ".openai_token" "sk-..."   "optional"
_write_token "OpenRouter key"     ".openrouter_token" "sk-or-..." "optional"

# ── 6. Summary ────────────────────────────────────────────────────────────────

echo
echo "========================================"
echo " Setup complete"
echo "========================================"
echo
echo "Activate the venv:"
echo "  source .venv/bin/activate"
echo
echo "Run the pipeline (heuristic-only, no LLM):"
echo "  python main.py --top 50"
echo
echo "Run with LLM fallback:"
echo "  # OpenAI (default):"
echo "  python main.py --top 50 --llm"
echo
echo "  # Local vLLM (Gemma 4 E2B) — start the server first, then:"
echo "  vllm serve google/gemma-4-e2b-it --port 8000 --served-model-name gemma4-e2b"
echo "  LLM_LOCAL_MODEL=gemma4-e2b python main.py --top 50 --llm --provider LOCAL"
echo
echo "See README.md → VM Setup for a full cloud GPU walkthrough."
echo
if $INSTALL_VLLM; then
    echo "Start vLLM (in a separate terminal or tmux pane):"
    echo "  source .venv/bin/activate"
    echo "  vllm serve google/gemma-4-e2b-it --port 8000 --served-model-name gemma4-e2b \\"
    echo "       --gpu-memory-utilization 0.9 --max-num-seqs 16 --max-model-len 8192"
    echo
    echo "Then run the pipeline:"
    echo "  LLM_LOCAL_MODEL=gemma4-e2b python main.py --top 50 --llm --provider LOCAL"
fi
