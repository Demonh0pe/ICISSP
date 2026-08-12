#!/usr/bin/env bash
# AutoDL RTX 5090 (Blackwell, sm_120) environment setup.
#
# Why cu128: the 5090 is compute capability 12.0. PyTorch wheels built against
# CUDA 12.4 or earlier contain no sm_120 kernels, so they import fine, report
# cuda.is_available() == True, and then die on the first real kernel launch with
#   "CUDA error: no kernel image is available for execution on the device"
# That failure looks like a model/code bug but is purely a wheel-arch mismatch.
#
# Usage:
#   bash env/setup.sh            # install, then self-check
#   SKIP_INSTALL=1 bash env/setup.sh   # self-check only
#
# Overrides:
#   TORCH_INDEX_URL  torch wheel index (default: official cu128)
#   PIP_INDEX_URL    index for everything else (set to a mirror if pypi is slow)

set -euo pipefail

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "!! nvidia-smi not found -- this is not a GPU machine, aborting." >&2
    exit 1
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo
    echo "=== removing any preinstalled torch (AutoDL images ship cu121/cu124) ==="
    # Not fatal if absent.
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

    echo
    echo "=== installing torch from ${TORCH_INDEX_URL} ==="
    pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

    echo
    echo "=== installing training stack ==="
    # No version pins on the HF stack: Blackwell support landed recently enough
    # that pinning to older releases reintroduces the sm_120 problem.
    pip install -U \
        "transformers>=4.51" \
        "peft>=0.14" \
        "accelerate>=1.4" \
        "datasets>=3.0" \
        "bitsandbytes>=0.45" \
        sentencepiece protobuf safetensors \
        scikit-learn pandas numpy scipy \
        matplotlib tqdm

    # Deliberately NOT installed: flash-attn. Prebuilt wheels lag Blackwell and
    # source builds take ~1h on a 4-core box. Use attn_implementation="sdpa";
    # PyTorch's fused SDPA covers this workload with no accuracy difference.
fi

echo
echo "=== self-check ==="
python "${HERE}/verify_env.py"
