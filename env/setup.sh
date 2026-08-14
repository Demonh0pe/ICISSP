#!/usr/bin/env bash
# AutoDL RTX 5090 (Blackwell, sm_120) environment setup.
#
# Why cu128: the 5090 is compute capability 12.0. PyTorch wheels built against
# CUDA 12.4 or earlier contain no sm_120 kernels, so they import fine, report
# cuda.is_available() == True, and then die on the first real kernel launch with
#   "CUDA error: no kernel image is available for execution on the device"
# That failure looks like a model/code bug but is purely a wheel-arch mismatch.
# AutoDL images ship torch 2.1.2+cu121, which is exactly this trap.
#
# Two things this script works around, both hit from AutoDL:
#
#   * Old pip normalises distribution names inconsistently and rejects valid
#     wheels with "inconsistent Name: expected 'typing-extensions', but metadata
#     has 'typing_extensions'", then falls back to building the sdist. So pip is
#     upgraded before anything else.
#   * files.pythonhosted.org is slow from Chinese hosts (single-digit kB/s) while
#     pypi.nvidia.com and download.pytorch.org are fine. Pure-Python
#     dependencies are therefore pre-seeded from a mirror so the torch install
#     never has to reach pythonhosted.
#
# Usage:
#   bash env/setup.sh                  # install, then self-check
#   SKIP_INSTALL=1 bash env/setup.sh   # self-check only
#
# Overrides:
#   TORCH_INDEX_URL  torch wheel index      (default: official cu128)
#   PIP_MIRROR       index for everything else (default: TUNA; set to "" for pypi)
#
# On AutoDL, `source /etc/network_turbo` before running this speeds up the
# non-mirrored hosts considerably.

set -euo pipefail

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_MIRROR="${PIP_MIRROR-https://pypi.tuna.tsinghua.edu.cn/simple}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Slow links need patience, not a shorter timeout.
PIP_NET=(--timeout 120 --retries 10)
MIRROR_ARGS=()
if [ -n "${PIP_MIRROR}" ]; then
    MIRROR_ARGS=(-i "${PIP_MIRROR}")
fi

echo "=== driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "!! nvidia-smi not found -- this is not a GPU machine, aborting." >&2
    exit 1
fi

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo
    echo "=== upgrading pip (old pip rejects valid wheels on name normalisation) ==="
    python -m pip install "${PIP_NET[@]}" "${MIRROR_ARGS[@]}" -U pip setuptools wheel

    echo
    echo "=== removing any preinstalled torch (AutoDL images ship cu121/cu124) ==="
    # Not fatal if absent.
    python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

    echo
    echo "=== pre-seeding torch's pure-Python deps from ${PIP_MIRROR:-pypi} ==="
    # Fetching these from a fast mirror first means the torch install below finds
    # them already satisfied and never touches files.pythonhosted.org.
    python -m pip install "${PIP_NET[@]}" "${MIRROR_ARGS[@]}" -U \
        typing_extensions sympy networkx jinja2 filelock fsspec mpmath

    echo
    echo "=== installing torch from ${TORCH_INDEX_URL} ==="
    # The nvidia-* CUDA runtime wheels come from pypi.nvidia.com, which is fast;
    # only the generic deps were the problem, and those are already installed.
    python -m pip install "${PIP_NET[@]}" --index-url "${TORCH_INDEX_URL}" \
        torch torchvision torchaudio

    echo
    echo "=== installing training stack ==="
    # No version pins on the HF stack: Blackwell support landed recently enough
    # that pinning to older releases reintroduces the sm_120 problem.
    python -m pip install "${PIP_NET[@]}" "${MIRROR_ARGS[@]}" -U \
        "transformers>=4.51" \
        "peft>=0.14" \
        "accelerate>=1.4" \
        "datasets>=3.0" \
        "bitsandbytes>=0.45" \
        sentencepiece protobuf safetensors \
        scikit-learn pandas numpy scipy \
        matplotlib tqdm

    # Deliberately NOT installed: flash-attn. Prebuilt wheels lag Blackwell and
    # source builds take about an hour on this box. Use attn_implementation="sdpa";
    # PyTorch's fused SDPA covers this workload with no accuracy difference.
fi

echo
echo "=== self-check ==="
python "${HERE}/verify_env.py"
