"""Fail-fast check that this box can actually train, not just import torch.

The check that matters on a 5090 is the kernel launch. `torch.cuda.is_available()`
returns True even when the wheel has no sm_120 kernels, so anything that only
prints version strings will pass on a broken install. Every GPU check below
forces a real kernel and a real synchronize.

Exit code 0 = ready to train. Non-zero = do not start a run.
"""

import sys

FAIL = []
WARN = []


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    WARN.append(msg)
    print(f"  WARN  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


print("[1] torch import")
try:
    import torch
except ImportError as e:
    print(f"  FAIL  cannot import torch: {e}")
    sys.exit(1)
ok(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")

print("[2] device visible")
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False -- driver/wheel mismatch")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
ok(f"{name}, sm_{cap[0]}{cap[1]}, {total:.1f} GiB")

print("[3] wheel contains kernels for this arch")
arches = torch.cuda.get_arch_list()
want = f"sm_{cap[0]}{cap[1]}"
if want in arches:
    ok(f"{want} present in {arches}")
else:
    # This is the classic 5090 trap: import works, first matmul dies.
    fail(f"{want} NOT in wheel arch list {arches} -- reinstall from a cu128 index")

print("[4] real kernel launch (fp32 matmul)")
try:
    a = torch.randn(512, 512, device="cuda")
    (a @ a).sum().item()  # .item() forces a sync, surfacing async kernel errors
    ok("fp32 matmul executed")
except Exception as e:
    fail(f"fp32 matmul failed: {type(e).__name__}: {e}")

print("[5] bf16 compute + autograd")
try:
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (x @ w).float().pow(2).mean().backward()
    torch.cuda.synchronize()
    if w.grad is None:
        fail("backward produced no gradient")
    else:
        ok("bf16 forward + backward executed")
except Exception as e:
    fail(f"bf16 autograd failed: {type(e).__name__}: {e}")

print("[6] fused SDPA (the attention path we train with)")
try:
    q, k, v = (torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    torch.nn.functional.scaled_dot_product_attention(q, k, v).float().sum().item()
    ok("scaled_dot_product_attention executed")
except Exception as e:
    fail(f"SDPA failed: {type(e).__name__}: {e}")

print("[7] training stack versions")
for mod in ("transformers", "peft", "accelerate", "datasets", "sklearn", "scipy"):
    try:
        m = __import__(mod)
        ok(f"{mod} {getattr(m, '__version__', '?')}")
    except ImportError:
        fail(f"{mod} not installed")

print("[8] bitsandbytes (only needed for quantized runs)")
try:
    import bitsandbytes  # noqa: F401
    ok(f"bitsandbytes {bitsandbytes.__version__}")
except Exception as e:
    warn(f"bitsandbytes unusable ({type(e).__name__}) -- fine unless you load in 4/8-bit")

print()
if FAIL:
    print(f"NOT READY: {len(FAIL)} check(s) failed. Do not start a training run.")
    for m in FAIL:
        print(f"  - {m}")
    sys.exit(1)
print(f"READY{f' ({len(WARN)} warning(s))' if WARN else ''}: environment can train.")
