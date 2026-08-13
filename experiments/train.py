"""Unified continual-learning trainer for the temporal vulnerability-detection study.

One entry point for all nine strategies, replacing the ten divergent notebooks.
Consolidating them is not cosmetic: the notebooks disagree on whether the LoRA
adapter carries across windows, which changes what a method *is*, and that
difference was not visible when each lived in its own file.

Reproduces the conference setup by default: phi-2, LoRA r=16 alpha=32
dropout=0.05, 3 epochs, lr 2e-4, batch 32, FP32, seed 42, bi-monthly windows,
forward evaluation on t+1 and backward evaluation at lags 1/3/5/6.

Two things differ from the notebooks, both deliberate:

1. Metrics are logged per class plus a true macro average. The notebooks logged
   `f1_score(y_true, y_pred)` -- sklearn's binary F1 of label 1, which is FIXED
   -- and the paper printed it as Macro-F1. The old number is still logged as
   `f1_binary_pos1_LEGACY` so runs can be compared across the fix.
2. --adapter overrides a method's default continuity, so "does OLoRA fail
   because orthogonality is rigid, or because it restarts every window?" can be
   answered instead of assumed.

Examples:
    # smoke test on CPU, tiny model, two windows -- run this before booking GPU time
    python experiments/train.py --method hybrid-casr --data-dir DATA --out runs/ --smoke

    # reproduce the published configuration
    python experiments/train.py --method hybrid-casr --data-dir DATA --out runs/

    # three seeds
    for s in 42 43 44; do
      python experiments/train.py --method hybrid-casr --data-dir DATA --out runs/ --seed $s
    done

    # olora as published vs olora actually continual
    python experiments/train.py --method olora --data-dir DATA --out runs/
    python experiments/train.py --method olora --data-dir DATA --out runs/ --adapter inherit
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BACKWARD_LAGS, LABEL_MAP, METHODS, MetricsWriter,  # noqa: E402
                    compute_metrics, make_windows, resolve_replay)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="microsoft/phi-2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--granularity", type=int, default=2,
                   help="window width in months (1/2/3/6/12)")
    p.add_argument("--adapter", choices=["inherit", "fresh", "none"],
                   help="override the method's adapter policy")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                   help="fp32 matches the published runs; bf16 is faster but not "
                        "numerically identical, so do not mix them within a comparison")
    p.add_argument("--no-backward", action="store_true", help="skip backward retention eval")
    p.add_argument("--keep-adapters", action="store_true",
                   help="keep every window's adapter (needed to resume; uses disk)")
    p.add_argument("--limit-windows", type=int, help="stop after N windows (debugging)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny model, 2 windows, 1 epoch, CPU-friendly: validates the "
                        "pipeline end to end without touching a real model")
    return p.parse_args()


def load_window(path, tokenizer, max_length):
    """Load one JSONL window. Filter matches the notebooks exactly."""
    import pandas as pd
    from datasets import Dataset

    df = pd.read_json(path, lines=True)
    df = df[df["prompt"].astype(str).str.len() > 10]
    df = df[df["response"].isin(LABEL_MAP)]
    if df.empty:
        return None
    df["label"] = df["response"].map(LABEL_MAP)
    ds = Dataset.from_pandas(df[["prompt", "label"]], preserve_index=False)
    return ds.map(
        lambda x: tokenizer(x["prompt"], truncation=True, padding="max_length",
                            max_length=max_length),
        batched=True, remove_columns=["prompt"])


def entropy_scores(model, dataset, batch_size=32):
    """Predictive entropy per sample, in dataset order."""
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(dataset.with_format("torch"), batch_size=batch_size)
    out = []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            probs = F.softmax(model(**inputs).logits.float(), dim=-1)
            out.extend((-(probs * torch.log(probs + 1e-10)).sum(-1)).cpu().tolist())
    return out


def select_uncertain(dataset, model, k, balanced):
    """Top-k by entropy; with `balanced`, k//2 from each class.

    Faithful to the committed Hybrid-CASR: entropy ranks within each class, and
    random draws only fill a class that cannot supply its half. Note this is not
    the 70/30 uncertain/uniform mix the paper describes.
    """
    from datasets import concatenate_datasets

    if k <= 0 or len(dataset) == 0:
        return None
    scored = dataset.add_column("entropy", entropy_scores(model, dataset))
    if not balanced:
        return scored.sort("entropy", reverse=True) \
                     .select(range(min(k, len(scored)))).remove_columns(["entropy"])

    half = max(1, k // 2)
    parts = []
    pools = {}
    for cls in (0, 1):
        pool = scored.filter(lambda ex, c=cls: ex["label"] == c).sort("entropy", reverse=True)
        take = pool.select(range(min(half, len(pool))))
        parts.append(take)
        pools[cls] = pool.select(range(len(take), len(pool)))
    # Cross-fill from the other class only if one class ran out.
    for cls in (0, 1):
        short = half - len(parts[cls])
        if short > 0 and len(pools[1 - cls]):
            other = pools[1 - cls].shuffle(seed=123 + cls)
            parts.append(other.select(range(min(short, len(other)))))
    return concatenate_datasets(parts).remove_columns(["entropy"])


def orthogonalise_lora_a(model, history=None):
    """Orthogonalise LoRA A matrices.

    With `history`, Gram-Schmidt against previously used directions (OLoRA).
    Without, QR-orthogonalise each matrix in place (LB-CL initialisation).
    """
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" not in name or param.ndim != 2:
                continue
            w = param.data
            if history is not None and name in history and history[name].numel():
                basis = history[name].to(w.device, w.dtype)   # (rank_seen, in_features)
                w = w - (w @ basis.T) @ basis
                norms = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
                param.data = w / norms * param.data.norm(dim=1, keepdim=True)
            else:
                q, _ = torch.linalg.qr(w.T.float())
                param.data = q.T[: w.shape[0]].to(w.dtype)


def collect_lora_directions(model, history):
    """Accumulate row-orthonormal bases of the LoRA A matrices seen so far."""
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" not in name or param.ndim != 2:
                continue
            rows = torch.nn.functional.normalize(param.data.float(), dim=1)
            prev = history.get(name)
            stacked = rows if prev is None else torch.cat([prev.to(rows.device), rows], 0)
            # Re-orthonormalise so the basis does not degenerate as it grows.
            q, _ = torch.linalg.qr(stacked.T)
            history[name] = q.T.cpu()


def main():
    args = parse_args()
    spec = dict(METHODS[args.method])
    adapter_policy = args.adapter or spec["adapter"]

    import numpy as np
    import torch
    from datasets import concatenate_datasets
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    if args.smoke:
        args.model = os.environ.get("SMOKE_MODEL", "hf-internal-testing/tiny-random-gpt2")
        args.epochs, args.batch_size, args.max_length = 1, 4, 64
        args.limit_windows = args.limit_windows or 3
        print(f"[smoke] model={args.model} epochs=1 windows={args.limit_windows}")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    windows = make_windows(args.granularity)
    available = [w for w in windows
                 if os.path.exists(os.path.join(args.data_dir, f"{w}.jsonl"))]
    if len(available) < 2:
        sys.exit(f"need at least 2 window files in {args.data_dir}, found {len(available)}")
    if len(available) < len(windows):
        print(f"note: {len(windows) - len(available)} of {len(windows)} windows have no file; "
              f"using the {len(available)} present")
    windows = available
    if args.limit_windows:
        windows = windows[: args.limit_windows]

    run = f"{args.method}_g{args.granularity}m_seed{args.seed}"
    if args.adapter:
        run += f"_adapter-{args.adapter}"
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    metrics = MetricsWriter(os.path.join(args.out, "metrics.csv"))
    res_path = os.path.join(args.out, "resources.csv")
    new_res = not os.path.exists(res_path)
    res_fh = open(res_path, "a", newline="", encoding="utf-8")
    res_w = csv.writer(res_fh)
    if new_res:
        res_w.writerow(["method", "seed", "granularity", "train_window", "n_train",
                        "fit_time_sec", "gpu_mem_peak_mb", "params_trainable"])

    json.dump({**vars(args), "adapter_policy": adapter_policy,
               "method_desc": spec["desc"],
               "paper_mismatch": spec.get("paper_mismatch"),
               "windows": windows},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    print(f"\n{'=' * 72}\n{run}\n  {spec['desc']}\n  adapter={adapter_policy} "
          f"replay={spec['replay']}\n  {len(windows)} windows, seed {args.seed}")
    if spec.get("paper_mismatch"):
        print(f"  NOTE: {spec['paper_mismatch']}")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout, bias="none",
                             task_type=TaskType.SEQ_CLS)

    cache = {}

    def window_ds(tag):
        if tag not in cache:
            cache[tag] = load_window(os.path.join(args.data_dir, f"{tag}.jsonl"),
                                     tokenizer, args.max_length)
        return cache[tag]

    def fresh_base():
        m = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, torch_dtype=dtype)
        m.config.pad_token_id = tokenizer.pad_token_id
        return m

    lora_history = {}
    n_windows = len(windows) - 1

    for i in range(n_windows):
        train_tag, eval_tag = windows[i], windows[i + 1]
        print(f"\n[{i + 1}/{n_windows}] {args.method}: train {train_tag} -> eval {eval_tag}")

        train_ds, eval_ds = window_ds(train_tag), window_ds(eval_tag)
        if train_ds is None or eval_ds is None:
            print("  skipped: a window is empty after filtering")
            continue

        base = fresh_base()
        prev_dir = os.path.join(out_dir, windows[i - 1]) if i > 0 else None
        if adapter_policy == "none":
            model = base
        elif adapter_policy == "inherit" and i > 0 and prev_dir and os.path.isdir(prev_dir):
            model = PeftModel.from_pretrained(base, prev_dir, is_trainable=True,
                                              torch_dtype=dtype)
        else:
            model = get_peft_model(base, peft_config)
            if spec.get("orthogonal_init"):
                orthogonalise_lora_a(model)
            if spec.get("orthogonalise_against_history") and lora_history:
                orthogonalise_lora_a(model, lora_history)

        if torch.cuda.is_available():
            model = model.cuda()

        n_replay = 0
        if adapter_policy != "none":
            tags, mode, budget = resolve_replay(spec["replay"], i, windows)
            extra = []
            for j, tag in enumerate(tags):
                past = window_ds(tag)
                if past is None:
                    continue
                if mode in ("full", "cumulative"):
                    extra.append(past)
                else:
                    k = budget[j] if isinstance(budget, list) else budget
                    picked = select_uncertain(past, model, k,
                                              balanced=(mode == "uncertain-balanced"))
                    if picked is not None and len(picked):
                        extra.append(picked)
            if extra:
                n_replay = sum(len(e) for e in extra)
                train_ds = concatenate_datasets([train_ds] + extra)
                print(f"  replay: +{n_replay} samples from {len(extra)} window(s)")

        if adapter_policy != "none":
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        save_dir = os.path.join(out_dir, train_tag)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(save_dir, "_hf"),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs, learning_rate=args.lr,
                save_strategy="no", eval_strategy="no", logging_strategy="no",
                report_to="none", fp16=False, bf16=(args.dtype == "bf16"),
                seed=args.seed, disable_tqdm=True),
            train_dataset=train_ds, eval_dataset=eval_ds)

        if adapter_policy != "none":
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            trainer.train()
            fit_time = round(time.time() - t0, 3)
            peak_mb = int(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else -1
            res_w.writerow([args.method, args.seed, args.granularity, train_tag,
                            len(train_ds), fit_time, peak_mb,
                            sum(p.numel() for p in model.parameters() if p.requires_grad)])
            res_fh.flush()
            print(f"  trained on {len(train_ds)} samples in {fit_time}s, peak {peak_mb} MB")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            if spec.get("orthogonalise_against_history"):
                collect_lora_directions(model, lora_history)

        def evaluate(ds, eval_window, direction):
            pred = trainer.predict(ds)
            m = compute_metrics(pred.label_ids, pred.predictions.argmax(-1))
            metrics.write(method=args.method, seed=args.seed, granularity=args.granularity,
                          train_window=train_tag, eval_window=eval_window,
                          direction=direction, n_train=len(train_ds), n_replay=n_replay, **m)
            return m

        fwd = evaluate(eval_ds, eval_tag, "forward")
        print(f"  forward  macro_f1={fwd['macro_f1']:.4f}  "
              f"vuln={fwd['f1_vulnerable']:.4f} fixed={fwd['f1_fixed']:.4f}  "
              f"(legacy f1={fwd['f1_binary_pos1_LEGACY']:.4f})")

        if not args.no_backward:
            for lag in BACKWARD_LAGS:
                if i - lag < 0:
                    continue
                back = window_ds(windows[i - lag])
                if back is not None:
                    evaluate(back, windows[i - lag], f"backward_{lag}p")

        # Only the previous window's adapter is needed to continue the chain.
        if not args.keep_adapters and i > 0 and prev_dir and os.path.isdir(prev_dir):
            import shutil
            shutil.rmtree(prev_dir, ignore_errors=True)

        del model, base, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics.close(); res_fh.close()
    print(f"\ndone -> {os.path.join(args.out, 'metrics.csv')}")


if __name__ == "__main__":
    main()
