# -*- coding: utf-8 -*-
"""
Round 2, Reviewer 1, Comment 3 — newline vs. `Step k:` segmentation comparison.

R1's round-2 concern: the manuscript (§3.2) segments reasoning paths into steps
based on newline delimiters. Newline splitting can cut a step in half or merge
steps, which directly affects the step-level contrastive training signal. R1
asks for a quick comparison with explicit `Step k:` markers to show the method
is robust to the choice of delimiter.

This script runs the D-ProtoCoT encoder training pipeline twice on
GSM8K / Qwen3-8B with identical settings (epochs, K, seed, paths, data split),
differing only in the step-segmentation scheme:

  * "newline"     — split paths at every '\n' (the scheme described in §3.2)
  * "step_marker" — split paths at `Step N:` markers (explicit boundaries)

For each mode it reports the test-time path-selection accuracy and the average
number of steps per path, so that the granularity difference is visible.

Usage (server, GPU), from the repo root:

  python newrun/reviewer1_q3_segmentation.py \
      --train_path newrundata/gsm8k_merged_flat.jsonl \
      --test_path  newrundata/gsm8k_test_flat.jsonl \
      --epochs 10 --seed 42 \
      --output   newrun/reviewer1_q3_segmentation.json
"""

import os
import sys
import json
import re
import argparse

import torch
import torch.nn.functional as F

# make the dprotocot package importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_DPROTO = os.path.join(_HERE, "..", "baseline", "dprotocot")
sys.path.insert(0, os.path.abspath(_DPROTO))

from config import Config                                     # noqa: E402
from data import load_splits, trainable_questions, \
    question_text, path_text                                  # noqa: E402
from train import train_encoder                               # noqa: E402
from prototype import build_prototype                         # noqa: E402


_STEP_MARKER_RE = re.compile(r"Step\s+\d+\s*[:.)]")


def build_cfg(args, mode: str) -> Config:
    cfg = Config()
    cfg.train_path = args.train_path
    cfg.test_path = args.test_path
    cfg.epochs = args.epochs
    cfg.seed = args.seed
    cfg.k_paths = args.k_paths
    cfg.step_segmentation = mode              # "newline" or "step_marker"
    cfg.output_dir = os.path.join("outputs", f"r1q3_seg_{mode}")
    if args.device:
        cfg.device = args.device
    return cfg.resolve()


@torch.no_grad()
def evaluate(cfg, test_groups, encoder):
    """Return (selection_accuracy_%, n_test_questions, avg_steps_per_path)."""
    encoder.eval()
    n_correct, n_q = 0, 0
    total_steps, total_paths = 0, 0

    for g in test_groups:
        texts = [path_text(p["cot"], g, cfg) for p in g["paths"]]
        if len(texts) < 2:
            continue
        n_q += 1
        z_q = encoder.encode_text_pooled(question_text(g, cfg))
        _, path_mat = encoder.encode_paths(texts)
        proto, _ = build_prototype(z_q, path_mat)
        proto = F.normalize(proto, dim=-1)
        zp = F.normalize(path_mat, dim=-1)
        align = (zp @ proto).tolist()
        sel = int(max(range(len(align)), key=lambda i: align[i]))
        n_correct += int(g["paths"][sel]["is_correct"])

        # measure granularity: re-segment each path under the same cfg
        for p in g["paths"]:
            cot = path_text(p["cot"], g, cfg)
            if cfg.step_segmentation == "step_marker":
                n_step = len(_STEP_MARKER_RE.findall(cot))
            else:
                n_step = len([s for s in cot.split(cfg.step_delimiter) if s.strip()])
            total_steps += max(1, n_step)
            total_paths += 1

    acc = 100.0 * n_correct / max(1, n_q)
    avg_steps = total_steps / max(1, total_paths)
    return acc, n_q, avg_steps


def run_mode(args, mode: str) -> dict:
    print("\n" + "=" * 68)
    print(f"[mode] step_segmentation = {mode}")
    print("=" * 68)
    cfg = build_cfg(args, mode)
    train_g, val_g, test_g = load_splits(cfg)
    train_t = trainable_questions(train_g)
    val_t = trainable_questions(val_g)

    print(f"[run] training encoder (mode={mode}, epochs={cfg.epochs}) ...")
    enc = train_encoder(cfg, train_t, val_t)

    acc, n_test, avg_steps = evaluate(cfg, test_g, enc)
    print(f"[run] {mode:12s}  selected-path accuracy = {acc:.2f}%  "
          f"(n_test={n_test}, avg_steps/path={avg_steps:.2f})")
    return {
        "mode": mode,
        "test_questions": n_test,
        "selected_accuracy_pct": acc,
        "avg_steps_per_path": avg_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="newrundata/gsm8k_merged_flat.jsonl")
    ap.add_argument("--test_path",  default="newrundata/gsm8k_test_flat.jsonl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--k_paths", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="newrun/reviewer1_q3_segmentation.json")
    args = ap.parse_args()

    results = [run_mode(args, m) for m in ("newline", "step_marker")]

    print("\n" + "=" * 68)
    print("COMPARISON — Round 2 R1-Q3 (newline vs. Step k: segmentation)")
    print("-" * 68)
    for r in results:
        print(f"  {r['mode']:12s}  acc={r['selected_accuracy_pct']:.2f}%  "
              f"avg_steps={r['avg_steps_per_path']:.2f}  "
              f"(n_test={r['test_questions']})")
    delta = results[1]["selected_accuracy_pct"] - results[0]["selected_accuracy_pct"]
    print(f"  delta (step_marker - newline) = {delta:+.2f}")
    print("=" * 68)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "train_path": args.train_path,
            "test_path": args.test_path,
            "epochs": args.epochs,
            "seed": args.seed,
            "results": results,
            "delta_step_marker_minus_newline": delta,
        }, f, ensure_ascii=False, indent=2)
    print(f"[saved] -> {args.output}")


if __name__ == "__main__":
    main()
