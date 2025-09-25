#!/usr/bin/env python
"""
Evaluate one or many checkpoints using pure_bart_eval.py and write an aggregated JSONL.

Usage examples:

1) Evaluate only step 7980 for three roots and write history per root
   CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoints.py \
       --roots outputs/pure_bart_base_lr1e5 outputs/pure_bart_base_lr2e5 outputs/pure_bart_base_lr3e5 \
       --steps 7980 --split all --dataset pure_esconv --history_name eval_history_eval.jsonl

2) Evaluate all checkpoints under a single root
   CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoints.py \
       --roots outputs/pure_bart_base_lr3e5 --split all --dataset pure_esconv

Notes:
- This script DOES NOT modify pure_bart_eval.py. It shells out to it per checkpoint.
- Per-checkpoint metrics JSON files are written into the checkpoint directory by
  passing --output_dir <checkpoint_dir> to pure_bart_eval.py to avoid overwrites.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _extract_global_step(checkpoint_dir: Path) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
    return int(match.group(1)) if match else None


def _list_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    pairs: List[Tuple[int, Path]] = []
    for p in candidates:
        step = _extract_global_step(p)
        if step is not None:
            pairs.append((step, p))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def _run_eval_for_checkpoint(
    checkpoint_dir: Path,
    dataset: str,
    split: str,
    batch_size: int,
    seed: int,
    tiny_frac: Optional[float],
    max_src_length: int,
    max_tgt_length: int,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
) -> None:
    """Invoke pure_bart_eval.py for a single checkpoint, writing metrics into the checkpoint dir."""
    import subprocess, sys

    script_path = Path(__file__).resolve().parents[1] / "pure_bart_eval.py"
    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(checkpoint_dir),
        "--dataset",
        dataset,
        "--split",
        split,
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--max_src_length",
        str(max_src_length),
        "--max_tgt_length",
        str(max_tgt_length),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--temperature",
        str(temperature),
        "--repetition_penalty",
        str(repetition_penalty),
    ]

    env = os.environ.copy()
    # Honor externally provided CUDA_VISIBLE_DEVICES; do not override here.
    subprocess.run(cmd, check=True, env=env)


def _read_metrics(checkpoint_dir: Path, split: str) -> Dict[str, float]:
    if split == "all":
        val_path = checkpoint_dir / "validation_metrics.json"
        test_path = checkpoint_dir / "test_metrics.json"
        result: Dict[str, float] = {}
        if val_path.exists():
            val = json.loads(val_path.read_text())
            result.update({f"val_{k}": float(v) for k, v in val.items()})
        if test_path.exists():
            test = json.loads(test_path.read_text())
            result.update({f"test_{k}": float(v) for k, v in test.items()})
        return result
    else:
        p = checkpoint_dir / f"{split}_metrics.json"
        return {k: float(v) for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def _parse_steps_arg(steps: Optional[str]) -> Optional[Sequence[int]]:
    if not steps:
        return None
    return [int(s.strip()) for s in steps.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=str, nargs="+", required=True, help="One or more checkpoint roots")
    parser.add_argument("--history_name", type=str, default="eval_history_eval.jsonl", help="Aggregated JSONL filename per root")
    parser.add_argument("--dataset", type=str, default="pure_esconv")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "all"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None)
    parser.add_argument("--max_src_length", type=int, default=1024)
    parser.add_argument("--max_tgt_length", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated step numbers to evaluate (e.g., '7980,7182'). If omitted, evaluate all.")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip checkpoints where target metrics JSON already exists.")

    args = parser.parse_args()

    selected_steps = _parse_steps_arg(args.steps)

    for root_str in args.roots:
        root = Path(root_str)
        checkpoints = _list_checkpoints(root)
        if selected_steps is not None:
            checkpoints = [p for p in checkpoints if _extract_global_step(p) in selected_steps]
        if not checkpoints:
            continue

        history_path = root / args.history_name

        for ckpt in checkpoints:
            if args.skip_if_exists:
                # Determine if metrics already exist for this split selection
                if args.split == "all":
                    if (ckpt / "validation_metrics.json").exists() and (ckpt / "test_metrics.json").exists():
                        continue
                else:
                    if (ckpt / f"{args.split}_metrics.json").exists():
                        continue

            _run_eval_for_checkpoint(
                checkpoint_dir=ckpt,
                dataset=args.dataset,
                split=args.split,
                batch_size=args.batch_size,
                seed=args.seed,
                tiny_frac=args.tiny_frac,
                max_src_length=args.max_src_length,
                max_tgt_length=args.max_tgt_length,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )

            # After evaluation, read produced metrics and append to JSONL
            metrics = _read_metrics(ckpt, args.split)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "global_step": _extract_global_step(ckpt),
                "checkpoint": ckpt.name,
                **{k: float(v) for k, v in metrics.items()},
            }
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()


