"""
pure_bart_eval.py
==================
Evaluation-only script for BART checkpoints on ESConv.

Features:
- Loads a specified checkpoint directory and evaluates with generation metrics.
- Supports evaluating on validation/test/train splits.
- Default generation uses top-p=0.9 sampling (no beam search).

Example usage:
CUDA_VISIBLE_DEVICES=0 python pure_bart_eval.py --checkpoint_dir outputs/pure_bart_base/checkpoint-XXXX --output_dir outputs/pure_bart_base

CUDA_VISIBLE_DEVICES=0 python pure_bart_eval.py --checkpoint_dir outputs/problem_type_bart/checkpoint-XXXX --output_dir outputs/problem_type_bart

CUDA_VISIBLE_DEVICES=1 python pure_bart_eval.py --checkpoint_dir outputs/strategy_bart/checkpoint-XXXX --output_dir outputs/strategy_bart

CUDA_VISIBLE_DEVICES=2 python pure_bart_eval.py --checkpoint_dir outputs/situation_bart/checkpoint-XXXX --output_dir outputs/situation_bart

CUDA_VISIBLE_DEVICES=3 python pure_bart_eval.py --checkpoint_dir outputs/emotion_type_bart/checkpoint-XXXX --output_dir outputs/emotion_type_bart
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from metric.myMetrics import Metric
from utils.dataset_registry import get_dataset, dataset_choices
from utils.tokens import SPEAKER_TOKENS, STRATEGY_TOKENS
from utils.ppl_callback import PerplexityCallback


def build_compute_metrics(tokenizer: BartTokenizer):
    """Return a function that computes generation metrics using `Metric`.

    The function replaces label -100 with PAD for decoding fairness.
    """
    metric = Metric(toker=tokenizer)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for ref, hyp in zip(label_texts, pred_texts):
            metric.forword([ref], hyp)
        result, _ = metric.close()
        return {k: float(v) for k, v in result.items()}

    return compute_metrics


def evaluate_split(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    split: str,
    output_dir: Path,
    batch_size: int,
    max_src_length: int,
    max_tgt_length: int,
    tiny_frac: Optional[float],
    dataset_name: str,
) -> Dict[str, float]:
    """Run evaluation on a specific ESConv split and save metrics JSON.

    Returns the metrics dict.
    """
    DS = get_dataset(dataset_name)
    ds = DS(
        split=split,
        tokenizer=tokenizer,
        max_src=max_src_length,
        max_tgt=max_tgt_length,
        tiny_frac=tiny_frac,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=max_tgt_length,
        generation_num_beams=1,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        eval_dataset=ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    # Add callback to compute eval_ppl from eval_loss
    trainer.add_callback(PerplexityCallback())

    metrics = trainer.evaluate(eval_dataset=ds)
    metrics = {k: float(v) for k, v in metrics.items() if v is not None}
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write evaluation JSONs")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "all"], help="Which split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None)
    parser.add_argument("--max_src_length", type=int, default=1024)
    parser.add_argument("--max_tgt_length", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="pure_esconv", choices=dataset_choices())

    # Generation params (sampling defaults)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    # Logging
    ckpt_name = Path(args.checkpoint_dir).name
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"eval_{ckpt_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs/pure_bart_eval")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"run_{ckpt_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="% (asctime)s | % (levelname)s | % (message)s".replace(" ", ""),
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")],
    )
    logger = logging.getLogger("pure_bart_eval")
    set_seed(args.seed)

    # Tokenizer & model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    special_tokens = list(SPEAKER_TOKENS.values())
    if args.dataset == "strategy_esconv":
        special_tokens += list(STRATEGY_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = BartForConditionalGeneration.from_pretrained(args.checkpoint_dir)
    model.resize_token_embeddings(len(tokenizer))

    # Generation config: top-p sampling
    gen_cfg = model.generation_config
    gen_cfg.max_length = args.max_tgt_length
    gen_cfg.repetition_penalty = args.repetition_penalty
    gen_cfg.num_beams = 1
    gen_cfg.do_sample = True
    gen_cfg.top_k = args.top_k
    gen_cfg.top_p = args.top_p
    gen_cfg.temperature = args.temperature
    gen_cfg.early_stopping = False
    model.generation_config = gen_cfg

    # Evaluate
    logger.info("Evaluating checkpoint: %s", args.checkpoint_dir)
    if args.split == "all":
        metrics_val = evaluate_split(
            model,
            tokenizer,
            split="validation",
            output_dir=out_dir,
            batch_size=args.batch_size,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            tiny_frac=args.tiny_frac,
            dataset_name=args.dataset,
        )
        logger.info("Validation metrics: %s", json.dumps(metrics_val, indent=2))

        metrics_test = evaluate_split(
            model,
            tokenizer,
            split="test",
            output_dir=out_dir,
            batch_size=args.batch_size,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            tiny_frac=args.tiny_frac,
            dataset_name=args.dataset,
        )
        logger.info("Test metrics: %s", json.dumps(metrics_test, indent=2))
    else:
        metrics = evaluate_split(
            model,
            tokenizer,
            split=args.split,
            output_dir=out_dir,
            batch_size=args.batch_size,
            max_src_length=args.max_src_length,
            max_tgt_length=args.max_tgt_length,
            tiny_frac=args.tiny_frac,
            dataset_name=args.dataset,
        )
        logger.info("%s metrics: %s", args.split.capitalize(), json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


