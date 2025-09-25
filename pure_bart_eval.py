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
    # Prepare strategy token ids for robust first-token comparison
    strategy_tokens = list(STRATEGY_TOKENS.values())
    strategy_token_ids = set()
    for tok in strategy_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            strategy_token_ids.add(tid)

    def compute_metrics(eval_pred):
        # Create a fresh Metric object at each evaluation call to avoid state carryover
        metric = Metric(toker=tokenizer)
        preds, labels = eval_pred
        labels = np.array(labels)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(np.where(labels == -100, tokenizer.pad_token_id, labels), skip_special_tokens=True)

        allowed_strat_tokens = set(strategy_tokens)
        tp = {tok: 0 for tok in strategy_tokens}
        fp = {tok: 0 for tok in strategy_tokens}
        fn = {tok: 0 for tok in strategy_tokens}
        strat_support_total = 0

        for i, (ref, hyp) in enumerate(zip(label_texts, pred_texts)):
            metric.forword([ref], hyp)
            ref_ids = labels[i]
            ref_first_id = next((int(t) for t in ref_ids if t != -100 and t != tokenizer.pad_token_id), None)
            pred_seq = preds[i]
            pred_iter = (int(t) for t in pred_seq)
            pred_first_id = None
            for t in pred_iter:
                if tokenizer.bos_token_id is not None and t == tokenizer.bos_token_id:
                    continue
                if t == tokenizer.pad_token_id:
                    continue
                pred_first_id = t
                break

            if ref_first_id in strategy_token_ids and ref_first_id is not None and pred_first_id is not None:
                strat_support_total += 1
                ref_first_tok = tokenizer.convert_ids_to_tokens(ref_first_id)
                hyp_first_tok = tokenizer.convert_ids_to_tokens(pred_first_id)
                if hyp_first_tok == ref_first_tok:
                    tp[ref_first_tok] += 1
                else:
                    fn[ref_first_tok] += 1
                    if hyp_first_tok in allowed_strat_tokens:
                        fp[hyp_first_tok] += 1

        result, _ = metric.close()
        if strat_support_total > 0:
            total_tp = sum(tp.values())
            total_fp = sum(fp.values())
            total_fn = sum(fn.values())
            result["strat_acc"] = total_tp / strat_support_total if strat_support_total else 0.0
            micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            result["strat_f1_micro"] = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
            f1_list = []
            for tok in allowed_strat_tokens:
                supp = tp[tok] + fn[tok]
                if supp == 0:
                    continue
                p_c = tp[tok] / (tp[tok] + fp[tok]) if (tp[tok] + fp[tok]) > 0 else 0.0
                r_c = tp[tok] / (tp[tok] + fn[tok]) if (tp[tok] + fn[tok]) > 0 else 0.0
                f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
                f1_list.append(f1_c)
            if f1_list:
                result["strat_f1_macro"] = float(sum(f1_list) / len(f1_list))
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
    if args.dataset in ("strategy_esconv", "strategy_all_esconv"):
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


