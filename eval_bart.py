# -*- coding: utf-8 -*-
"""
Evaluation-only script for BART on ESConv.
- Loads a trained model (or pretrained if not provided)
- Computes teacher-forcing PPL and generation metrics on the test split
- Also computes word-level perplexity via streaming for fair comparison with baselines

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval_bart.py \
    --model_dir outputs/sample_p_0.9/checkpoint-798 \
    --tokenizer_dir facebook/bart-base \
    --top_k 50 \
    --output_dir outputs/eval_ckpt798_k50

    CUDA_VISIBLE_DEVICES=3 python eval_bart.py \
    --model_dir outputs/sample_p_0.9/checkpoint-798 \
    --tokenizer_dir facebook/bart-base \
    --top_p 0.9 \
    --output_dir outputs/eval_ckpt798_p0.9
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import math
from typing import Optional

import torch
from transformers import (
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

from metric.myMetrics import Metric
from utils.tokens import SPEAKER_TOKENS, STRATEGY_NAMES
from utils.stats import compute_word_perplexity_streaming
from utils.strategy import (
    DataCollatorWithStrategy,
    compute_teacher_strategy_report,
    compute_teacher_strategy_scores,
)

# Reuse helpers from bart.py to avoid duplication
from bart import ESConvDataset, build_compute_metrics, ESCTrainer  # type: ignore
from models.bart_mtl_strategy import BartForESCWithStrategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Path to fine-tuned model directory")
    parser.add_argument("--pretrained", type=str, default="facebook/bart-base", help="HF model id if model_dir is None")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Optional tokenizer source dir/model id")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save metrics; defaults to model_dir")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None)
    # Reserve +1 position for strategy token at inference
    parser.add_argument("--max_src_length", type=int, default=1023)
    parser.add_argument("--max_tgt_length", type=int, default=256)

    # generation params
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    # logging to both console and logs/bart
    log_dir = Path("logs/bart")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"eval_{Path(args.output_dir or args.model_dir or 'eval').__str__().split('/')[-1]}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")],
    )
    logger = logging.getLogger("eval_bart")
    set_seed(args.seed)

    # tokenizer (robust loading)
    tok_src = args.tokenizer_dir or args.model_dir or args.pretrained
    try:
        tokenizer = BartTokenizer.from_pretrained(tok_src)
    except Exception:
        logger.warning(f"Could not load tokenizer from '{tok_src}'. Falling back to '{args.pretrained}'.")
        tokenizer = BartTokenizer.from_pretrained(args.pretrained)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPEAKER_TOKENS.values())})

    # model (strategy-aware)
    model = BartForESCWithStrategy.from_pretrained(args.model_dir or args.pretrained,
                                                   num_strategies=len(STRATEGY_NAMES))
    model.resize_token_embeddings(len(tokenizer))

    # generation config (match bart.py behavior exactly)
    gen_cfg = model.generation_config
    gen_cfg.max_length = args.max_tgt_length
    gen_cfg.repetition_penalty = args.repetition_penalty

    if args.num_beams > 1:
        gen_cfg.num_beams = args.num_beams
        gen_cfg.do_sample = False
        # Unset sampling-specific parameters to avoid validation warnings when do_sample=False
        gen_cfg.top_k = None
        gen_cfg.top_p = None
        gen_cfg.temperature = 1.0
        gen_cfg.early_stopping = True
        gen_cfg.no_repeat_ngram_size = 3
        logger.info(f"Using beam search with {args.num_beams} beams.")
    else:
        gen_cfg.num_beams = 1
        gen_cfg.early_stopping = False  # ensure unset for non-beam
        if args.top_k > 0 or args.top_p < 1.0:
            gen_cfg.do_sample = True
            gen_cfg.top_k = args.top_k
            gen_cfg.top_p = args.top_p
            gen_cfg.temperature = args.temperature
            logger.info(
                f"Using sampling with top_k={args.top_k}, top_p={args.top_p}, temperature={args.temperature}."
            )
        else:
            gen_cfg.do_sample = False
            logger.info("Using greedy search.")
    model.generation_config = gen_cfg

    # dataset
    test_ds = ESConvDataset(
        "test", tokenizer, max_src=args.max_src_length, max_tgt=args.max_tgt_length, tiny_frac=args.tiny_frac
    )
    data_collator = DataCollatorWithStrategy(tokenizer, model=model, padding="longest")

    # trainer for evaluation only
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir or args.model_dir or "outputs/eval_bart",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_length,
        generation_num_beams=args.num_beams,
        report_to="none",
        seed=args.seed,
    )
    trainer: Seq2SeqTrainer = ESCTrainer(
        model=model,
        args=targs,
        train_dataset=None,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    logger.info("Evaluating on test set (teacher-forcing loss for PPL)…")
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    trainer.args.predict_with_generate = False
    try:
        # make sure early_stopping is false in non-beam modes
        if trainer.model.generation_config.num_beams == 1:
            trainer.model.generation_config.early_stopping = False
        ppl_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True

    logger.info("Evaluating on test set (generation metrics)…")
    if trainer.model.generation_config.num_beams == 1:
        trainer.model.generation_config.early_stopping = False
        trainer.model.generation_config.length_penalty = 1.0
    gen_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    # combine and add word-level ppl
    final_test_metrics = {
        "test_loss": ppl_metrics.get("test_loss"),
        "test_perplexity": ppl_metrics.get("test_perplexity", (math.exp(ppl_metrics.get("test_loss")) if ppl_metrics.get("test_loss") is not None else None)),
        **{k: v for k, v in gen_metrics.items()},
    }
    try:
        test_w_ppl = compute_word_perplexity_streaming(trainer, test_ds, tokenizer, exclude_token_ids=[tokenizer.pad_token_id])
        final_test_metrics["test_word_perplexity"] = float(test_w_ppl)
    except Exception:
        pass

    out_dir = Path(args.output_dir or args.model_dir or "outputs/eval_bart")
    out_dir.mkdir(parents=True, exist_ok=True)
    # teacher strategy scores
    try:
        scores = compute_teacher_strategy_scores(trainer, test_ds, STRATEGY_NAMES)
        final_test_metrics["test_strategy_acc"] = scores["acc"]
        final_test_metrics["test_strategy_f1_weighted"] = scores["f1_weighted"]
    except Exception:
        pass

    (out_dir / "test_metrics.json").write_text(json.dumps({k: float(v) for k, v in final_test_metrics.items() if v is not None}, indent=2))

    # classification report (teacher strategy)
    try:
        rep_str = compute_teacher_strategy_report(trainer, test_ds, STRATEGY_NAMES)
        (out_dir / "test_strategy_report.txt").write_text(rep_str)
        logger.info("\nTeacher Strategy Classification Report (test):\n%s", rep_str)
    except Exception as e:
        logger.warning("strategy report failed: %s", e)
    logger.info("Test metrics: %s", json.dumps(final_test_metrics, indent=2))

    # sample generations
    model.eval()
    import random

    indices = random.sample(range(len(test_ds)), min(5, len(test_ds)))
    for idx in indices:
        ex = test_ds[idx]
        input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
        attn_mask = torch.tensor([ex["attention_mask"]]).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attn_mask)
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ctx_text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        logger.info("\n----- SAMPLE %d -----\nCTX: %s\nREF: %s\nGEN: %s\n", idx, ctx_text, ex["response"], gen_text)


if __name__ == "__main__":
    main()
