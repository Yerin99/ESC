# -*- coding: utf-8 -*-
"""
Evaluation-only script for BlenderBot Small (90M) on ESConv.
- Loads a trained model (or pretrained) and evaluates on the test split
- Reports teacher-forcing PPL, generation metrics, and word-level PPL

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_blenderbot.py --model_dir outputs/bb90M_beam5
  CUDA_VISIBLE_DEVICES=0 python eval_blenderbot.py --pretrained facebook/blenderbot-90M --output_dir outputs/bb90M_eval
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

from metric.myMetrics import Metric
from utils.tokens import SPEAKER_TOKENS

from blenderbot import ESConvDataset, build_compute_metrics, CustomTrainer  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default="facebook/blenderbot-90M")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None)
    parser.add_argument("--max_src_length", type=int, default=512)
    parser.add_argument("--max_tgt_length", type=int, default=128)

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("eval_blenderbot")
    set_seed(args.seed)

    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_dir or args.pretrained)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPEAKER_TOKENS.values())})

    model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_dir or args.pretrained)
    model.resize_token_embeddings(len(tokenizer))

    gen_cfg = model.generation_config
    gen_cfg.max_length = args.max_tgt_length
    gen_cfg.repetition_penalty = args.repetition_penalty
    if args.num_beams > 1:
        gen_cfg.num_beams = args.num_beams
        gen_cfg.do_sample = False
        gen_cfg.top_k = None
        gen_cfg.top_p = None
        gen_cfg.temperature = 1.0
        gen_cfg.early_stopping = True
        gen_cfg.no_repeat_ngram_size = 3
        logger.info(f"Using beam search with {args.num_beams} beams.")
    else:
        gen_cfg.num_beams = 1
        gen_cfg.early_stopping = False
        gen_cfg.length_penalty = 1.0
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

    test_ds = ESConvDataset("test", tokenizer, max_src=args.max_src_length, max_tgt=args.max_tgt_length, tiny_frac=args.tiny_frac)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # Safety: ensure labels' PAD are masked (dataset should already do it)
    try:
        for ex in getattr(test_ds, "examples", [])[:1]:
            if tokenizer.pad_token_id in ex.get("labels", []):
                ex["labels"] = [(-100 if t == tokenizer.pad_token_id else t) for t in ex["labels"]]
    except Exception:
        pass

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir or args.model_dir or "outputs/eval_blenderbot",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_length,
        generation_num_beams=args.num_beams,
        report_to="none",
        seed=args.seed,
    )

    trainer: Seq2SeqTrainer = CustomTrainer(
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
        ppl_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True

    logger.info("Evaluating on test set (generation metrics)…")
    if trainer.model.generation_config.num_beams == 1:
        trainer.model.generation_config.early_stopping = False
        trainer.model.generation_config.length_penalty = 1.0
    gen_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    final_test_metrics = {
        **{k: v for k, v in gen_metrics.items() if k not in ["test_loss", "test_perplexity"]},
    }

    out_dir = Path(args.output_dir or args.model_dir or "outputs/eval_blenderbot")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps({k: float(v) for k, v in final_test_metrics.items()}, indent=2))
    logger.info("Test metrics: %s", json.dumps(final_test_metrics, indent=2))

    # sample generations
    import random
    model.eval()
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
