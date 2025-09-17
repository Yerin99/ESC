"""
pure_bart.py
==============
Minimal BART training & evaluation on ESConv with generation metrics only.

Key points:
- No strategy/CLS or MTL. Pure seq2seq pretraining on (context -> response).
- Keep tiny_frac for quick debugging.
- Default generation uses top-p (0.9) sampling (no beam search).

Example usage
-------------
CUDA_VISIBLE_DEVICES=0 python pure_bart_train.py --output_dir outputs/pure_bart_base
CUDA_VISIBLE_DEVICES=0 python pure_bart_train.py --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny

CUDA_VISIBLE_DEVICES=0 python pure_bart_train.py --dataset problem_type_esconv --output_dir outputs/problem_type_bart
CUDA_VISIBLE_DEVICES=0 python pure_bart_train.py  --dataset problem_type_esconv --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny_problem_type

CUDA_VISIBLE_DEVICES=1 python pure_bart_train.py --dataset strategy_esconv --output_dir outputs/strategy_bart
CUDA_VISIBLE_DEVICES=1 python pure_bart_train.py  --dataset strategy_esconv --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny_strategy

CUDA_VISIBLE_DEVICES=2 python pure_bart_train.py --dataset situation_esconv --output_dir outputs/situation_bart
CUDA_VISIBLE_DEVICES=2 python pure_bart_train.py  --dataset situation_esconv --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny_situation

CUDA_VISIBLE_DEVICES=3 python pure_bart_train.py --dataset emotion_type_esconv --output_dir outputs/emotion_type_bart
CUDA_VISIBLE_DEVICES=3 python pure_bart_train.py  --dataset emotion_type_esconv --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny_emotion_type
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    The returned function expects `(preds, labels)` where `preds` are generated
    token ids and `labels` contain -100 for ignored positions. We replace -100
    with PAD before decoding for fair string comparison.
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


def apply_generation_sampling_config(model: BartForConditionalGeneration, args) -> None:
    """Apply top-p sampling config (no beam) to the model's generation_config."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/pure_bart_esconv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None, help="0~1 range for quick debugging")
    parser.add_argument("--max_src_length", type=int, default=1024)
    parser.add_argument("--max_tgt_length", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="pure_esconv", choices=dataset_choices())

    # Generation params (default: top-p sampling, no beam)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    # Logging
    log_dir = Path("logs/pure_bart")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"run_{Path(args.output_dir).name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")],
    )
    logger = logging.getLogger("pure_bart")
    set_seed(args.seed)

    # Tokenizer & model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    # Always add speaker tokens
    special_tokens = list(SPEAKER_TOKENS.values())
    # If dataset uses strategy tokens in context, add them as well
    if args.dataset == "strategy_esconv":
        special_tokens += list(STRATEGY_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.resize_token_embeddings(len(tokenizer))

    # Generation config: top-p sampling by default (no beam)
    apply_generation_sampling_config(model, args)

    # Datasets
    logger.info("Loading dataset: %s", args.dataset)
    DS = get_dataset(args.dataset)
    train_ds = DS(
        split="train",
        tokenizer=tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )
    val_ds = DS(
        split="validation",
        tokenizer=tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )
    test_ds = DS(
        split="test",
        tokenizer=tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # Training args - generation metrics only
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_tgt_length,
        generation_num_beams=1,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu-4",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer)
    )

    # Add callback to compute eval_ppl from eval_loss on every evaluation
    trainer.add_callback(PerplexityCallback())

    # Initial evaluation (optional)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Evaluating initial model (epoch 0)...")
    init_metrics = trainer.evaluate(eval_dataset=val_ds)
    (Path(args.output_dir) / "init_val_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in init_metrics.items() if v is not None}, indent=2)
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Re-apply generation config and resized embeddings after best model is loaded
    try:
        trainer.model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    apply_generation_sampling_config(trainer.model, args)

    # Validation
    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    (Path(args.output_dir) / "val_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in val_metrics.items() if v is not None}, indent=2)
    )

    # Test
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    (Path(args.output_dir) / "test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in test_metrics.items() if v is not None}, indent=2)
    )

    # Sample generations
    logger.info("Sampling few generations from validation set…")
    sample_n = 5
    indices = random.sample(range(len(val_ds)), min(sample_n, len(val_ds)))
    model.eval()
    for idx in indices:
        ex = val_ds[idx]
        input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
        attention_mask = torch.tensor([ex["attention_mask"]]).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        ctx_text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        logger.info("\n----- SAMPLE %d -----\nCTX: %s\nREF: %s\nGEN: %s\n", idx, ctx_text, ex["response"], gen_text)


if __name__ == "__main__":
    main()


