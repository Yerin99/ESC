# -*- coding: utf-8 -*-
"""
blenderbot.py
==============
Pure BlenderBot-small (90M) training & evaluation script for ESConv **without** strategy tokens.

사용법 예시
+------------
# ➊ 기본 greedy 학습 + 검증 BLEU 등 메트릭 계산
CUDA_VISIBLE_DEVICES=0 python blenderbot.py --output_dir outputs/pure_blenderbot_90M

# ➋ tiny_frac (0.05)로 빠른 디버깅
#   Beam Search (빔 2개 사용)
CUDA_VISIBLE_DEVICES=1 python blenderbot.py --tiny_frac 0.05 --epochs 1 --num_beams 2 --output_dir outputs/tiny_bb90M_beam_2
#   Top-k Sampling (k=50)
CUDA_VISIBLE_DEVICES=2 python blenderbot.py --tiny_frac 0.05 --epochs 1 --top_k 50 --output_dir outputs/tiny_bb90M_sample_k_50
#   Top-p Sampling (p=0.9)
CUDA_VISIBLE_DEVICES=3 python blenderbot.py --tiny_frac 0.05 --epochs 1 --top_p 0.9 --output_dir outputs/tiny_bb90M_sample_p_0.9

# ➌ 다양한 생성 파라미터 실험
#   Beam Search (빔 4개 사용)
CUDA_VISIBLE_DEVICES=1 python blenderbot.py --num_beams 4 --output_dir outputs/bb90M_beam_4
#   Top-k Sampling (k=50)
CUDA_VISIBLE_DEVICES=2 python blenderbot.py --top_k 50 --output_dir outputs/bb90M_sample_k_50
#   Top-p Sampling (p=0.9)
CUDA_VISIBLE_DEVICES=3 python blenderbot.py --top_p 0.9 --output_dir outputs/bb90M_sample_p_0.9

# ➍  (scripts/run_blenderbot.sh 로 자동 실행)
bash scripts/run_blenderbot.sh
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)

from metric.myMetrics import Metric
from utils.tokens import SPEAKER_TOKENS

# ======================= Dataset =======================
class ESConvDataset(torch.utils.data.Dataset):
    """Dataset that builds (context, response) pairs without strategy tokens."""

    def __init__(
        self,
        split: str,
        tokenizer: BlenderbotSmallTokenizer,
        max_src: int = 512,
        max_tgt: int = 128,
        tiny_frac: Optional[float] = None,
        dataset_name: str = "thu-coai/esconv",
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'  # Truncate from the beginning of the context
        self.max_src = max_src
        self.max_tgt = max_tgt

        raw = load_dataset(dataset_name, split=split)
        if tiny_frac:
            raw = raw.shuffle(seed=42).select(range(int(len(raw) * tiny_frac)))

        usr_tok, sys_tok = SPEAKER_TOKENS["usr"], SPEAKER_TOKENS["sys"]
        separator = " "  # Use a space to join turns

        self.examples: List[Dict[str, Any]] = []
        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]
            for turn_idx, turn in enumerate(dialog):
                if turn["speaker"] != "sys":
                    continue

                # ------ build context ------
                ctx_parts: List[str] = []
                for prev in dialog[:turn_idx]:
                    spk_tok = usr_tok if prev["speaker"] == "usr" else sys_tok
                    ctx_parts.append(f"{spk_tok}{prev['text']}")
                context = separator.join(ctx_parts)

                if not context.strip():
                    continue

                enc = tokenizer(
                    context,
                    max_length=max_src,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,  # Let tokenizer add BOS/EOS
                )
                dec = tokenizer(
                    turn["text"],
                    max_length=max_tgt,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )

                # Mask PAD (and optionally BOS) for fairer PPL and stable training
                labels = list(dec.input_ids)
                # mask all PAD positions
                labels = [(-100 if tok == tokenizer.pad_token_id else tok) for tok in labels]
                # optional: also mask BOS
                if labels and labels[0] == tokenizer.bos_token_id:
                    labels[0] = -100  # keep EOS to train proper stopping

                self.examples.append(
                    {
                        "input_ids": enc.input_ids,
                        "attention_mask": enc.attention_mask,
                        "labels": labels,
                        "context": context,
                        "response": turn["text"],
                    }
                )

    def __len__(self) -> int:  # noqa: D401
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        return self.examples[idx]


# ======================= Metric helpers =======================

def build_compute_metrics(tokenizer: BlenderbotSmallTokenizer):
    metric_obj = Metric(toker=tokenizer)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for ref, hyp in zip(label_texts, pred_texts):
            metric_obj.forword([ref], hyp)
        result, _ = metric_obj.close()
        return {k: float(v) for k, v in result.items()}

    return compute_metrics


# ======================= Custom Trainer for PPL =======================
class CustomTrainer(Seq2SeqTrainer):
    """
    A Seq2SeqTrainer that computes and logs perplexity, ensuring it's available for early stopping.
    """
    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in metrics:
            metrics[f"{metric_key_prefix}_perplexity"] = math.exp(metrics[loss_key])
        return metrics

def int_or_none(value: str) -> Optional[int]:
    """Custom argparse type for integer or 'None' string."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value or 'None': {value}")

# ======================= Main =======================

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/blenderbot_esconv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None)
    parser.add_argument("--max_src_length", type=int, default=512)
    parser.add_argument("--max_tgt_length", type=int, default=128)

    # generation params
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--early_stopping_patience",
        type=int_or_none,
        default=None,
        help="Enable early stopping with specified patience. Pass 'None' to disable.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("blenderbot")
    set_seed(args.seed)

    # ---------------- tokenizer & model ----------------
    pretrained_name = "facebook/blenderbot-90M"
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_name)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPEAKER_TOKENS.values())})

    model = BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_name)
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
        gen_cfg.length_penalty = 1.0  # Turn off for greedy/sampling
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
            gen_cfg.temperature = 1.0
            logger.info("Using greedy search.")

    model.generation_config = gen_cfg

    # ---------------- datasets ----------------
    logger.info("Loading datasets…")
    train_ds = ESConvDataset(
        "train",
        tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )
    val_ds = ESConvDataset(
        "validation",
        tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )
    test_ds = ESConvDataset(
        "test",
        tokenizer,
        max_src=args.max_src_length,
        max_tgt=args.max_tgt_length,
        tiny_frac=args.tiny_frac,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # ---------------- trainer ----------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_tgt_length,
        generation_num_beams=args.num_beams,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience else [],
    )

    # ---------------- train & eval ----------------
    logger.info("✅ Starting training…")
    trainer.train()

    logger.info("📊 Evaluating on validation set…")
    # The original `trainer.evaluate()` uses `predict_with_generate=True` from TrainingArguments,
    # which does not yield the correct cross-entropy loss for PPL.
    # We must evaluate in two steps, just like for the test set.

    # 1. PPL calculation (teacher forcing)
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    trainer.args.predict_with_generate = False
    try:
        ppl_metrics_val = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True

    # 2. Generation metrics (autoregressive)
    if trainer.model.generation_config.num_beams == 1:
        trainer.model.generation_config.early_stopping = False
        trainer.model.generation_config.length_penalty = 1.0
    gen_metrics_val = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")

    # 3. Combine and log validation metrics
    eval_metrics = {
        "eval_loss": ppl_metrics_val.get("eval_loss"),
        "eval_perplexity": ppl_metrics_val.get("eval_perplexity"),
        **{k: v for k, v in gen_metrics_val.items() if k not in ["eval_loss", "eval_perplexity"]},
    }
    logger.info("Validation metrics: %s", json.dumps(eval_metrics, indent=2))

    # word-level PPL removed (non-standard and slow)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_eval = {k: float(v) for k, v in eval_metrics.items() if k not in ["eval_loss", "eval_perplexity"]}
    (out_dir / "val_metrics.json").write_text(json.dumps(_save_eval, indent=2))

    logger.info("🧪 Evaluating on test set…")
    # 1. PPL calculation (teacher forcing)
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    trainer.args.predict_with_generate = False
    try:
        ppl_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True

    # 2. Generation metrics (autoregressive)
    if trainer.model.generation_config.num_beams == 1:
        trainer.model.generation_config.early_stopping = False
        trainer.model.generation_config.length_penalty = 1.0
    gen_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    # 3. Combine metrics and save (exclude BPE-level loss/ppl)
    final_test_metrics = {**{k: v for k, v in gen_metrics.items() if k not in ["test_loss", "test_perplexity"]}}
    logger.info("Test metrics: %s", json.dumps(final_test_metrics, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps({k: float(v) for k, v in final_test_metrics.items()}, indent=2))


    # ---------------- sample generations ----------------
    sample_n = 5
    indices = random.sample(range(len(val_ds)), min(sample_n, len(val_ds)))
    model.eval()

    # Ensure generation config is correct before generating samples
    if model.generation_config.num_beams == 1:
        model.generation_config.early_stopping = False
        model.generation_config.length_penalty = 1.0

    for idx in indices:
        ex = val_ds[idx]
        input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
        attn_mask = torch.tensor([ex["attention_mask"]]).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attn_mask)
        gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ctx_text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        logger.info(
            "\n----- SAMPLE %d -----\nCTX: %s\nREF: %s\nGEN: %s\n",
            idx,
            ctx_text,
            ex["response"],
            gen_text,
        )


if __name__ == "__main__":
    main()
