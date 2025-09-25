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

CUDA_VISIBLE_DEVICES=1 python pure_bart_train.py --dataset strategy_all_esconv --output_dir outputs/strategy_all_bart
CUDA_VISIBLE_DEVICES=1 python pure_bart_train.py  --dataset strategy_all_esconv --tiny_frac 0.01 --epochs 1 --output_dir outputs/tiny_strategy_all

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
from utils.metrics_logger_callback import EvalMetricsLoggerCallback


def build_compute_metrics(tokenizer: BartTokenizer):
    """Return a function that computes generation metrics using `Metric`.

    The returned function expects `(preds, labels)` where `preds` are generated
    token ids and `labels` contain -100 for ignored positions. We replace -100
    with PAD before decoding for fair string comparison.
    """
    # Prepare strategy token ids for robust first-token comparison (no decoding ambiguity)
    from utils.tokens import STRATEGY_TOKENS as _STRAT_TOKENS
    strategy_tokens = list(_STRAT_TOKENS.values())
    strategy_token_ids = set()
    for tok in strategy_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            strategy_token_ids.add(tid)

    def compute_metrics(eval_pred):
        # Instantiate a fresh Metric object per evaluation to avoid cross-evaluation accumulation
        metric = Metric(toker=tokenizer)
        preds, labels = eval_pred
        labels = np.array(labels)
        # For generation metrics we keep skip_special_tokens=True (as before)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(np.where(labels == -100, tokenizer.pad_token_id, labels), skip_special_tokens=True)

        # For strategy metrics (strategy_all_esconv), we will compare by token IDs (not text),
        # so skip_special_tokens setting does not affect the measurement.

        # Strategy-first metrics (strategy_all_esconv): compare first tokens
        allowed_strat_tokens = set(strategy_tokens)
        tp: Dict[str, int] = {tok: 0 for tok in strategy_tokens}
        fp: Dict[str, int] = {tok: 0 for tok in strategy_tokens}
        fn: Dict[str, int] = {tok: 0 for tok in strategy_tokens}
        strat_support_total = 0
        for i, (ref, hyp) in enumerate(zip(label_texts, pred_texts)):
            metric.forword([ref], hyp)
            # Use ids to extract first strategy token robustly
            ref_ids = labels[i]
            # first non -100 and not PAD as ground-truth first token
            ref_first_id = next((int(t) for t in ref_ids if t != -100 and t != tokenizer.pad_token_id), None)
            pred_seq = preds[i]
            # skip decoder_start_token_id once, then skip BOS/PAD
            # For BART, decoder_start_token_id == eos_token_id in practice; use tokenizer value.
            start_id = tokenizer.eos_token_id
            first_seen = True
            pred_iter = (int(t) for t in pred_seq)
            pred_first_id = None
            for t in pred_iter:
                if first_seen and start_id is not None and t == start_id:
                    first_seen = False
                    continue
                first_seen = False
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
        # Attach strategy metrics if applicable
        if strat_support_total > 0:
            total_tp = sum(tp.values())
            total_fp = sum(fp.values())
            total_fn = sum(fn.values())
            # Accuracy equals micro recall when one label per sample
            result["strat_acc"] = total_tp / strat_support_total if strat_support_total else 0.0
            micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            result["strat_f1_micro"] = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
            # Macro-F1 over classes with support
            f1_list: List[float] = []
            f1_weighted_sum = 0.0
            total_support = 0
            for tok in allowed_strat_tokens:
                supp = tp[tok] + fn[tok]
                if supp == 0:
                    continue
                p_c = tp[tok] / (tp[tok] + fp[tok]) if (tp[tok] + fp[tok]) > 0 else 0.0
                r_c = tp[tok] / (tp[tok] + fn[tok]) if (tp[tok] + fn[tok]) > 0 else 0.0
                f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
                f1_list.append(f1_c)
                f1_weighted_sum += f1_c * supp
                total_support += supp
            if f1_list:
                result["strat_f1_macro"] = float(sum(f1_list) / len(f1_list))
            if total_support > 0:
                result["strat_f1_weighted"] = float(f1_weighted_sum / total_support)
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


def decode_with_visible_specials(tokenizer: BartTokenizer, ids) -> str:
    """Decode while keeping special tokens visible except PAD.

    - Keeps speaker/strategy/BOS/EOS tokens so we can inspect sequences.
    - Removes PAD tokens to avoid clutter.
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    pad_tok = tokenizer.pad_token or "<pad>"
    text = text.replace(pad_tok, "")
    return " ".join(text.split())


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

    # Optimization params
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="fraction of total steps for warmup (0~1)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="number of warmup steps (overrides ratio if >0)")
    parser.add_argument("--weight_decay", type=float, default=0.0)

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
    if args.dataset in ("strategy_esconv", "strategy_all_esconv"):
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
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_tgt_length,
        generation_num_beams=1,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
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
    # Add callback to persist eval_* metrics each evaluation
    trainer.add_callback(EvalMetricsLoggerCallback())

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
    # Use a distinct prefix so the eval-history callback can ignore this entry
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="final_val")
    (Path(args.output_dir) / "val_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in val_metrics.items() if v is not None}, indent=2)
    )

    # Test
    logger.info("Evaluating on test set...")
    # Use a distinct prefix so the eval-history callback can ignore this entry
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="final_test")
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
        gen_text = decode_with_visible_specials(tokenizer, gen_ids[0])
        ctx_text = decode_with_visible_specials(tokenizer, ex["input_ids"])
        logger.info("\n----- SAMPLE %d -----\nCTX: %s\nREF: %s\nGEN: %s\n", idx, ctx_text, ex["response"], gen_text)


if __name__ == "__main__":
    main()


