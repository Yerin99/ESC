# -*- coding: utf-8 -*-
"""
bart.py
==========
BART training & evaluation script for ESConv (Emotional Support Conversation)

사용법 예시
+------------
# ➊ 기본 greedy 학습 + 검증 BLEU 등 메트릭 계산
CUDA_VISIBLE_DEVICES=0 python bart.py --output_dir outputs/pure_bart_base

# ➋ tiny_frac (0.01)로 빠른 디버깅
#   Beam Search (빔 2개 사용)
CUDA_VISIBLE_DEVICES=1 python bart.py --tiny_frac 0.05 --epochs 1 --num_beams 2 --output_dir outputs/tiny_beam_2
#   Top-k Sampling (k=50)
CUDA_VISIBLE_DEVICES=2 python bart.py --tiny_frac 0.01 --epochs 1 --top_k 50 --output_dir outputs/tiny_sample_k_50
#   Top-p Sampling (p=0.9)
CUDA_VISIBLE_DEVICES=3 python bart.py --tiny_frac 0.05 --epochs 1 --top_p 0.9 --output_dir outputs/tiny_sample_p_0.9

# ➌ 다양한 생성 파라미터 실험
#   Beam Search (빔 4개 사용)
CUDA_VISIBLE_DEVICES=1 python bart.py --num_beams 4 --output_dir outputs/beam_4
#   Top-k Sampling (k=50)
CUDA_VISIBLE_DEVICES=2 python bart.py --top_k 50 --output_dir outputs/sample_k_50
#   Top-p Sampling (p=0.9)
CUDA_VISIBLE_DEVICES=3 python bart.py --top_p 0.9 --output_dir outputs/sample_p_0.9

# ➍ 다양한 생성 파라미터 실험 (scripts/run_experiments.sh 로 자동 실행)
bash scripts/run_experiments.sh

핵심 기능
1. context: `[USR]…` / `[SYS]…` 로 구분
2. myMetrics 를 활용해 BLEU, Distinct, F1, ROUGE-L 산출.
3. `--tiny_frac` 로 데이터 일부만 사용해 빠른 테스트 가능.
4. 결과는 JSON/로그로 출력.
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)
# Shared utility

from metric.myMetrics import Metric
from utils.tokens import SPEAKER_TOKENS, STRATEGY_NAMES
from utils.strategy import (
    DataCollatorWithStrategy,
    compute_strategy_report,
    compute_strategy_scores,
)
from models.bart_mtl_strategy import BartForESCWithStrategyDualAttn

# ======================= Dataset =======================
class ESConvDataset(torch.utils.data.Dataset):
    """Dataset that builds (context, response) pairs without strategy tokens.

    Also returns `strategy_labels` for each system turn if `turn["strategy"]` is
    present in the raw ESConv JSON. Missing labels are set to -100 so the
    classification loss can ignore them.
    """

    def __init__(
        self,
        split: str,
        tokenizer: BartTokenizer,
        max_src: int = 1024,
        max_tgt: int = 256,
        tiny_frac: Optional[float] = None,
        dataset_name: str = "thu-coai/esconv",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

        raw = load_dataset(dataset_name, split=split)
        if tiny_frac:
            raw = raw.shuffle(seed=42).select(range(int(len(raw) * tiny_frac)))

        usr_tok, sys_tok = SPEAKER_TOKENS["usr"], SPEAKER_TOKENS["sys"]
        eos = f" {tokenizer.eos_token}"

        self.examples: List[Dict[str, Any]] = []
        strat2id = {name: i for i, name in enumerate(STRATEGY_NAMES)}
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
                context = tokenizer.bos_token + (eos.join(ctx_parts) if ctx_parts else "") + tokenizer.eos_token

                # skip empty context (should not happen)
                if not context.strip():
                    continue

                # tokenise
                enc = tokenizer(
                    context,
                    max_length=max_src,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False,
                )
                dec = tokenizer(
                    turn["text"],
                    max_length=max_tgt,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )

                # Masking for fair CE/PPL: ignore PAD and BOS in labels; keep EOS
                labels = list(dec.input_ids)
                labels = [(-100 if tok == tokenizer.pad_token_id else tok) for tok in labels]
                if labels and dec.input_ids[0] == tokenizer.bos_token_id:
                    labels[0] = -100

                # Strategy label always exists for sys turns in ESConv
                strat_name = turn.get("strategy")
                assert strat_name in strat2id, f"Unknown strategy name: {strat_name}"
                strat_label = strat2id[strat_name]

                self.examples.append(
                    {
                        "input_ids": enc.input_ids,
                        "attention_mask": enc.attention_mask,
                        "labels": labels,
                        # For debugging / sample printing
                        "context": context,
                        "response": turn["text"],
                        "strategy_labels": strat_label,
                    }
                )

    def __len__(self) -> int:  # noqa: D401
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        return self.examples[idx]

# ======================= Metric helpers =======================

def build_compute_metrics(tokenizer: BartTokenizer):
    metric_obj = Metric(toker=tokenizer)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # replace label -100 with pad for decoding
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for ref, hyp in zip(label_texts, pred_texts):
            metric_obj.forword([ref], hyp)

        result, _ = metric_obj.close()
        # convert numpy/np.float to python float
        return {k: float(v) for k, v in result.items()}

    return compute_metrics


def int_or_none(value: str) -> Optional[int]:
    """Custom argparse type for integer or 'None' string."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value or 'None': {value}")


# ======================= Trainer with PPL + MTL =======================
class ESCTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that:
    - Adds eval perplexity metric
    - Combines LM loss with strategy CE (label smoothing) during training
    - Updates model.alpha via epoch-based linear schedule (alpha_start→alpha_end over warmup epochs)
    """

    def __init__(self, *args,
                 alpha_start: float = 0.5,
                 alpha_end: float = 1.0,
                 alpha_warmup_epochs: float = 2.0,
                 cls_weight: float = 0.3,
                 detach_steps: int = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.alpha_warmup_epochs = float(alpha_warmup_epochs)
        self.cls_weight = float(cls_weight)
        self.detach_steps = int(detach_steps)

    def _update_model_knobs(self):
        # epoch may be None initially
        epoch = float(self.state.epoch or 0.0)
        if self.alpha_warmup_epochs <= 0:
            ratio = 1.0
        else:
            ratio = min(max(epoch / self.alpha_warmup_epochs, 0.0), 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * ratio
        model = self.model
        # push knobs for current step
        if hasattr(model, "alpha"):
            model.alpha = float(alpha)
        if hasattr(model, "cls_weight"):
            model.cls_weight = float(self.cls_weight)
        if hasattr(model, "detach_steps"):
            model.detach_steps = int(self.detach_steps)
        if hasattr(model, "global_step"):
            model.global_step = int(self.state.global_step)

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        # Ensure no early_stopping in non-beam/sampling modes to avoid warnings
        try:
            if getattr(self.model.generation_config, "num_beams", 1) == 1:
                self.model.generation_config.early_stopping = False
        except Exception:
            pass
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in metrics:
            metrics[f"{metric_key_prefix}_perplexity"] = math.exp(metrics[loss_key])
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        # Keep strategy labels separate
        strategy_labels = inputs.pop("strategy_labels", None)
        self._update_model_knobs()
        outputs = model(**inputs, strategy_labels=strategy_labels)

        lm_loss = outputs.loss
        total_loss = lm_loss
        cls_loss = getattr(outputs, "cls_loss", None)
        if self.model.training and cls_loss is not None:
            total_loss = lm_loss + (self.cls_weight * cls_loss)

        return (total_loss, outputs) if return_outputs else total_loss


# ======================= Main =======================

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/pure_bart_esconv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None, help="0~1 range for quick debugging")
    # Use full source length; no extra token concatenation
    parser.add_argument("--max_src_length", type=int, default=1024)
    parser.add_argument("--max_tgt_length", type=int, default=256)

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

    # ---- Strategy-aware MTL knobs ----
    parser.add_argument("--num_strategies", type=int, default=8)
    parser.add_argument("--cls_weight", type=float, default=0.3)
    parser.add_argument("--alpha_start", type=float, default=0.5)
    parser.add_argument("--alpha_end", type=float, default=1.0)
    parser.add_argument("--alpha_warmup_epochs", type=float, default=2.0)
    parser.add_argument("--detach_steps", type=int, default=0)

    args = parser.parse_args()

    # logging to both console and file under logs/bart/
    log_dir = Path("logs/bart")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"run_{Path(args.output_dir).name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")],
    )
    logger = logging.getLogger("pure_bart")
    set_seed(args.seed)

    # -------------------- tokenizer & model --------------------
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPEAKER_TOKENS.values())})

    model = BartForESCWithStrategyDualAttn.from_pretrained(
        "facebook/bart-base",
        num_strategies=args.num_strategies or len(STRATEGY_NAMES),
    )
    model.resize_token_embeddings(len(tokenizer))

    # generation config:
    # - if num_beams > 1, it's beam search
    # - if num_beams == 1, it's greedy (or sampling if top_k/top_p is set)
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
            logger.info(f"Using sampling with top_k={args.top_k}, top_p={args.top_p}, temperature={args.temperature}.")
        else:
            gen_cfg.do_sample = False
            logger.info("Using greedy search.")

    model.generation_config = gen_cfg
    # 안전성을 위해 dual-attn 초기 버전에서는 캐시를 비활성화 (속도 영향만, 정확도 동일)
    model.generation_config.use_cache = False
    model.config.use_cache = False

    # -------------------- datasets --------------------
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

    data_collator = DataCollatorWithStrategy(tokenizer, model=model, padding="longest")

    # -------------------- trainer --------------------
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
        generation_num_beams=args.num_beams,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
    )

    trainer = ESCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience else [],
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_warmup_epochs=args.alpha_warmup_epochs,
        cls_weight=args.cls_weight,
        detach_steps=args.detach_steps,
    )

    # Optional: allow switching to pure ground-truth strategy token mixing via env/flag
    if int(os.environ.get("ESC_USE_GT_STRATEGY", "0")) == 1:
        try:
            model.use_ground_truth_strategy = True
            logger.info("Using ground-truth strategy for mixture token (pure one-hot).")
        except Exception:
            pass

    # -------------------- optional init eval (epoch 0) --------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📊 Evaluating initial model (epoch 0)…")
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    trainer.args.predict_with_generate = False
    try:
        init_ppl = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="init_eval")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True
    init_gen = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="init_eval")
    # combine like normal eval: include loss/perplexity and generation metrics
    init_metrics = {
        "init_eval_loss": init_ppl.get("init_eval_loss"),
        "init_eval_perplexity": init_ppl.get("init_eval_perplexity", (math.exp(init_ppl.get("init_eval_loss")) if init_ppl.get("init_eval_loss") is not None else None)),
        **{k: v for k, v in init_gen.items()},
    }
    # word-level PPL removed (non-standard and slow)
    # strategy scores
    try:
        strat_scores_init = compute_strategy_scores(trainer, val_ds, STRATEGY_NAMES)
        init_metrics["init_eval_strategy_acc"] = strat_scores_init["acc"]
        init_metrics["init_eval_strategy_f1_weighted"] = strat_scores_init["f1_weighted"]
    except Exception:
        pass
    (out_dir / "init_val_metrics.json").write_text(json.dumps({k: float(v) for k, v in init_metrics.items() if v is not None}, indent=2))
    try:
        rep_init = compute_strategy_report(trainer, val_ds, STRATEGY_NAMES)
        (out_dir / "init_val_strategy_report.txt").write_text(rep_init)
        logger.info("\nStrategy Classification Report (init validation):\n%s", rep_init)
    except Exception:
        pass

    # -------------------- train & eval --------------------
    logger.info("✅ Starting training…")
    trainer.train()

    logger.info("📊 Evaluating on validation set…")
    # The original evaluate uses predict_with_generate=True, which prevents CE-based ppl.
    # Evaluate in two steps (teacher forcing -> generation) as before.

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

    # 3. Combine and log validation metrics (preserve loss+ppl and add gen metrics)
    eval_metrics = {
        "eval_loss": ppl_metrics_val.get("eval_loss"),
        "eval_perplexity": ppl_metrics_val.get("eval_perplexity", (math.exp(ppl_metrics_val.get("eval_loss")) if ppl_metrics_val.get("eval_loss") is not None else None)),
        **{k: v for k, v in gen_metrics_val.items()},
    }
    logger.info("Validation metrics: %s", json.dumps(eval_metrics, indent=2))

    # Strategy classification report on validation set
    try:
        rep_str = compute_strategy_report(trainer, val_ds, STRATEGY_NAMES)
        logger.info("\nStrategy Classification Report (validation):\n%s", rep_str)
    except Exception as e:
        logger.warning("strategy report (val) failed: %s", e)

    # word-level perplexity (streaming over eval dataloader)
    # word-level PPL removed (non-standard and slow)

    # -------------------- save metrics & run test --------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strategy accuracy/F1
    try:
        strat_scores_val = compute_strategy_scores(trainer, val_ds, STRATEGY_NAMES)
        eval_metrics["eval_strategy_acc"] = strat_scores_val["acc"]
        eval_metrics["eval_strategy_f1_weighted"] = strat_scores_val["f1_weighted"]
    except Exception:
        pass

    _save_eval = {k: float(v) for k, v in eval_metrics.items() if v is not None}
    (out_dir / "val_metrics.json").write_text(json.dumps(_save_eval, indent=2))
    # Save validation strategy report as text for reproducibility
    try:
        rep_str_val = compute_strategy_report(trainer, val_ds, STRATEGY_NAMES)
        (out_dir / "val_strategy_report.txt").write_text(rep_str_val)
    except Exception:
        pass

    logger.info("🧪 Evaluating on test set…")
    # 1. Teacher-forcing PPL
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    trainer.args.predict_with_generate = False
    try:
        ppl_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    finally:
        trainer.compute_metrics = original_compute_metrics
        trainer.args.predict_with_generate = True
    
    # 2. Generation metrics
    if trainer.model.generation_config.num_beams == 1:
        trainer.model.generation_config.early_stopping = False
        trainer.model.generation_config.length_penalty = 1.0
    gen_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    # 3. Combine metrics and save (add word-level ppl before writing)
    # Include loss/perplexity explicitly
    final_test_metrics = {
        "test_loss": ppl_metrics.get("test_loss"),
        "test_perplexity": ppl_metrics.get("test_perplexity", (math.exp(ppl_metrics.get("test_loss")) if ppl_metrics.get("test_loss") is not None else None)),
        **{k: v for k, v in gen_metrics.items()},
    }
    # word-level PPL removed (non-standard and slow)
    # strategy acc/f1
    try:
        strat_scores_test = compute_strategy_scores(trainer, test_ds, STRATEGY_NAMES)
        final_test_metrics["test_strategy_acc"] = strat_scores_test["acc"]
        final_test_metrics["test_strategy_f1_weighted"] = strat_scores_test["f1_weighted"]
    except Exception:
        pass

    logger.info("Test metrics: %s", json.dumps(final_test_metrics, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps({k: float(v) for k, v in final_test_metrics.items() if v is not None}, indent=2))

    # Strategy classification report on test set
    try:
        rep_str_test = compute_strategy_report(trainer, test_ds, STRATEGY_NAMES)
        (out_dir / "test_strategy_report.txt").write_text(rep_str_test)
        logger.info("\nStrategy Classification Report (test):\n%s", rep_str_test)
    except Exception as e:
        logger.warning("strategy report (test) failed: %s", e)

    # -------------------- sample generations --------------------
    sample_n = 5
    indices = random.sample(range(len(val_ds)), min(sample_n, len(val_ds)))
    model.eval()

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
        logger.info("\n----- SAMPLE %d -----\nCTX: %s\nREF: %s\nGEN: %s\n", idx, ctx_text, ex["response"], gen_text)


if __name__ == "__main__":
    main()
