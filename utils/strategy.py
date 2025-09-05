# -*- coding: utf-8 -*-
"""
utils/strategy.py
------------------
Reusable helpers for strategy-aware training/evaluation:
- DataCollatorWithStrategy: adds `strategy_labels` tensor to batches
- compute_strategy_report: run encoder CLS strategy head and return sklearn classification_report
"""

from __future__ import annotations

from typing import List

import torch
from transformers import DataCollatorForSeq2Seq
from sklearn.metrics import classification_report, accuracy_score, f1_score


class DataCollatorWithStrategy(DataCollatorForSeq2Seq):
    def __call__(self, features):  # type: ignore[override]
        strat = [f.get("strategy_labels", -100) for f in features]
        batch = super().__call__(features)
        batch["strategy_labels"] = torch.tensor(strat, dtype=torch.long)
        return batch


def compute_strategy_report(trainer, dataset, strategy_names: List[str]) -> str:
    """Compute classification_report for encoder CLS-based strategy predictions.

    Labels come from `dataset.examples[i]["strategy_labels"]`.
    """
    model = trainer.model
    device = model.device

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for ex in getattr(dataset, "examples", []):
            input_ids = torch.tensor([ex["input_ids"]], device=device)
            attention_mask = torch.tensor([ex["attention_mask"]], device=device)
            enc = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            h_cls = enc.last_hidden_state[:, 0, :]
            logits = model.strategy_head(h_cls)
            pred = int(logits.argmax(dim=-1).item())
            y_pred.append(pred)
            y_true.append(int(ex.get("strategy_labels", -100)))

    # drop any invalid labels if present
    pairs = [(p, t) for p, t in zip(y_pred, y_true) if 0 <= int(t) < len(strategy_names)]
    if not pairs:
        return ""
    y_pred_f = [p for p, _ in pairs]
    y_true_f = [t for _, t in pairs]
    return classification_report(y_true_f, y_pred_f, labels=list(range(len(strategy_names))),
                                 target_names=strategy_names, digits=3, zero_division=0)


def compute_strategy_scores(trainer, dataset, strategy_names: List[str]) -> dict:
    """Return accuracy and weighted-F1 for encoder CLS-based strategy prediction."""
    model = trainer.model
    device = model.device

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for ex in getattr(dataset, "examples", []):
            input_ids = torch.tensor([ex["input_ids"]], device=device)
            attention_mask = torch.tensor([ex["attention_mask"]], device=device)
            enc = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            h_cls = enc.last_hidden_state[:, 0, :]
            logits = model.strategy_head(h_cls)
            pred = int(logits.argmax(dim=-1).item())
            y_pred.append(pred)
            y_true.append(int(ex.get("strategy_labels", -100)))

    pairs = [(p, t) for p, t in zip(y_pred, y_true) if 0 <= int(t) < len(strategy_names)]
    if not pairs:
        return {"acc": 0.0, "f1_weighted": 0.0}
    y_pred_f = [p for p, _ in pairs]
    y_true_f = [t for _, t in pairs]
    return {
        "acc": float(accuracy_score(y_true_f, y_pred_f)),
        "f1_weighted": float(f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)),
    }

# Backward-compat aliases (safe to keep temporarily)
compute_teacher_strategy_report = compute_strategy_report
compute_teacher_strategy_scores = compute_strategy_scores


