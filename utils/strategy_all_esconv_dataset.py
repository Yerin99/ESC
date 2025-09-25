"""
Strategy-ALL ESConv dataset: decoder targets start with the ground-truth strategy token
followed by the system response text. Context only contains previous turns, with previous
sys turns annotated by their (past) strategy tokens. No leakage of the current turn's
strategy into the context.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import BartTokenizer

from utils.tokens import SPEAKER_TOKENS, STRATEGY_TOKENS


class StrategyAllESConvDataset(torch.utils.data.Dataset):
    """ESConv dataset variant that teaches the decoder to first output the strategy token.

    For each target sys turn, the decoder target is: <BOS> [STRAT_X] response_text ... <EOS>.
    The context is built from all prior turns; previous sys turns include their strategy token.
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
        strat2tok = STRATEGY_TOKENS
        eos_sep = f" {tokenizer.eos_token}"

        self.examples: List[Dict[str, Any]] = []
        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]
            for turn_index, turn in enumerate(dialog):
                if turn.get("speaker") != "sys":
                    continue

                # Build context from prior turns, annotate only previous sys with their strategy
                context_parts: List[str] = []
                for prev in dialog[:turn_index]:
                    if prev.get("speaker") == "usr":
                        context_parts.append(f"{usr_tok}{prev['text']}")
                    else:
                        prev_strat = prev.get("strategy", "Others")
                        strat_tok = strat2tok.get(prev_strat, STRATEGY_TOKENS["Others"])
                        context_parts.append(f"{sys_tok}{strat_tok}{prev['text']}")

                context_text = (
                    tokenizer.bos_token
                    + (eos_sep.join(context_parts) if context_parts else "")
                    + tokenizer.eos_token
                )
                if not context_text.strip():
                    # allow empty context but keep BOS/EOS as constructed
                    pass

                # Left truncation for context
                old_side = tokenizer.truncation_side
                tokenizer.truncation_side = "left"
                enc = tokenizer(
                    context_text,
                    max_length=max_src,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False,
                )
                tokenizer.truncation_side = old_side

                # Build decoder target = [STRAT_cur] + response_text
                cur_strat = turn.get("strategy", "Others")
                cur_strat_tok = strat2tok.get(cur_strat, STRATEGY_TOKENS["Others"])
                target_text = f"{cur_strat_tok}{turn['text']}"
                dec = tokenizer(
                    target_text,
                    max_length=max_tgt,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )

                # Labels: mask PAD and BOS only; keep strategy token (first non-BOS) and EOS
                labels = list(dec.input_ids)
                labels = [(-100 if tok == tokenizer.pad_token_id else tok) for tok in labels]
                if labels and dec.input_ids[0] == tokenizer.bos_token_id:
                    labels[0] = -100

                self.examples.append(
                    {
                        "input_ids": enc.input_ids,
                        "attention_mask": enc.attention_mask,
                        "labels": labels,
                        "context": context_text,
                        "response": turn["text"],
                        "strategy": cur_strat,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


