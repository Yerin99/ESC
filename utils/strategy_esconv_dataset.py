"""
Strategy ESConv dataset: Append a strategy token right after the sys speaker token
for every previous system turn in the context. Target remains plain response text.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import BartTokenizer

from utils.tokens import SPEAKER_TOKENS, STRATEGY_TOKENS


class StrategyESConvDataset(torch.utils.data.Dataset):
    """ESConv dataset variant that injects strategy tokens after sys speaker tokens.

    The context sequence is built as: [BOS] (USR|SYS+STRAT) text [EOS] ... [EOS] [EOS]
    Only previous turns are included in the context; the target is the current sys turn text.
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

                # Build context from all prior turns; for sys turns, add strategy token
                context_parts: List[str] = []
                for prev in dialog[:turn_index]:
                    if prev.get("speaker") == "usr":
                        context_parts.append(f"{usr_tok}{prev['text']}")
                    else:
                        strat_name = prev.get("strategy", "Others")
                        strat_tok = strat2tok.get(strat_name, STRATEGY_TOKENS["Others"])
                        context_parts.append(f"{sys_tok}{strat_tok}{prev['text']}")

                context_text = (
                    tokenizer.bos_token
                    + (eos_sep.join(context_parts) if context_parts else "")
                    + tokenizer.eos_token
                )
                if not context_text.strip():
                    continue

                # Tokenize source/target with left truncation for context
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
                dec = tokenizer(
                    turn["text"],
                    max_length=max_tgt,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )

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
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


