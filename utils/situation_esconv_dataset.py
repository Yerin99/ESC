"""
Situation-prefixed ESConv dataset: Prepend natural-language situation text to the dialog context.

- Add a separator token after the situation (tokenizer.sep_token, fallback to eos).
- If the first utterance in the dialog is from SYS, delay inserting the situation until
  the first USR utterance appears in the history (prefix attaches to that first USR line).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import BartTokenizer

from utils.tokens import SPEAKER_TOKENS


class SituationESConvDataset(torch.utils.data.Dataset):
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
        eos_sep = f" {tokenizer.eos_token}"
        sep_tok = tokenizer.sep_token or tokenizer.eos_token

        self.examples: List[Dict[str, Any]] = []
        for ex in raw:
            js = json.loads(ex["text"])
            dialog = js["dialog"]
            situation = js.get("situation", "")
            situ_line = f"Situation: {situation}." if situation else ""

            for turn_index, turn in enumerate(dialog):
                if turn.get("speaker") != "sys":
                    continue

                # Build context with situation prefix
                context_parts: List[str] = []
                seen_usr = False
                for prev in dialog[:turn_index]:
                    if prev.get("speaker") == "usr":
                        if not seen_usr and situ_line:
                            seen_usr = True
                            # Attach situation after the first USR text with a sep token
                            context_parts.append(f"{usr_tok}{prev['text']}{sep_tok}{situ_line}{sep_tok}")
                        else:
                            context_parts.append(f"{usr_tok}{prev['text']}")
                    else:
                        context_parts.append(f"{sys_tok}{prev['text']}")

                context_text = (
                    tokenizer.bos_token
                    + (eos_sep.join(context_parts) if context_parts else "")
                    + tokenizer.eos_token
                )
                if not context_text.strip():
                    continue

                enc = tokenizer(
                    context_text,
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


