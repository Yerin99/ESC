"""
Problem-type prefixed ESConv dataset: Prepend natural-language problem type to the dialog context.

- Add a separator token after the problem type (tokenizer.sep_token, fallback to eos).
- If the first utterance is from SYS, delay inserting the prefix until the first USR utterance,
  attaching the prefix after that first USR text with a sep token.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import BartTokenizer

from utils.tokens import SPEAKER_TOKENS


class ProblemTypeESConvDataset(torch.utils.data.Dataset):
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
            problem_type = js.get("problem_type", "")
            prob_line = f"Problem type: {problem_type}." if problem_type else ""

            for turn_index, turn in enumerate(dialog):
                if turn.get("speaker") != "sys":
                    continue

                # Build context with problem_type prefix
                context_parts: List[str] = []
                seen_usr = False
                for prev in dialog[:turn_index]:
                    if prev.get("speaker") == "usr":
                        if not seen_usr and prob_line:
                            seen_usr = True
                            context_parts.append(f"{usr_tok}{prev['text']}{sep_tok}{prob_line}{sep_tok}")
                        else:
                            context_parts.append(f"{usr_tok}{prev['text']}")
                    else:
                        context_parts.append(f"{sys_tok}{prev['text']}")

                # Build tokens with prefix-reserved left truncation
                pre_parts: List[str] = []
                first_usr_block: Optional[str] = None
                post_parts: List[str] = []

                seen_usr2 = False
                for prev in dialog[:turn_index]:
                    if prev.get("speaker") == "usr":
                        if not seen_usr2:
                            seen_usr2 = True
                            if prob_line:
                                first_usr_block = f"{usr_tok}{prev['text']}{sep_tok}{prob_line}{sep_tok}"
                            else:
                                first_usr_block = f"{usr_tok}{prev['text']}"
                        else:
                            post_parts.append(f"{usr_tok}{prev['text']}")
                    else:
                        if not seen_usr2:
                            pre_parts.append(f"{sys_tok}{prev['text']}")
                        else:
                            post_parts.append(f"{sys_tok}{prev['text']}")

                bos_id = tokenizer.bos_token_id
                eos_id = tokenizer.eos_token_id
                pad_id = tokenizer.pad_token_id

                pre_ids = tokenizer(eos_sep.join(pre_parts), add_special_tokens=False).input_ids if pre_parts else []
                first_ids = tokenizer(first_usr_block, add_special_tokens=False).input_ids if first_usr_block else []
                post_ids = tokenizer(eos_sep.join(post_parts), add_special_tokens=False).input_ids if post_parts else []

                reserved_suffix_len = 0
                if prob_line and first_usr_block:
                    reserved_suffix_len = len(tokenizer(f"{sep_tok}{prob_line}{sep_tok}", add_special_tokens=False).input_ids)

                total_len = 1 + len(pre_ids) + len(first_ids) + len(post_ids) + 1
                overflow = max(0, total_len - max_src)

                if overflow > 0 and pre_ids:
                    cut = min(overflow, len(pre_ids))
                    pre_ids = pre_ids[cut:]
                    overflow -= cut

                if overflow > 0 and first_ids:
                    keep_min = reserved_suffix_len
                    max_cut = max(0, len(first_ids) - keep_min)
                    cut = min(overflow, max_cut)
                    if cut > 0:
                        first_ids = first_ids[cut:]
                        overflow -= cut

                if overflow > 0 and post_ids:
                    cut = min(overflow, len(post_ids))
                    post_ids = post_ids[cut:]
                    overflow -= cut

                ids = [bos_id] + pre_ids + first_ids + post_ids + [eos_id]
                if not ids:
                    continue
                if len(ids) < max_src:
                    attn = [1] * len(ids) + [0] * (max_src - len(ids))
                    ids = ids + [pad_id] * (max_src - len(ids))
                else:
                    attn = [1] * max_src
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
                        "input_ids": ids,
                        "attention_mask": attn,
                        "labels": labels,
                        "context": tokenizer.decode([t for t in ids if t != pad_id], skip_special_tokens=True),
                        "response": turn["text"],
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


