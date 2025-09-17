"""
Pure ESConv dataset for BART pretraining (context -> response), without strategy labels.

This dataset constructs source sequences by concatenating prior dialog turns with
explicit speaker tokens and adds BOS/EOS according to the tokenizer settings.
The target sequence is the next system turn. Labels are masked for PAD and BOS
to ensure fair cross-entropy computation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import BartTokenizer

from utils.tokens import SPEAKER_TOKENS


class PureESConvDataset(torch.utils.data.Dataset):
    """ESConv dataset that yields dicts of input/attention/labels for seq2seq.

    Args:
        split: One of "train", "validation", "test".
        tokenizer: A `transformers.BartTokenizer` (with speaker tokens already added).
        max_src: Max source length for the context sequence.
        max_tgt: Max target length for the response sequence.
        tiny_frac: Optional fraction (0~1) of the split for quick debugging.
        dataset_name: Hugging Face dataset name. Defaults to "thu-coai/esconv".

    Returns:
        Each item is a dict suitable for `DataCollatorForSeq2Seq`, including:
        - input_ids, attention_mask, labels
        - context, response (for logging/debugging)
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
        eos_sep = f" {tokenizer.eos_token}"

        self.examples: List[Dict[str, Any]] = []
        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]
            for turn_index, turn in enumerate(dialog):
                if turn.get("speaker") != "sys":
                    continue

                # Build context from all prior turns
                context_parts: List[str] = []
                for prev in dialog[:turn_index]:
                    spk = usr_tok if prev.get("speaker") == "usr" else sys_tok
                    context_parts.append(f"{spk}{prev['text']}")

                context_text = (
                    tokenizer.bos_token
                    + (eos_sep.join(context_parts) if context_parts else "")
                    + tokenizer.eos_token
                )
                if not context_text.strip():
                    continue

                # Tokenize source/target
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


