# utils/stats.py
"""Utility functions for evaluation statistics across scripts."""
from typing import Tuple, Sequence, Optional

from transformers import PreTrainedTokenizerBase


def _tokenize_words(text: str) -> int:
    """Return number of words using NLTK word_tokenize if available; fallback to regex split."""
    try:
        import nltk
        try:
            # If punkt is not available, this will raise and fall back
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # Fallback to simple split; we don't download in library code
            raise
        from nltk.tokenize import word_tokenize
        return len(word_tokenize(text.lower()))
    except Exception:
        import re
        # Split on sequences of alphabetic characters as an approximation
        return len([w for w in re.findall(r"[A-Za-z]+", text)]) or max(1, len(text.split()))


def calc_token_word_stats(dataset, tokenizer: PreTrainedTokenizerBase,
                          label_key: str = "labels", response_key: str = "response") -> Tuple[int, int]:
    """Return (total_valid_subword_tokens, total_words) for word-level PPL.

    Args:
        dataset: A Dataset object whose `.examples` is a list of dicts.
        tokenizer: HuggingFace tokenizer to identify `pad_token_id`.
        label_key: Key in each example that contains target ids (with -100 masking).
        response_key: Key that stores reference response string.
    """
    total_tok, total_word = 0, 0
    pad_id = tokenizer.pad_token_id

    for ex in getattr(dataset, "examples", []):
        labels = ex.get(label_key, [])
        total_tok += sum(1 for t in labels if t not in (-100, pad_id))
        total_word += _tokenize_words(ex.get(response_key, ""))

    return total_tok, max(total_word, 1)


def compute_word_perplexity_streaming(trainer, dataset, tokenizer: PreTrainedTokenizerBase,
                                      exclude_token_ids: Optional[Sequence[int]] = None) -> float:
    """Compute word-level PPL by streaming over eval dataloader with logits.

    Notes on correctness:
    - Standard perplexity in NLP is subword-level: exp(average NLL over non-ignored target tokens).
      Our training/eval loop computes this via `exp(eval_loss)` and exposes it as `*_perplexity`.
    - This function returns an approximate word-level PPL by dividing accumulated token-level NLL by
      the number of reference words measured with a robust tokenizer fallback. It is useful for
      cross-paper comparison but should not replace the standard subword PPL.
    """
    import math
    import torch
    import torch.nn.functional as F

    # count words once from raw dataset
    _, total_words = calc_token_word_stats(dataset, tokenizer)

    model = trainer.model
    model.eval()

    eval_loader = trainer.get_eval_dataloader(dataset)

    sum_nll = 0.0

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    special_exclude = set(exclude_token_ids or [])
    for tid in (bos_id,):  # BOS is typically excluded by labels=-100; keep for safety
        if tid is not None:
            special_exclude.add(tid)

    device = model.device
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            outputs = model(**batch, use_cache=False)
            logits = outputs.logits  # (B, T, V)
            labels = batch.get("labels")  # (B, T)
            if labels is None:
                continue

            log_probs = F.log_softmax(logits, dim=-1)
            # mask valid positions
            mask = labels.ne(-100)
            if special_exclude:
                for tid in special_exclude:
                    if tid is None:
                        continue
                    mask &= labels.ne(int(tid))

            # gather log p for gold ids (clamp negative to 0 just for gather safety)
            gather_ids = labels.clamp_min(0).unsqueeze(-1)
            gold_logp = log_probs.gather(dim=-1, index=gather_ids).squeeze(-1)
            nll = -gold_logp[mask].sum().item()
            sum_nll += nll

    # word-level cross-entropy and ppl
    ce_per_word = sum_nll / float(total_words)
    return math.exp(ce_per_word)
