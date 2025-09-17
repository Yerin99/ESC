from __future__ import annotations

import math
from typing import Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class PerplexityCallback(TrainerCallback):
    """Compute eval_ppl from eval_loss whenever evaluation completes.

    Adds `eval_ppl` into `metrics` dict so it is logged and saved alongside others.
    """

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics: Dict[str, float] = kwargs.get("metrics", {})
        # eval_loss may be present; compute ppl = exp(loss)
        if metrics is not None and "eval_loss" in metrics and metrics["eval_loss"] is not None:
            try:
                metrics["eval_ppl"] = float(math.exp(metrics["eval_loss"]))
            except OverflowError:
                metrics["eval_ppl"] = float("inf")
        return control


