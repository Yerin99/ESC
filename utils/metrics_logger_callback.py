from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class EvalMetricsLoggerCallback(TrainerCallback):
    """Append eval metrics (incl. eval_loss) to a JSONL file each evaluation.

    Writes to <output_dir>/eval_history.jsonl with one JSON object per line containing
    epoch, step, and all eval_* metrics provided by the Trainer.
    """

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        metrics: Dict[str, float] = kwargs.get("metrics", {}) or {}
        # Coerce epoch None (pre-train initial eval) to 0.0
        epoch_value = float(state.epoch) if state.epoch is not None else 0.0
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch": epoch_value,
            "global_step": int(state.global_step),
        }
        # Only keep eval_* keys (Trainer per-epoch or initial eval). Ignore final_val_*/final_test_*
        for k, v in metrics.items():
            if k.startswith("eval_"):
                try:
                    record[k] = float(v) if v is not None else None
                except Exception:
                    # Fallback to raw value if it cannot be cast
                    record[k] = v

        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "eval_history.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return control


