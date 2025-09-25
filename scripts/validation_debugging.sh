#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from repo root so Python can import top-level modules like `utils.*`
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.dataset_registry import get_dataset
from utils.tokens import SPEAKER_TOKENS, STRATEGY_TOKENS
import numpy as np, torch
from sklearn.metrics import classification_report, f1_score, accuracy_score

CKPT = "/home/yerin/ESC/outputs/strategy_all_bart/checkpoint-7980"
SPLIT = "validation"     # or "test"
MAX_SRC, MAX_TGT = 1024, 256
BATCH = 16
SAMPLE_N = 64            # None for full split

# tokenizer + special tokens
tok = BartTokenizer.from_pretrained("facebook/bart-base")
tok.add_special_tokens({"additional_special_tokens": list(SPEAKER_TOKENS.values()) + list(STRATEGY_TOKENS.values())})

# model
model = BartForConditionalGeneration.from_pretrained(CKPT)
model.resize_token_embeddings(len(tok))
gc = model.generation_config
gc.max_length = MAX_TGT; gc.num_beams = 1; gc.do_sample = True; gc.top_k = 0; gc.top_p = 0.9; gc.temperature = 1.0; gc.early_stopping = False
model.generation_config = gc

# data
DS = get_dataset("strategy_all_esconv")
ds = DS(split=SPLIT, tokenizer=tok, max_src=MAX_SRC, max_tgt=MAX_TGT, tiny_frac=None)
if SAMPLE_N is not None:
    ds = torch.utils.data.Subset(ds, list(range(min(len(ds), SAMPLE_N))))

collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest")
args = Seq2SeqTrainingArguments(output_dir="outputs/_debug", per_device_eval_batch_size=BATCH, predict_with_generate=True, generation_max_length=MAX_TGT, generation_num_beams=1, report_to="none")
trainer = Seq2SeqTrainer(model=model, args=args, data_collator=collator)

pred = trainer.predict(ds)
pred_ids = pred.predictions
label_ids = pred.label_ids

# strategy id set and mapping to class indices
strategy_names = list(STRATEGY_TOKENS.keys())
strategy_tokens = list(STRATEGY_TOKENS.values())
strat_ids = set()
id2cls = {}
for idx, st in enumerate(strategy_tokens):
    tid = tok.convert_tokens_to_ids(st)
    if tid is not None and tid != tok.unk_token_id:
        tid = int(tid)
        strat_ids.add(tid)
        id2cls[tid] = idx

bos, pad, eos = tok.bos_token_id, tok.pad_token_id, tok.eos_token_id
start = model.config.decoder_start_token_id

def ref_first_id(arr):
    for t in arr:
        if t != -100 and t != pad:
            return int(t)
    return None

def hyp_first_id(arr):
    """Return the first token after skipping the initial decoder_start_token_id once,
    then skipping BOS/PAD if present."""
    first_seen = True
    for t in arr:
        t = int(t)
        if first_seen and start is not None and t == start:
            first_seen = False
            continue
        first_seen = False
        if bos is not None and t == bos:
            continue
        if t == pad:
            continue
        return t
    return None

tp=fp=fn=0; support=0
pred_is_eos=pred_is_strat=pred_is_other=0

samples=[]
y_true=[]; y_pred=[]
for i in range(len(pred_ids)):
    r = ref_first_id(label_ids[i])
    h = hyp_first_id(pred_ids[i])
    if r in strat_ids and r is not None and h is not None:
        support += 1
        if h == r: tp += 1
        else:
            fn += 1
            if h in strat_ids: fp += 1
        # collect for classification metrics
        rc = id2cls.get(int(r), -1)
        hc = id2cls.get(int(h), -1)
        if rc != -1 and hc != -1:
            y_true.append(rc)
            y_pred.append(hc)
    # distribution is based on h (first token after skipping decoder start)
    if h == eos: pred_is_eos += 1
    elif h in strat_ids: pred_is_strat += 1
    else: pred_is_other += 1
    if i < 20:
        samples.append(dict(
            ref_id=r, ref_tok=tok.convert_ids_to_tokens(r) if r is not None else None,
            hyp_id=h, hyp_tok=tok.convert_ids_to_tokens(h) if h is not None else None
        ))

acc_simple = tp/support if support>0 else 0.0

# classification metrics over classes
if y_true:
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, labels=list(range(len(strategy_names))), target_names=strategy_names, digits=3, zero_division=0)
else:
    acc = 0.0
    f1_weighted = 0.0
    report = ""

print("support:", support, "tp:", tp, "fp:", fp, "fn:", fn)
print("pred_first distribution -> eos:", pred_is_eos, "strategy:", pred_is_strat, "other:", pred_is_other)
print("token-level acc (tp/support):", acc_simple)
print("class-metrics -> acc:", acc, "f1_weighted:", f1_weighted)
print("classification_report:\n" + report)
print("samples:")
for s in samples: print(s)
PY


