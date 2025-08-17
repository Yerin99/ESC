# -*- coding: utf-8 -*-
"""
models/bart_mtl_strategy.py
---------------------------------
Strategy-aware BART wrapper for ESConv with:
- CLS pooling from BOS (default) or EOS
- A strategy classification head on top of encoder CLS
- Mixture strategy embedding appended as an extra encoder token (mask +1)
- Alpha-controlled mixture during training, with optional detachment for stability
- Returns LM loss (from parent) and cls_loss (for trainer to combine)

본 모듈은 `Seq2SeqTrainer(predict_with_generate=True)`와 완전히 호환되도록 설계되었고,
generate() 경로에서도 `prepare_encoder_decoder_kwargs_for_generation`를 오버라이드하여
인코더 출력 뒤에 전략 임베딩 토큰을 붙이고 mask도 +1 하도록 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class StrategySeq2SeqOutput(Seq2SeqLMOutput):
    """Seq2Seq output extended with strategy fields."""
    cls_loss: Optional[torch.FloatTensor] = None
    strategy_logits: Optional[torch.FloatTensor] = None
    strategy_probs: Optional[torch.FloatTensor] = None


class BartForESCWithStrategy(BartForConditionalGeneration):
    """BART with a strategy head and mixture token appended to encoder outputs.

    Runtime knobs (populated by trainer):
        - self.alpha: teacher forcing mixture ratio (0..1)
        - self.cls_weight: weight for classification loss (combined in trainer)
        - self.detach_steps: use detached probs for first N steps
        - self.global_step: updated by trainer for scheduling
        - self.label_smoothing: CE smoothing for strategy labels
    """

    def __init__(self, config, num_strategies: int = 8, **kwargs):
        super().__init__(config)
        self.num_strategies = int(num_strategies)

        self.strategy_head = nn.Linear(config.d_model, self.num_strategies)
        self.strategy_embeddings = nn.Embedding(self.num_strategies, config.d_model)

        # runtime knobs (to be set by trainer)
        self.alpha: float = 1.0
        self.cls_weight: float = 0.3
        self.detach_steps: int = 0
        self.global_step: int = 0
        self.label_smoothing: float = 0.1

        # CE for strategy prediction (ignore_index=-100 by default)
        self._ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # control whether to use teacher (ground-truth) for e_mix construction
        # can be toggled externally: model.use_ground_truth_strategy = True/False
        self.use_ground_truth_strategy: bool = False

    # --------- pooling helpers ---------
    def _pool_bos(self, enc_last_hidden: torch.Tensor) -> torch.Tensor:
        # enc_last_hidden: [B, S, d]
        return enc_last_hidden[:, 0, :]

    def _append_strategy_token(self, encoder_outputs, attention_mask: torch.Tensor, e_mix: torch.Tensor):
        # encoder_outputs.last_hidden_state: [B, S, d]
        h_enc = encoder_outputs.last_hidden_state
        e_mix = e_mix.unsqueeze(1)  # [B,1,d]
        h_plus = torch.cat([h_enc, e_mix], dim=1)  # [B, S+1, d]
        mask_plus = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)

        # in-place update is fine; ensure shapes
        encoder_outputs.last_hidden_state = h_plus
        assert encoder_outputs.last_hidden_state.shape[1] == attention_mask.shape[1] + 1
        assert mask_plus.shape[1] == encoder_outputs.last_hidden_state.shape[1]
        return encoder_outputs, mask_plus

    # --------- generation wiring ---------
    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs, **unused):  # type: ignore
        # Let super compute regular encoder_outputs first
        model_kwargs = super().prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs, **unused)
        encoder_outputs = model_kwargs.get("encoder_outputs")
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Pool CLS (BOS)
        last_hidden = encoder_outputs.last_hidden_state
        h_cls = self._pool_bos(last_hidden)

        # Strategy probs and mixture token (inference: alpha=1.0 effectively)
        logits_s = self.strategy_head(h_cls)
        p_s = torch.softmax(logits_s, dim=-1)
        e_mix = torch.matmul(p_s, self.strategy_embeddings.weight)

        # Append token and grow mask(+1)
        encoder_outputs, attention_mask = self._append_strategy_token(encoder_outputs, attention_mask, e_mix)

        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = attention_mask
        # Flag to avoid double-appending inside forward() during generation
        model_kwargs["strategy_appended"] = True
        return model_kwargs

    # --------- forward override ---------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        strategy_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> StrategySeq2SeqOutput:  # type: ignore
        """Run encoder -> pool -> strategy head -> append e_mix -> decoder (LM).

        Returns LM loss (from parent) and cls_loss separately. The trainer combines them.
        """

        # 1) Run encoder if needed
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        enc_last_hidden = encoder_outputs.last_hidden_state  # [B,S,d]

        # 2) Pool CLS (BOS)
        h_cls = self._pool_bos(enc_last_hidden)

        # 3) Strategy head
        logits_s = self.strategy_head(h_cls)  # [B, num_strategies]
        p_s = torch.softmax(logits_s, dim=-1)

        # 4) Mixture distribution
        p_mix = p_s
        if self.training and strategy_labels is not None:
            # valid entries: label in [0..num_strategies-1]
            valid = (strategy_labels >= 0) & (strategy_labels < self.num_strategies)
            if valid.any():
                if self.use_ground_truth_strategy:
                    # teacher forcing for e_mix: pure one-hot from ground-truth
                    p_mix = p_mix.clone()
                    p_mix[valid] = F.one_hot(strategy_labels[valid], num_classes=self.num_strategies).float()
                else:
                    oh = F.one_hot(strategy_labels[valid], num_classes=self.num_strategies).float()
                    p_mix = p_mix.clone()
                    p_mix_valid = self.alpha * p_s[valid] + (1.0 - self.alpha) * oh
                    p_mix[valid] = p_mix_valid

        # 5) Optional detach for early stability when building e_mix
        if self.training and self.global_step < int(self.detach_steps):
            p_for_token = p_mix.detach()
        else:
            p_for_token = p_mix

        # 6) Build e_mix token and append to encoder outputs; grow mask(+1)
        assert attention_mask is not None, "attention_mask is required"
        enc_len = encoder_outputs.last_hidden_state.size(1)
        mask_len = attention_mask.size(1)
        if enc_len == mask_len:
            # Training/teacher-forcing path: append both hidden and mask
            e_mix = torch.matmul(p_for_token, self.strategy_embeddings.weight)  # [B,d]
            encoder_outputs, attention_mask_plus = self._append_strategy_token(encoder_outputs, attention_mask, e_mix)
        elif enc_len + 1 == mask_len:
            # Mask already extended (e.g., prepare_* extended mask but not hidden). Append hidden only.
            e_mix = torch.matmul(p_for_token, self.strategy_embeddings.weight)  # [B,d]
            h_enc = encoder_outputs.last_hidden_state
            h_plus = torch.cat([h_enc, e_mix.unsqueeze(1)], dim=1)
            encoder_outputs.last_hidden_state = h_plus
            attention_mask_plus = attention_mask  # keep as provided
        elif enc_len == mask_len + 1:
            # Hidden already extended (should be rare). Trim hidden to match mask for safety.
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[:, :mask_len, :]
            attention_mask_plus = attention_mask
        else:
            # Unexpected mismatch; align by min length
            min_len = min(enc_len, mask_len)
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[:, :min_len, :]
            attention_mask_plus = attention_mask[:, :min_len]

        # 7) Decoder + LM loss via parent forward (use prepared encoder_outputs)
        lm_outputs: Seq2SeqLMOutput = super().forward(
            input_ids=None,  # prevent re-encoding
            attention_mask=attention_mask_plus,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # 8) Classification loss with label smoothing
        cls_loss = None
        if strategy_labels is not None:
            # CrossEntropyLoss with label smoothing handles ignore_index=-100 by default
            cls_loss = self._ce(logits_s, strategy_labels)

        if not return_dict:
            output = (lm_outputs.logits,)
            return ((lm_outputs.loss,),) + output  # type: ignore

        return StrategySeq2SeqOutput(
            loss=lm_outputs.loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            decoder_hidden_states=lm_outputs.decoder_hidden_states,
            decoder_attentions=lm_outputs.decoder_attentions,
            cross_attentions=lm_outputs.cross_attentions,
            encoder_last_hidden_state=lm_outputs.encoder_last_hidden_state,
            encoder_hidden_states=lm_outputs.encoder_hidden_states,
            encoder_attentions=lm_outputs.encoder_attentions,
            cls_loss=cls_loss,
            strategy_logits=logits_s,
            strategy_probs=p_s,
        )


