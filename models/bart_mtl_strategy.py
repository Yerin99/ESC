# -*- coding: utf-8 -*-
"""
models/bart_mtl_strategy.py
---------------------------------
Strategy-aware BART wrapper for ESConv with:
- CLS pooling from BOS (default) or EOS
- A strategy classification head on top of encoder CLS
- Mixture strategy embedding computed from strategy distribution p, but NOT concatenated to encoder outputs (to enable future MISC-style separate cross-attention)
- Alpha-controlled mixture during training, with optional detachment for stability
- Returns LM loss (from parent) and cls_loss (for trainer to combine)

본 모듈은 `Seq2SeqTrainer(predict_with_generate=True)`와 완전히 호환되도록 설계되었고,
generate() 경로에서도 `prepare_encoder_decoder_kwargs_for_generation`를 오버라이드하여
전략 분포/벡터만 계산해 전달하며, 인코더 출력 뒤에 토큰을 더 이상 붙이지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import (
    BartDecoderLayer,
    BartAttention,
)
from transformers.activations import ACT2FN
import copy
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

        logits_s = self.strategy_head(h_cls)
        p_s = torch.softmax(logits_s, dim=-1)
        e_mix = torch.matmul(p_s, self.strategy_embeddings.weight)

        # Do NOT append e_mix to encoder outputs anymore. Keep lengths unchanged.
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = attention_mask
        # Expose strategy vector for potential downstream usage (e.g., custom cross-attn)
        model_kwargs["strategy_vector"] = e_mix
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

        # 6) Build e_mix vector (not appended). Keep for potential auxiliary uses.
        assert attention_mask is not None, "attention_mask is required"
        e_mix = torch.matmul(p_for_token, self.strategy_embeddings.weight)  # [B,d]

        # 7) Decoder + LM loss via parent forward (use prepared encoder_outputs)
        lm_outputs: Seq2SeqLMOutput = super().forward(
            input_ids=None,  # prevent re-encoding
            attention_mask=attention_mask,
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



# ========================= Dual Cross-Attention (MISC-style) =========================
class StrategyAwareBartDecoderLayer(BartDecoderLayer):
    """BartDecoderLayer 확장: 문맥 cross-attn과 별도의 전략 cross-attn을 추가한다.

    - 기존 self-attn → encoder cross-attn 순서를 유지하고, 그 뒤에 strategy cross-attn을 수행한다.
    - 전략 메모리는 길이 1의 시퀀스(e_mix)에 해당하며, 마스크는 생략 가능하다.
    - 게이트 파라미터를 통해 전략 경로의 기여도를 학습적으로 조절한다.
    """

    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.d_model
        # 별도 전략 cross-attn: 현재 레이어의 encoder_attn과 완전히 동일한 구성으로 복제하여
        # 버전 차이에 따른 시그니처 이슈를 피한다.
        self.strategy_attn = copy.deepcopy(self.encoder_attn)
        self.strategy_attn_layer_norm = nn.LayerNorm(embed_dim)
        # 게이트는 (0,1) 범위로 사용하기 위해 시그모이드로 squashing
        self.strategy_gate = nn.Parameter(torch.tensor(0.2))

        # 레이어 호출 시 전략 메모리 컨텍스트를 임시로 주입하기 위한 버퍼
        self._strategy_hidden_states: Optional[torch.Tensor] = None
        self._strategy_attention_mask: Optional[torch.Tensor] = None

        # 활성화/드롭아웃은 부모의 구성을 사용 (self.activation_fn 등)

    def set_strategy_context(
        self,
        strategy_hidden_states: Optional[torch.Tensor],
        strategy_attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._strategy_hidden_states = strategy_hidden_states
        self._strategy_attention_mask = strategy_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        # ---- Self-Attention ----
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # ---- Encoder Cross-Attention ----
        cross_attn_weights = None
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            encoder_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=encoder_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # ---- Strategy Cross-Attention (length=1 memory) ----
        if self._strategy_hidden_states is not None:
            residual = hidden_states
            strat_out, _, _ = self.strategy_attn(
                hidden_states=hidden_states,
                key_value_states=self._strategy_hidden_states,
                attention_mask=self._strategy_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=False,
            )
            gate = torch.sigmoid(self.strategy_gate)
            strat_out = gate * strat_out
            strat_out = nn.functional.dropout(strat_out, p=self.dropout, training=self.training)
            hidden_states = residual + strat_out
            hidden_states = self.strategy_attn_layer_norm(hidden_states)

        # ---- FFN ----
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # 캐시는 상위에서 비활성화했으나, 호출 호환성을 위해 방어적 처리 유지
        if use_cache and present_key_value is not None and cross_attn_present_key_value is not None:
            present_key_value = present_key_value + cross_attn_present_key_value
        elif use_cache and present_key_value is not None:
            present_key_value = present_key_value
        else:
            present_key_value = None

        return hidden_states, self_attn_weights, cross_attn_weights, present_key_value


class BartForESCWithStrategyDualAttn(BartForESCWithStrategy):
    """MISC 스타일: 디코더 레이어마다 전략용 cross-attn을 추가한 모델.

    - `BartForESCWithStrategy`의 전략 추정/로스 경로는 유지.
    - 인코더 출력 뒤 concat은 하지 않으며, 전략 벡터는 별도 cross-attn 메모리로 주입.
    - 생성 경로에서는 `prepare_encoder_decoder_kwargs_for_generation`에서 전략 메모리를 전달한다.
    """

    def __init__(self, config, num_strategies: int = 8, **kwargs):
        super().__init__(config, num_strategies=num_strategies, **kwargs)
        # 디코더 레이어를 전략-aware 레이어로 교체
        new_layers = nn.ModuleList()
        for old in self.model.decoder.layers:
            new_layer = StrategyAwareBartDecoderLayer(config)
            # 공통 가중치 복사 (존재하는 모듈만)
            try:
                new_layer.self_attn.load_state_dict(old.self_attn.state_dict())
                if getattr(old, "encoder_attn", None) is not None:
                    new_layer.encoder_attn.load_state_dict(old.encoder_attn.state_dict())
                new_layer.self_attn_layer_norm.load_state_dict(old.self_attn_layer_norm.state_dict())
                new_layer.encoder_attn_layer_norm.load_state_dict(old.encoder_attn_layer_norm.state_dict())
                new_layer.fc1.load_state_dict(old.fc1.state_dict())
                new_layer.fc2.load_state_dict(old.fc2.state_dict())
                new_layer.final_layer_norm.load_state_dict(old.final_layer_norm.state_dict())
            except Exception:
                pass
            new_layers.append(new_layer)
        self.model.decoder.layers = new_layers

    # generation 시 전략 메모리를 전달
    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs, **unused):  # type: ignore
        model_kwargs = super().prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs, **unused)
        encoder_outputs = model_kwargs.get("encoder_outputs")
        last_hidden = encoder_outputs.last_hidden_state
        h_cls = self._pool_bos(last_hidden)
        logits_s = self.strategy_head(h_cls)
        p_s = torch.softmax(logits_s, dim=-1)
        e_mix = torch.matmul(p_s, self.strategy_embeddings.weight)  # [B,d]
        # 길이 1 전략 메모리와 마스크 전달
        model_kwargs["strategy_hidden_states"] = e_mix.unsqueeze(1)  # [B,1,d]
        model_kwargs["strategy_attention_mask"] = None
        return model_kwargs

    def _set_strategy_context_on_layers(self, strategy_hidden_states: Optional[torch.Tensor], strategy_attention_mask: Optional[torch.Tensor]) -> None:
        for layer in self.model.decoder.layers:
            if isinstance(layer, StrategyAwareBartDecoderLayer):
                layer.set_strategy_context(strategy_hidden_states, strategy_attention_mask)

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
        strategy_hidden_states: Optional[torch.Tensor] = None,
        strategy_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> StrategySeq2SeqOutput:  # type: ignore
        # strategy_hidden_states가 주어지지 않으면 현재 배치에서 계산
        if strategy_hidden_states is None:
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
            enc_last_hidden = encoder_outputs.last_hidden_state
            h_cls = self._pool_bos(enc_last_hidden)
            logits_s = self.strategy_head(h_cls)
            p_s = torch.softmax(logits_s, dim=-1)
            p_mix = p_s
            if self.training and strategy_labels is not None:
                valid = (strategy_labels >= 0) & (strategy_labels < self.num_strategies)
                if valid.any():
                    if self.use_ground_truth_strategy:
                        p_mix = p_mix.clone()
                        p_mix[valid] = nn.functional.one_hot(strategy_labels[valid], num_classes=self.num_strategies).float()
                    else:
                        oh = nn.functional.one_hot(strategy_labels[valid], num_classes=self.num_strategies).float()
                        p_mix = p_mix.clone()
                        p_mix_valid = self.alpha * p_s[valid] + (1.0 - self.alpha) * oh
                        p_mix[valid] = p_mix_valid
            if self.training and self.global_step < int(self.detach_steps):
                p_for_token = p_mix.detach()
            else:
                p_for_token = p_mix
            e_mix = torch.matmul(p_for_token, self.strategy_embeddings.weight)
            strategy_hidden_states = e_mix.unsqueeze(1)
            strategy_attention_mask = None

        # 레이어에 전략 컨텍스트 주입
        self._set_strategy_context_on_layers(strategy_hidden_states, strategy_attention_mask)
        try:
            # 부모 클래스의 forward를 호출하여 LM/CLS 경로 유지
            out = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                strategy_labels=strategy_labels,
            )
        finally:
            # 컨텍스트 정리 (메모리 누수/캐시 오염 방지)
            self._set_strategy_context_on_layers(None, None)
        return out

