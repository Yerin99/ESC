# -*- coding: utf-8 -*-
"""tokens.py
공통 토큰 정의 모듈.
- SPEAKER_TOKENS: 발화자 구분
- STRATEGY_TOKENS: 전략 라벨 (향후 사용)
BART/다른 모델 스크립트에서 import 해서 사용한다.
"""
from typing import Dict, List

# Speaker tokens
SPEAKER_TOKENS: Dict[str, str] = {
    "usr": "[USR]",
    "sys": "[SYS]",
}

# Strategy tokens (reserved)
STRATEGY_TOKENS: Dict[str, str] = {
    "Question": "[STRAT_Question]",
    "Restatement or Paraphrasing": "[STRAT_Paraphrasing]",
    "Reflection of feelings": "[STRAT_Reflection]",
    "Self-disclosure": "[STRAT_SelfDisclosure]",
    "Affirmation and Reassurance": "[STRAT_Reassurance]",
    "Providing Suggestions": "[STRAT_Suggestion]",
    "Information": "[STRAT_Information]",
    "Others": "[STRAT_Others]",
}

# Unified dictionary (speaker + strategy) for convenience
SPECIAL_TOKENS: Dict[str, str] = {**SPEAKER_TOKENS, **STRATEGY_TOKENS}

# List of strategy names
STRATEGY_NAMES: List[str] = list(STRATEGY_TOKENS.keys())
