from transformers import BartTokenizer

# BART 토크나이저 로드
tok = BartTokenizer.from_pretrained("facebook/bart-base")

def as_single_token(s):
    # 선행 공백을 붙여 실제 문중 토큰화를 흉내
    toks = tok.tokenize(" " + s)
    return toks[-1], len(toks) == 1

# 테스트
print(as_single_token("question"))     # ('Ġquestion', True/False)
print(as_single_token("summary"))     # ('Ġsummary', True/False)
print(as_single_token("reflection"))   # ('Ġreflection', True/False)
print(as_single_token("disclosure"))     # ('Ġdisclosure', True/False)
print(as_single_token("reassure"))      # ('Ġreassure', True/False)
print(as_single_token("suggest"))      # ('Ġsuggest', True/False)
print(as_single_token("information"))  # ('Ġinformation', True/False) 
print(as_single_token("other"))        # ('Ġother', True/False)