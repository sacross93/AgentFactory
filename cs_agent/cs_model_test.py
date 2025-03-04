from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_model():
    # 모델과 토크나이저 로드
    model_name = "perplexity-ai/r1-1776"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # GPU 사용 가능시 GPU로 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    # 입력 텍스트 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 생성된 텍스트 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 모델 설정
model, tokenizer = setup_model()

# 사용 예시
prompt = "당신의 의견을 말씀해주세요:"
response = generate_response(model, tokenizer, prompt)
print(response)