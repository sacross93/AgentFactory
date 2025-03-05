import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. 테스트 데이터 준비: 각 문장 쌍과 예상 유사도(유사 / 비유사)를 포함합니다.
test_data = [
    {
        "sentence1": "오늘 날씨가 맑고 화창합니다.",
        "sentence2": "하늘이 맑아서 기분이 좋습니다.",
        "expected": "유사"
    },
    {
        "sentence1": "오늘 날씨가 맑고 화창합니다.",
        "sentence2": "나는 오늘 점심으로 김치찌개를 먹었습니다.",
        "expected": "비유사"
    },
    {
        "sentence1": "이 책은 정말 재미있어요.",
        "sentence2": "이 소설은 매우 흥미진진합니다.",
        "expected": "유사"
    },
    {
        "sentence1": "나는 자전거를 타고 출근합니다.",
        "sentence2": "컴퓨터 프로그래밍은 내 취미입니다.",
        "expected": "비유사"
    },
    {
        "sentence1": "한국어 임베딩 모델은 자연어 처리에 중요합니다.",
        "sentence2": "자연어 처리에서 한국어 모델은 핵심 역할을 합니다.",
        "expected": "유사"
    },
    {
        "sentence1": "오늘은 회의를 진행합니다.",
        "sentence2": "저녁에 친구들과 영화를 봅니다.",
        "expected": "비유사"
    }
]

# 2. 테스트할 모델 목록
model_names = [
    "intfloat/multilingual-e5-large",
    "jhgan/ko-sbert-multitask",
    "BAAI/bge-m3",
    "upskyy/bge-m3-korean"
    # "Alibaba-NLP/gte-Qwen2-7B-instruct"
]

# 3. 각 모델에 대해 테스트 실행
results = {}

for model_name in model_names:
    print(f"테스트 중인 모델: {model_name}")
    model = SentenceTransformer(model_name)
    model_results = []
    for pair in test_data:
        s1 = pair["sentence1"]
        s2 = pair["sentence2"]
        expected = pair["expected"]
        # 문장별 임베딩 생성
        emb1 = model.encode(s1)
        emb2 = model.encode(s2)
        # 코사인 유사도 계산
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        model_results.append({
            "문장1": s1,
            "문장2": s2,
            "예상": expected,
            "계산된 유사도": round(similarity, 4)
        })
    results[model_name] = model_results

# 4. 결과 출력
for model_name, res in results.items():
    print(f"\n모델: {model_name} 테스트 결과")
    df = pd.DataFrame(res)
    print(df.to_string(index=False))
