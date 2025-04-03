# AgentFactory

LLM을 기반으로 다양한 목적의 AI 에이전트를 제작하는 프로젝트입니다.  
현재는 고객 상담용 CS 에이전트를 제공하며, 앞으로 다양한 도메인에 특화된 에이전트들을 지속적으로 추가할 예정입니다.

---

## 📦 현재 제공 중인 Agent: ZeppleMall CS Agent

### 🧾 개요
**Jchyunplace CS Agent**는 제이씨현시스템이 운영하는 쇼핑몰 **제플몰(Jchyunplace)** 이용 고객의 문의에 AI가 자동으로 응답하는 고객 상담용 챗봇입니다.  
쇼핑몰 이용 중 발생하는 궁금증이나 컴퓨터 부품 관련 질문에 대해 빠르고 정확한 답변을 제공합니다.

### 💡 주요 기능
- **제플몰 정보 안내**  
  운영 시간, 배송 정책, AS 안내, 교환/환불 등 기본적인 쇼핑몰 관련 정보 제공
- **AI PC 관련 Q&A**  
  오프라인 LLM 실행에 적합한 AI PC 사양 추천 및 컴퓨터 부품에 대한 설명 제공
- **에이전트 오케스트레이션**  
  사용자의 질문을 분석하고 적절한 서브 에이전트를 호출해 통합 응답 생성

---

## ⚙️ 기술 스택
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: FastAPI, Python
- **LLM 운영**: LangGraph, GPT 기반 Agent
- **Communication**: RESTful API

---

## 🚀 설치 및 실행

### 백엔드 (FastAPI 서버)
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### 프론트엔드
index.html 파일을 웹 브라우저에서 열기

## 🧠 에이전트 구조
1. **Jchyunplace Info Agent**
    - 제플몰 정책 및 정보 안내 (운영 시간, 배송, AS 등)

2. **AI PC Recommendation Agent**
    - 사용자의 용도/예산에 따라 최적의 AI PC 구성 제안

3. **Orchestrator Agent**
    - 질문 유형 분석 → 적절한 서브 에이전트 호출 → 응답 통합

## 📜 라이선스
이 프로젝트는 제이씨현시스템 소속 프로젝트로, 내부 목적에 따라 운영 및 배포됩니다.