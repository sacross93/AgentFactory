from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import duckdb
import pandas as pd
import re

# Ollama 모델 초기화 - 온도 추가
llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1  # 낮은 온도로 더 일관된 응답 유도
)

# 데이터베이스 연결
conn = duckdb.connect('pc_parts.db')

# 상태 정의
class PCCompatibilityState(TypedDict):
    question: str
    search_keywords: List[str]
    part_types: List[str]
    queries: Dict[str, str]
    results: Dict[str, Any]
    errors: List[str]
    final_result: Optional[Dict[str, Any]]

# 데이터베이스 샘플 가져오기
def get_db_samples():
    samples = {}
    try:
        # CPU 샘플
        samples['cpu'] = conn.execute("SELECT model_name FROM cpu LIMIT 5").fetchall()
        # GPU 샘플
        samples['gpu'] = conn.execute("SELECT model_name FROM gpu LIMIT 5").fetchall()
        # 마더보드 샘플
        samples['motherboard'] = conn.execute("SELECT model_name FROM motherboard LIMIT 5").fetchall()
        
        # 리스트로 변환
        for key in samples:
            samples[key] = [item[0] for item in samples[key]]
    except Exception as e:
        print(f"Error getting samples: {str(e)}")
    
    return samples

# 데이터베이스 샘플 가져오기
db_samples = get_db_samples()

# 데이터베이스 스키마 정보 가져오기
def get_db_schema():
    tables = conn.execute("SHOW TABLES").fetchall()
    schema_info = {}
    
    for table in tables:
        table_name = table[0]
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        schema_info[table_name] = [col[1] for col in columns]
    
    return schema_info

# 스키마 정보 가져오기
db_schema = get_db_schema()

# 테이블 이름 매핑 확장
table_mapping = {
    "cpu_support": "cpu_mb_compatibility",
    "motherboard_compatibility": "cpu_mb_compatibility",
    "cpu_motherboard": "cpu_mb_compatibility",
    "mb": "motherboard",
    "mainboard": "motherboard",
    "case": "case_chassis",
    "case_product": "case_chassis",
    "cooler": "cpu_cooler",
    "psu": "power_supply",
    "ram": "memory",
    "gpu_compatibility": "gpu_case_compatibility",
    "cpu_compatibility": "cpu_mb_compatibility",
    "gpu_mb_compatibility": "mb_gpu_compatibility"
}

# 스키마 정보를 문자열로 변환
schema_str = "Database Schema:\n"
for table, columns in db_schema.items():
    schema_str += f"Table: {table}\n"
    schema_str += f"Columns: {', '.join(columns)}\n\n"

# 1. 질문 분석 노드
def analyze_question(state: PCCompatibilityState) -> PCCompatibilityState:
    """사용자 질문을 분석하여 관련된 부품 유형과 키워드 추출"""
    question = state["question"]
    
    # 모델 키워드 추출 프롬프트
    extract_prompt = """
    You are an expert in PC hardware compatibility analysis. Extract key information from the user's question.
    
    User Question: {question}
    
    Analyze the question and extract:
    1. The specific PC part models mentioned (e.g., RTX 4070, Ryzen 7800X3D)
    2. The types of PC parts involved in the question (e.g., GPU, CPU, motherboard, case)
    
    Return your analysis as a JSON object with these fields:
    - search_keywords: List of model keywords for search (most specific to least specific)
    - part_types: List of PC part types mentioned in the question
    
    Example formats for search keywords:
    - For GPU: ["RTX 4070", "4070", "RTX"]
    - For CPU: ["Ryzen 7800X3D", "7800X3D", "Ryzen"]
    
    IMPORTANT: Use English for JSON field names, but you can use Korean or English for the values.
    
    JSON response:
    """
    
    # 키워드 및 부품 유형 추출
    result = llm.invoke(extract_prompt.format(question=question))
    
    # JSON 파싱
    try:
        parsed = json.loads(result)
        search_keywords = parsed.get("search_keywords", [])
        part_types = parsed.get("part_types", [])
    except:
        # 파싱 실패 시 기본값
        search_keywords = []
        part_types = []
        
        # 간단한 정규표현식으로 키워드 추출 시도
        gpu_match = re.search(r'(RTX|GTX|RX)\s*(\d{3,4})(\s*Ti)?', question, re.IGNORECASE)
        if gpu_match:
            full_match = gpu_match.group(0)
            search_keywords.append(full_match)
            search_keywords.append(gpu_match.group(2))  # 숫자만
            if gpu_match.group(1):  # 접두사(RTX, GTX, RX)
                search_keywords.append(gpu_match.group(1))
            part_types.append("gpu")
        
        # 메인보드 언급 확인
        if re.search(r'(메인보드|마더보드|motherboard)', question, re.IGNORECASE):
            part_types.append("motherboard")
        
        # 케이스 언급 확인
        if re.search(r'(케이스|case)', question, re.IGNORECASE):
            part_types.append("case")
    
    # 상태 업데이트
    state["search_keywords"] = search_keywords
    state["part_types"] = part_types
    
    return state

# 2. 쿼리 생성 노드
def generate_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """부품 유형 및 키워드에 기반한 SQL 쿼리 생성"""
    question = state["question"]
    search_keywords = state["search_keywords"]
    part_types = state["part_types"]
    
    # 키워드가 없는 경우 기본 처리
    if not search_keywords:
        state["errors"].append("No search keywords found in the question.")
        return state
    
    # 가장 구체적인 키워드 사용
    keyword = search_keywords[0]
    
    # 관계 매핑
    relations = []
    
    # GPU와 메인보드 호환성
    if "gpu" in part_types and "motherboard" in part_types:
        relations.append("gpu_motherboard")
    
    # GPU와 케이스 호환성
    if "gpu" in part_types and "case" in part_types:
        relations.append("gpu_case")
    
    # CPU와 메인보드 호환성
    if "cpu" in part_types and "motherboard" in part_types:
        relations.append("cpu_motherboard")
    
    # 메인보드와 케이스 호환성
    if "motherboard" in part_types and "case" in part_types:
        relations.append("motherboard_case")
    
    # 부품 타입이 있지만 관계가 없는 경우, 하나만 언급된 경우
    if part_types and not relations:
        if "gpu" in part_types:
            if "motherboard" not in part_types:
                relations.append("gpu_motherboard")
            if "case" not in part_types:
                relations.append("gpu_case")
        elif "cpu" in part_types:
            if "motherboard" not in part_types:
                relations.append("cpu_motherboard")
        elif "motherboard" in part_types:
            if "case" not in part_types:
                relations.append("motherboard_case")
    
    # 관계가 여전히 없는 경우, 키워드를 기반으로 추정
    if not relations:
        # GPU 키워드 확인
        if any(kw in keyword.upper() for kw in ["RTX", "GTX", "RX", "RADEON", "GEFORCE"]):
            relations.append("gpu_motherboard")
            relations.append("gpu_case")
        # CPU 키워드 확인
        elif any(kw in keyword.upper() for kw in ["RYZEN", "INTEL", "CORE", "I7", "I9", "5800X", "7800X"]):
            relations.append("cpu_motherboard")
    
    # 쿼리 생성
    queries = {}
    
    # GPU와 메인보드 호환성 쿼리
    if "gpu_motherboard" in relations:
        queries["gpu_motherboard"] = f"""
        SELECT DISTINCT 
            g.model_name AS gpu_model, 
            m.model_name AS motherboard_model,
            m.manufacturer AS motherboard_manufacturer
        FROM gpu g
        JOIN mb_gpu_compatibility mgc ON g.gpu_id = mgc.gpu_id
        JOIN motherboard m ON mgc.mb_id = m.mb_id
        WHERE g.model_name LIKE '%{keyword}%'
        """
    
    # GPU와 케이스 호환성 쿼리
    if "gpu_case" in relations:
        queries["gpu_case"] = f"""
        SELECT DISTINCT 
            g.model_name AS gpu_model, 
            c.model_name AS case_model,
            c.product_name AS case_product_name
        FROM gpu g
        JOIN gpu_case_compatibility gcc ON g.gpu_id = gcc.gpu_id
        JOIN case_chassis c ON gcc.case_id = c.case_id
        WHERE g.model_name LIKE '%{keyword}%'
        """
    
    # CPU와 메인보드 호환성 쿼리
    if "cpu_motherboard" in relations:
        queries["cpu_motherboard"] = f"""
        SELECT DISTINCT 
            c.model_name AS cpu_model, 
            m.model_name AS motherboard_model,
            m.manufacturer AS motherboard_manufacturer
        FROM cpu c
        JOIN cpu_mb_compatibility cmc ON c.cpu_id = cmc.cpu_id
        JOIN motherboard m ON cmc.mb_id = m.mb_id
        WHERE c.model_name LIKE '%{keyword}%'
        """
    
    # 메인보드와 케이스 호환성 쿼리
    if "motherboard_case" in relations:
        queries["motherboard_case"] = f"""
        SELECT DISTINCT 
            m.model_name AS motherboard_model, 
            c.model_name AS case_model,
            c.product_name AS case_product_name
        FROM motherboard m
        JOIN mb_case_compatibility mcc ON m.mb_id = mcc.mb_id
        JOIN case_chassis c ON mcc.case_id = c.case_id
        WHERE m.model_name LIKE '%{keyword}%'
        """
    
    # 쿼리가 없는 경우
    if not queries:
        state["errors"].append("Could not determine appropriate compatibility queries for the question.")
    else:
        state["queries"] = queries
    
    return state

# 3. 쿼리 실행 노드
def execute_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """생성된 쿼리 실행 및 결과 수집"""
    queries = state["queries"]
    results = {}
    errors = []
    
    # 각 쿼리 실행
    for relation, query in queries.items():
        try:
            # 테이블 이름 검증 및 수정
            for original, corrected in table_mapping.items():
                # 테이블 이름 교체 (단어 경계 고려)
                query = re.sub(r'\b' + original + r'\b', corrected, query)
            
            # 쿼리 실행
            result = conn.execute(query).fetchdf()
            
            # 모델명에서 #숫자 패턴 제거
            for col in result.columns:
                if 'model' in col.lower():
                    result[col] = result[col].apply(lambda x: re.sub(r'#\d+\s*$', '', str(x)) if pd.notna(x) else x)
            
            # 중복 제거
            result = result.drop_duplicates()
            
            # 결과가 있는 경우만 저장
            if not result.empty:
                results[relation] = result.to_dict(orient="records")
            else:
                # 키워드가 더 있으면 더 일반적인 키워드로 시도
                if len(state["search_keywords"]) > 1:
                    for alt_keyword in state["search_keywords"][1:]:
                        alt_query = query.replace(f"LIKE '%{state['search_keywords'][0]}%'", f"LIKE '%{alt_keyword}%'")
                        alt_result = conn.execute(alt_query).fetchdf()
                        
                        # 모델명에서 #숫자 패턴 제거
                        for col in alt_result.columns:
                            if 'model' in col.lower():
                                alt_result[col] = alt_result[col].apply(lambda x: re.sub(r'#\d+\s*$', '', str(x)) if pd.notna(x) else x)
                        
                        # 중복 제거
                        alt_result = alt_result.drop_duplicates()
                        
                        if not alt_result.empty:
                            results[relation] = alt_result.to_dict(orient="records")
                            break
                
                if relation not in results:
                    product_name = state["search_keywords"][0] if state["search_keywords"] else "검색한 제품"
                    errors.append(f"데이터베이스에서 '{product_name}' 제품을 찾을 수 없습니다. {relation} 호환성 정보가 존재하지 않습니다.")
        except Exception as e:
            error_msg = f"Error executing query for {relation}: {str(e)}"
            errors.append(error_msg)
    
    # 상태 업데이트
    state["results"] = results
    state["errors"].extend(errors)
    
    return state

# 4. 결과 설명 생성 노드
def generate_explanation(state: PCCompatibilityState) -> PCCompatibilityState:
    """결과 통합 및 설명 생성"""
    question = state["question"]
    results = state["results"]
    errors = state["errors"]
    search_keywords = state["search_keywords"]
    part_types = state["part_types"]
    
    # 결과 요약
    summary = {}
    
    # 관계별 결과 요약
    for relation, result in results.items():
        if relation == "gpu_motherboard":
            summary["compatible_motherboards"] = [item["motherboard_model"] for item in result]
        elif relation == "gpu_case":
            summary["compatible_cases"] = [item["case_model"] for item in result]
        elif relation == "cpu_motherboard":
            summary["compatible_motherboards"] = [item["motherboard_model"] for item in result]
        elif relation == "motherboard_case":
            summary["compatible_cases"] = [item["case_model"] for item in result]
    
    # 설명 생성 프롬프트 - 한국어 응답 강조 및 하드웨어 호환성 원칙 명확화
    explanation_prompt = """
    당신은 PC 하드웨어 호환성 전문가입니다. 사용자에게 호환성 결과를 설명해 주세요.
    
    사용자 질문: {question}
    검색한 제품: {search_keywords}
    부품 유형: {part_types}
    
    호환성 결과:
    {results}
    
    오류 (있는 경우):
    {errors}
    
    다음 호환성 원칙을 반드시 지켜서 설명해주세요:
    1. CPU는 메인보드 소켓 타입(AM4, AM5, LGA1700 등)과 호환돼야 합니다.
       - 예: AMD Ryzen 5000 시리즈(5600X 등)는 AM4 소켓 필요
       - 예: AMD Ryzen 7000 시리즈는 AM5 소켓 필요
       - 예: 인텔 12/13세대는 LGA1700 소켓 필요
    
    2. GPU는 소켓과 관계없이 PCIe 슬롯과 호환됩니다. 
       - 현대 GPU는 대부분 메인보드의 PCIe 슬롯에 호환됨
       - GPU 호환성은 물리적 크기, 전력 요구사항이 중요함
    
    3. 케이스 호환성은 주로 물리적 크기(길이, 높이)에 관한 것입니다.
       - GPU 길이가 케이스 내부 공간에 맞는지 확인 필요
       - 메인보드 폼팩터(ATX, mATX, ITX 등)가 케이스와 호환되는지 확인 필요
    
    위 원칙에 따라 호환성 결과를 자세히 설명해주세요.
    가능한 경우 특정 호환 모델을 언급하세요.
    
    중요: 
    1. 결과가 없는 경우, 반드시 해당 제품이 데이터베이스에 존재하지 않는다는 점을 명확히 알려주세요.
    2. 반드시 한국어로 응답해 주세요.
    
    설명:
    """
    
    # 결과를 문자열로 변환
    results_str = ""
    for relation, result in results.items():
        relation_kr = relation.replace("gpu_motherboard", "GPU와 메인보드")
        relation_kr = relation_kr.replace("gpu_case", "GPU와 케이스")
        relation_kr = relation_kr.replace("cpu_motherboard", "CPU와 메인보드")
        relation_kr = relation_kr.replace("motherboard_case", "메인보드와 케이스")
        
        results_str += f"{relation_kr} 호환성:\n"
        if result:
            for item in result[:10]:  # 최대 10개만 표시
                results_str += f"- {', '.join([f'{k}: {v}' for k, v in item.items()])}\n"
            if len(result) > 10:
                results_str += f"... 그리고 {len(result) - 10}개 더 있음\n"
        else:
            results_str += "호환되는 부품을 찾을 수 없습니다.\n"
        results_str += "\n"
    
    # 오류를 문자열로 변환
    errors_str = "\n".join(errors) if errors else "오류 없음."
    
    # 설명 생성 - 한국어 응답 강조
    explanation = llm.invoke(explanation_prompt.format(
        question=question,
        search_keywords=search_keywords,
        part_types=part_types,
        results=results_str,
        errors=errors_str
    ))
    
    # 영어로 응답한 경우를 대비한 추가 요청
    if not any(char in explanation for char in '가나다라마바사아자차카타파하'):
        explanation = llm.invoke(
            explanation_prompt.format(
                question=question,
                search_keywords=search_keywords,
                part_types=part_types,
                results=results_str,
                errors=errors_str
            ) + "\n\n반드시 한국어로 응답해 주세요. DO NOT RESPOND IN ENGLISH, RESPOND IN KOREAN ONLY."
        )
    
    # 최종 결과 생성
    final_result = {
        "summary": summary,
        "detailed_results": results,
        "explanation": explanation,
        "errors": errors
    }
    
    # 상태 업데이트
    state["final_result"] = final_result
    
    return state

# 상태 흐름 그래프 구축
def build_graph():
    # 그래프 생성
    graph = StateGraph(PCCompatibilityState)
    
    # 노드 추가
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("execute_queries", execute_queries)
    graph.add_node("generate_explanation", generate_explanation)
    
    # 엣지 추가 (노드 간 연결)
    graph.add_edge("analyze_question", "generate_queries")
    graph.add_edge("generate_queries", "execute_queries")
    graph.add_edge("execute_queries", "generate_explanation")
    graph.add_edge("generate_explanation", END)
    
    # 시작 노드 설정
    graph.set_entry_point("analyze_question")
    
    # 컴파일 및 반환
    return graph.compile()

# 그래프 구축
pc_compatibility_graph = build_graph()

# Function calling을 위한 함수 정의
def process_pc_compatibility_query(query: str) -> dict:
    """
    PC 부품 호환성에 관한 사용자 질문을 처리합니다.
    
    Args:
        query: 사용자 질문 (예: "라이젠 7800X3D와 호환되는 메인보드 찾아줘")
        
    Returns:
        dict: 처리 결과를 담은 사전
    """
    # 초기 상태 설정
    initial_state = PCCompatibilityState(
        question=query,
        search_keywords=[],
        part_types=[],
        queries={},
        results={},
        errors=[],
        final_result=None
    )
    
    # 그래프 실행
    final_state = pc_compatibility_graph.invoke(initial_state)
    
    # 최종 결과 반환
    if final_state["final_result"]:
        return final_state["final_result"]
    else:
        return {
            "explanation": "질문 처리에 실패했습니다. 다른 질문으로 다시 시도해 주세요.",
            "errors": final_state["errors"]
        }


# 테스트 질문
test_query = "RX 7900이랑 호환되는 메인보드랑 케이스 알려줘 CPU는 5600x 사용하고싶어"

# 함수 호출
result = process_pc_compatibility_query(test_query)

# 결과 출력
print("\nExplanation:")
print(result["explanation"])