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
import os
from logging_config import get_logger
import time
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field

# TerminalLogCapture 클래스 - 터미널 로그를 캡처하는 핵심 클래스
class TerminalLogCapture:
    def __init__(self):
        self.logs = []
    
    def capture(self, message):
        """로그 메시지 캡처 및 터미널에 출력"""
        self.logs.append(message)
        print(message)  # 터미널에도 출력 - 개발 디버깅 용도
    
    def get_logs(self):
        """현재까지 캡처된 모든 로그 반환"""
        return self.logs
    
    def clear(self):
        """로그 초기화 - 새 질문이 들어올 때마다 초기화"""
        self.logs = []

# 로그 캡처 인스턴스 생성 - 이 인스턴스가 모든 로그를 캡처합니다
terminal_logger = TerminalLogCapture()

# Ollama 모델 초기화 - 온도 추가
llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1  # 낮은 온도로 더 일관된 응답 유도
)
DB_PATH = '/home/wlsdud022/AgentFactory/cs_agent/db/pc_parts.db'

# 상태 정의
class PCCompatibilityState(TypedDict):
    question: str
    search_keywords: List[str]
    part_types: List[str]
    queries: Dict[str, str]
    optimized_queries: Dict[str, List[str]]
    results: Dict[str, List[Dict[str, Any]]] # 이 필드는 execute_queries에서 채워짐
    errors: List[str]
    final_result: Optional[Dict[str, Any]]
    analysis_logs: List[str]
    query_logs: List[str]  # 쿼리 로그 필드 추가
    has_program_requirements: bool  # 게임 요구사항 플래그
    components: List[str]  # 질문에서 추출된 컴포넌트 목록
    program_requirements: str  # 프로그램 요구사항
    query_type: str  # 질문 유형
    existing_parts: Dict[str, str]  # 기존 부품 정보

# 데이터베이스 샘플 가져오기
def get_db_samples():
    samples = {}
        # 함수 내에서 연결 생성
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        # CPU 샘플
        samples['cpu'] = conn.execute("SELECT model_name FROM cpu LIMIT 5").fetchall()
        # GPU 샘플
        samples['gpu'] = conn.execute("SELECT model_name FROM gpu LIMIT 5").fetchall()
        # 마더보드 샘플
        samples['motherboard'] = conn.execute("SELECT model_name FROM motherboard LIMIT 5").fetchall()
        
        # 리스트로 변환
        for key in samples:
            samples[key] = [item[0] for item in samples[key]]

# 데이터베이스 샘플 가져오기
db_samples = get_db_samples()

# 데이터베이스 스키마 정보 가져오기
def get_db_schema():
    schema_info = {}
    
    # 함수 내에서 연결 생성
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        
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
    "gpu_mb_compatibility": "mb_gpu_compatibility",
    "memory_mb_compatibility": "mb_memory_compatibility",
    "cpu_cooler_compatibility": "cpu_cooler_compatibility",
    "psu_compatibility": "psu_case_compatibility",
    "storage_compatibility": "mb_storage_compatibility"
}

# 스키마 정보를 문자열로 변환
schema_str = "Database Schema:\n"
for table, columns in db_schema.items():
    schema_str += f"Table: {table}\n"
    schema_str += f"Columns: {', '.join(columns)}\n\n"

# 모듈별 로거 가져오기
logger = get_logger("PCCheckAgent")

# JSON 출력을 위한 Pydantic 모델 정의
class PCSpecsOutput(BaseModel):
    min_specs: Dict[str, str] = Field(description="게임/프로그램의 최소 사양 정보")
    recommended_specs: Dict[str, str] = Field(description="게임/프로그램의 권장 사양 정보")
    queries: Dict[str, str] = Field(description="각 부품 테이블에 대한 SQL 쿼리")
    compatibility_notes: Optional[Dict[str, str]] = Field(default=None, description="부품 간 호환성 고려사항")

# 질문 분석 프롬프트 정의 - 기존 부품 및 질문 의도 추출 추가
question_analysis_prompt = PromptTemplate.from_template("""
당신은 PC 부품 호환성을 분석하는 AI 전문가입니다. 다음 질문을 분석하여 사용자의 의도와 언급된 PC 부품 정보를 추출하세요.

질문: {question}

분석할 내용:
1. 질문 유형: 다음 중 하나를 선택하세요
   - "호환성 확인": 기존 부품들의 호환성만 확인하는 질문
   - "기존 부품 호환 PC 구성 추천": 이미 가지고 있는 부품과 호환되는 나머지 부품 추천
   - "새로운 PC 구성 추천": 완전히 새로운 PC 구성 추천
   - "게임 PC 구성 추천": 특정 게임 또는 용도에 맞는 PC 구성 추천
   - "프로그램 요구사항 분석": 특정 프로그램의 권장 사양 문의

2. 기존 보유 부품: 사용자가 이미 가지고 있거나 사용 중인 부품을 추출하세요
   예: "5600X CPU를 사용하고 있어요", "RTX 3080 그래픽카드 보유 중" 등

3. 다음 PC 부품 유형이 질문에 언급되었는지 분석하세요:
   - cpu: CPU 또는 프로세서
   - motherboard: 메인보드 또는 마더보드
   - gpu: GPU, 그래픽카드, 비디오카드
   - memory: RAM, 메모리
   - storage: SSD, HDD, 저장장치
   - power_supply: 파워서플라이, PSU, 전원공급장치
   - case_chassis: 케이스, 샤시
   - cpu_cooler: CPU 쿨러, 냉각기

4. 검색에 사용할 수 있는 구체적인 모델명이나 키워드를 추출해주세요.
   예: "RTX 3080", "5600X", "B550" 등

JSON 형식으로 답변해주세요:
```json
{
  "question_type": "질문 유형",
  "existing_parts": {
    "cpu": "CPU 모델명 또는 null",
    "gpu": "GPU 모델명 또는 null",
    "motherboard": "메인보드 모델명 또는 null",
    "기타 언급된 부품": "모델명 또는 null"
  },
  "part_types": ["언급된 부품 유형 목록"],
  "search_keywords": ["검색 키워드 목록"]
}
```
""")

# 1. 질문 분석 노드
def analyze_question(state: PCCompatibilityState) -> PCCompatibilityState:
    """질문을 분석하여 부품 유형, 검색 키워드, 질문 유형 및 기존 부품 정보 추출"""
    print("====================== ANALYZE QUESTION START ======================")
    start_time = time.time()
    question = state["question"]
    
    try:
        # LLM에 분석 요청
        result = llm.invoke(
            question_analysis_prompt.format(question=question)
        )
        
        # 결과 파싱
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            
            # 부품 유형 및 검색 키워드 적용
            state["part_types"] = data.get("part_types", [])
            state["search_keywords"] = data.get("search_keywords", [])
            
            # 질문 유형 및 기존 부품 정보 추출 (새로운 부분)
            question_type = data.get("question_type", "호환성 확인")
            
            # 질문 유형에 따라 state 업데이트
            if "게임" in question_type:
                state["query_type"] = "game_pc_recommendation"
                state["has_program_requirements"] = True
                terminal_logger.capture("로그 추가: 🎮 게임 PC 추천 모드 활성화 (LLM 판단)")
            elif "기존 부품 호환" in question_type:
                state["query_type"] = "pc_compatibility"
                terminal_logger.capture("로그 추가: 🔄 기존 부품 호환 PC 구성 추천 모드 활성화 (LLM 판단)")
            elif "프로그램 요구사항" in question_type:
                state["query_type"] = "program_requirements"
                terminal_logger.capture("로그 추가: 📊 프로그램 요구사항 분석 모드 활성화 (LLM 판단)")
            else:
                state["query_type"] = "pc_compatibility"
                
            # 기존 부품 정보 저장
            state["existing_parts"] = data.get("existing_parts", {})
            
            # 기존 부품이 있는 경우 로그에 기록
            if any(state["existing_parts"].values()):
                parts_list = [f"{part}: {model}" for part, model in state["existing_parts"].items() if model]
                terminal_logger.capture(f"로그 추가: 🔍 기존 부품 감지: {', '.join(parts_list)}")
                
            # 분석 결과 로그에 추가
            terminal_logger.capture(f"로그 추가: 📝 질문 유형: {question_type}")
            terminal_logger.capture(f"로그 추가: 🔎 검색 키워드: {', '.join(state['search_keywords'])}")
            
        else:
            # JSON을 찾지 못한 경우 기본값 설정
            state["part_types"] = ["cpu", "gpu", "motherboard"]
            state["search_keywords"] = []
            state["errors"].append("질문 분석에서 JSON 결과를 추출하지 못했습니다.")
    except Exception as e:
        # 예외 처리
        state["errors"].append(f"질문 분석 오류: {str(e)}")
        
    # 최소한의 키워드와 부품 유형이 없는 경우 기본값 사용
    if not state["search_keywords"]:
        terminal_logger.capture("로그 추가: ⚠️ 검색 키워드를 추출하지 못했습니다. 기본 키워드 사용.")
        # 기본 키워드 설정은 유지
        
    print(f"Keywords: {state['search_keywords']}")
    print(f"Part types: {state['part_types']}")
    print(f"Query type: {state.get('query_type', 'pc_compatibility')}")
    print(f"Existing parts: {state.get('existing_parts', {})}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("====================== ANALYZE QUESTION END ======================")
    
    return state

# 2. 쿼리 생성 노드
def generate_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """웹 검색 결과에서 추출한 권장사양을 기반으로 SQL 쿼리 생성 - 자가 진단 및 수정 기능 포함"""
    print("====================== GENERATE QUERIES START ======================")
    start_time = time.time()
    
    # 권장사양 먼저 분석 (아직 안 했으면)
    if not state.get("min_specs") and not state.get("recommended_specs"):
        state = analyze_requirements(state)
    
    # 권장 사양과 최소 사양
    min_specs = state.get("min_specs", {})
    recommended_specs = state.get("recommended_specs", {})
    
    # 데이터베이스 스키마 정보 가져오기
    tables_info = {}
    try:
        # 각 테이블의 컬럼 정보 가져오기
        print("DB 스키마 확인 중...")
        conn = duckdb.connect(DB_PATH)
        main_tables = ["cpu", "gpu", "motherboard", "memory", "power_supply", "case_chassis", "storage"]
        
        for table in main_tables:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            tables_info[table] = columns
            print(f"테이블 {table} 컬럼: {columns}")
        
        conn.close()
    except Exception as e:
        error_msg = f"DB 스키마 가져오기 오류: {str(e)}"
        state["errors"].append(error_msg)
        print(error_msg)
        return state
    
    # LLM을 사용하여 웹 검색 결과에서 추출한 정보로 검색 쿼리 생성
    program_requirements = state.get("program_requirements", "")
    
    # LLM 프롬프트 - 웹 검색 결과와 최소/권장 사양을 바탕으로 모델명 기반 쿼리 생성
    prompt = f"""
    당신은 컴퓨터 하드웨어 전문가입니다. 다음 게임 사양 정보를 분석하여 해당 게임을 원활하게 실행할 수 있는 
    PC 부품을 데이터베이스에서 검색하기 위한 SQL 쿼리 조건을 생성해주세요.

    # 게임 정보 및 사양
    {program_requirements}

    # 추출된 최소 사양
    {json.dumps(min_specs, ensure_ascii=False, indent=2)}

    # 추출된 권장 사양
    {json.dumps(recommended_specs, ensure_ascii=False, indent=2)}

    # 테이블 스키마 정보
    {json.dumps(tables_info, ensure_ascii=False, indent=2)}

    위 정보를 분석하여 다음 부품 테이블에 대한 SQL WHERE 조건을 생성해주세요.
    각 조건은 모델명 기반의 검색을 우선하고, 이후 성능 지표를 기준으로 검색하도록 해주세요.
    
    JSON 형식으로 응답해주세요:
    
    ```json
    {
      "cpu": "WHERE 절에 들어갈 조건 (권장 CPU 또는 동급 이상의 CPU를 찾기 위한 조건)",
      "gpu": "WHERE 절에 들어갈 조건 (권장 GPU 또는 동급 이상의 GPU를 찾기 위한 조건)",
      "memory": "WHERE 절에 들어갈 조건 (권장 메모리 용량/속도 이상의 메모리를 찾기 위한 조건)",
      "storage": "WHERE 절에 들어갈 조건 (권장 저장장치 이상의 저장장치를 찾기 위한 조건)",
      "motherboard": "WHERE 절에 들어갈 조건 (CPU와 호환되는 메인보드 조건)",
      "power_supply": "WHERE 절에 들어갈 조건 (시스템에 적합한 파워 서플라이 조건)",
      "case_chassis": "WHERE 절에 들어갈 조건 (기본적인 케이스 조건)"
    }
    ```
    
    중요: 
    1. 정확한 모델명으로 검색하되, 해당 모델이 없을 경우 성능 지표로 검색할 수 있도록 OR 조건을 사용하세요.
    2. 최신 유사 모델도 포함될 수 있도록 LIKE 연산자를 활용하세요. (예: model_name LIKE '%GTX 1060%' OR model_name LIKE '%RTX 2060%')
    3. 메모리의 경우 capacity 또는 memory_capacity 필드를 모두 고려하세요.
    4. WHERE 절만 작성하세요 (SELECT * FROM table은 제외).
    """
    
    try:
        # LLM 호출
        result = llm.invoke(prompt)
        
        # JSON 추출
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = result
        
        # 변환 시도
        try:
            where_conditions = json.loads(json_str)
        except Exception as e:
            print(f"JSON 파싱 오류: {str(e)}")
            where_conditions = {}
        
        # 쿼리 생성
        queries = {}
        
        # 각 테이블에 대한 쿼리 생성
        for table in main_tables:
            if table in where_conditions and where_conditions[table]:
                queries[table] = f"SELECT * FROM {table} WHERE {where_conditions[table]} LIMIT 15"
            else:
                # 테이블별 기본 쿼리
                if table == "cpu":
                    queries[table] = "SELECT * FROM cpu WHERE cores >= 4 ORDER BY cores DESC, threads DESC LIMIT 15"
                elif table == "gpu":
                    queries[table] = "SELECT * FROM gpu WHERE memory_capacity >= 4 ORDER BY memory_capacity DESC LIMIT 15"
                elif table == "memory":
                    queries[table] = "SELECT * FROM memory WHERE capacity >= 8 OR memory_capacity LIKE '%8%' ORDER BY capacity DESC, clock DESC LIMIT 15"
                elif table == "storage":
                    queries[table] = "SELECT * FROM storage WHERE capacity >= 250 LIMIT 15"
                elif table == "motherboard":
                    queries[table] = "SELECT * FROM motherboard LIMIT 15"
                elif table == "power_supply":
                    queries[table] = "SELECT * FROM power_supply WHERE wattage >= 500 ORDER BY wattage ASC LIMIT 15"
                elif table == "case_chassis":
                    queries[table] = "SELECT * FROM case_chassis LIMIT 15"
        
        # 쿼리 저장
        state["queries"] = queries
        
    except Exception as e:
        error_msg = f"LLM 쿼리 생성 오류: {str(e)}"
        state["errors"].append(error_msg)
        print(error_msg)
        
        # 기본 쿼리 초기화 대신 빈 쿼리 맵 생성 - 자가 진단 과정에서 채워짐
        state["queries"] = {}
    
    # 자가 진단 및 쿼리 수정 프로세스
    if not state.get("queries") or len(state.get("queries", {})) < len(main_tables):
        print("쿼리 자가 진단 프로세스 시작...")
        state = self_diagnose_and_fix_queries(state, tables_info, main_tables)
    
    # 쿼리 결과 확인
    query_count = len(state.get("queries", {}))
    print(f"생성된 쿼리 수: {query_count}")
    print(f"생성된 쿼리 목록: {list(state.get('queries', {}).keys())}")
    
    # 실행 시간 기록
    execution_time = time.time() - start_time
    print(f"쿼리 생성 시간: {execution_time:.2f}초")
    print("====================== GENERATE QUERIES END ======================")
    
    return state

def self_diagnose_and_fix_queries(state: PCCompatibilityState, tables_info, tables_list) -> PCCompatibilityState:
    """쿼리 오류를 자가 진단하고 수정하는 함수"""
    print("====================== SELF DIAGNOSE QUERIES START ======================")
    
    conn = None
    try:
        conn = duckdb.connect(DB_PATH)
        cursor = conn.cursor()
        
        existing_queries = state.get("queries", {})
        problematic_tables = []
        
        # 먼저 기본 쿼리로 각 테이블을 탐색해보고 결과가 있는지 확인
        for table in tables_list:
            if table not in existing_queries:
                probe_query = f"SELECT * FROM {table} LIMIT 5"
                try:
                    cursor.execute(probe_query)
                    sample_results = cursor.fetchall()
                    print(f"테이블 {table} 탐색 쿼리 성공: {len(sample_results)}개 결과")
                    
                    # 샘플 데이터 추출
                    sample_data = []
                    for row in sample_results:
                        row_dict = {}
                        for i, col in enumerate(cursor.description):
                            col_name = col[0]
                            row_dict[col_name] = row[i]
                        sample_data.append(row_dict)
                    
                    # 테이블 추가 정보 (중요 컬럼과 예시 값)
                    important_columns = []
                    for col in tables_info.get(table, []):
                        if col in ["model_name", "manufacturer", "capacity", "memory_capacity", "cores", "threads",
                                  "socket_type", "memory_type", "wattage", "clock", "memory_clock"]:
                            if sample_data and len(sample_data) > 0 and col in sample_data[0]:
                                important_columns.append(f"{col}: {sample_data[0].get(col)}")
                    
                    problematic_tables.append({
                        "table": table,
                        "sample_data": sample_data[:2],  # 최대 2개 샘플만 전달
                        "important_columns": important_columns
                    })
                except Exception as e:
                    print(f"테이블 {table} 탐색 쿼리 오류: {str(e)}")
                    problematic_tables.append({
                        "table": table, 
                        "error": str(e),
                        "schema": tables_info.get(table, [])
                    })
        
        # 문제가 있는 테이블이 있으면 LLM에게 쿼리 수정 요청
        if problematic_tables:
            prompt = f"""
            당신은 SQL 전문가입니다. 다음 테이블들에 대한 쿼리를 생성하는 데 문제가 발생했습니다.
            각 테이블의 샘플 데이터와 스키마를 분석하여 게임 '{state.get('question', '게임')}' 실행에 적합한 PC 부품을 찾기 위한 쿼리를 수정해주세요.
            
            # 게임 최소 사양
            {json.dumps(state.get('min_specs', {}), ensure_ascii=False, indent=2)}
            
            # 게임 권장 사양
            {json.dumps(state.get('recommended_specs', {}), ensure_ascii=False, indent=2)}
            
            # 문제가 있는 테이블 정보:
            {json.dumps(problematic_tables, ensure_ascii=False, indent=2)}
            
            각 테이블에 대해 아래 형식으로 응답해주세요:
            
            ```json
            {
              "테이블1": "SELECT * FROM 테이블1 WHERE 적절한_조건 LIMIT 15",
              "테이블2": "SELECT * FROM 테이블2 WHERE 적절한_조건 LIMIT 15",
              ...
            }
            ```
            
            다음 사항을 고려해주세요:
            1. 모델명 또는 주요 사양을 기준으로 검색하되, 결과가 나오지 않을 수 있으므로 조건을 너무 제한적으로 설정하지 마세요.
            2. 메모리의 경우 capacity 또는 memory_capacity 필드를 모두 고려하세요.
            3. WHERE 조건이 너무 복잡하면 간단하게 유지하되, 최소한의 필터링은 적용하세요.
            4. 샘플 데이터를 참고하여 실제 데이터베이스에 맞는 조건을 작성하세요.
            """
            
            try:
                # LLM에게 수정된 쿼리 요청
                fix_result = llm.invoke(prompt)
                
                # JSON 추출
                json_match = re.search(r'```json\s*(.*?)\s*```', fix_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = fix_result
                
                # 변환 시도
                try:
                    fixed_queries = json.loads(json_str)
                    
                    # 수정된 쿼리 테스트 및 적용
                    for table, query in fixed_queries.items():
                        try:
                            # 쿼리 실행해보기
                            cursor.execute(query)
                            test_results = cursor.fetchall()
                            result_count = len(test_results)
                            
                            # 결과가 있으면 쿼리 적용
                            if result_count > 0:
                                print(f"테이블 {table} 수정 쿼리 성공: {result_count}개 결과")
                                existing_queries[table] = query
                            else:
                                print(f"테이블 {table} 수정 쿼리 결과 없음: {query}")
                                # 결과가 없을 경우 더 간단한 쿼리로 다시 시도
                                fallback_query = f"SELECT * FROM {table} LIMIT 15"
                                cursor.execute(fallback_query)
                                if len(cursor.fetchall()) > 0:
                                    existing_queries[table] = fallback_query
                                    print(f"테이블 {table} 폴백 쿼리 적용: {fallback_query}")
                        except Exception as e:
                            print(f"테이블 {table} 수정 쿼리 실행 오류: {str(e)}")
                            # 오류 발생 시 가장 기본적인 쿼리 사용
                            existing_queries[table] = f"SELECT * FROM {table} LIMIT 15"
                            print(f"테이블 {table} 기본 쿼리 적용: {existing_queries[table]}")
                    
                except Exception as e:
                    print(f"수정된 쿼리 JSON 파싱 오류: {str(e)}")
                    # JSON 파싱 오류 시 각 테이블에 대한 기본 쿼리 생성
                    for table in problematic_tables:
                        table_name = table.get("table")
                        existing_queries[table_name] = f"SELECT * FROM {table_name} LIMIT 15"
                
            except Exception as e:
                print(f"LLM 쿼리 수정 오류: {str(e)}")
                # LLM 호출 실패 시 기본 쿼리 적용
                for table in problematic_tables:
                    table_name = table.get("table")
                    existing_queries[table_name] = f"SELECT * FROM {table_name} LIMIT 15"
        
        # 수정된 쿼리 목록 저장
        state["queries"] = existing_queries
        
    except Exception as e:
        print(f"자가 진단 프로세스 오류: {str(e)}")
        state["errors"].append(f"쿼리 자가 진단 오류: {str(e)}")
        
        # 오류 발생 시 모든 테이블에 대한 기본 쿼리 적용
        if not state.get("queries"):
            state["queries"] = {}
        
        for table in tables_list:
            if table not in state["queries"]:
                state["queries"][table] = f"SELECT * FROM {table} LIMIT 15"
    
    finally:
        if conn:
            conn.close()
    
    print("====================== SELF DIAGNOSE QUERIES END ======================")
    return state

def analyze_requirements(state: PCCompatibilityState) -> PCCompatibilityState:
    """웹 검색 결과(권장사양)를 분석하여 구체적인 부품 요구사항 추출"""
    program_requirements = state.get("program_requirements", "")
    if not program_requirements:
        state["errors"].append("권장사양 정보가 없습니다.")
        return state
    
    # LLM을 사용하여 권장사양에서 구체적인 부품 요구사항 추출
    prompt = f"""
    당신은 PC 하드웨어 전문가입니다. 다음 게임 권장사양 정보를 분석하여 필요한 최소/권장 하드웨어 요구사항을 추출해주세요.
    
    # 게임 권장사양 정보:
    {program_requirements}
    
    다음 정보를 추출하여 JSON 형식으로 반환해주세요:
    
    ```json
    {{
      "min_specs": {{
        "cpu": "정확한 CPU 모델명 또는 동급 사양",
        "gpu": "정확한 GPU 모델명 또는 동급 사양",
        "ram": "필요한 RAM 용량 및 타입",
        "storage": "필요한 저장장치 용량 및 타입"
      }},
      "recommended_specs": {{
        "cpu": "정확한 CPU 모델명 또는 동급 사양",
        "gpu": "정확한 GPU 모델명 또는 동급 사양",
        "ram": "필요한 RAM 용량 및 타입",
        "storage": "필요한 저장장치 용량 및 타입"
      }}
    }}
    ```
    
    가능한 정확한 모델명을 추출해주세요. 모델명이 없다면 동급 성능의 일반적인 모델을 제안해도 됩니다.
    """
    
    try:
        # LLM 호출하여 권장사양 분석
        result = llm.invoke(prompt)
        
        # JSON 파싱
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = result
        
        specs_data = json.loads(json_str)
        
        # 결과 저장
        state["min_specs"] = specs_data.get("min_specs", {})
        state["recommended_specs"] = specs_data.get("recommended_specs", {})
        
        terminal_logger.capture(f"로그 추가: 📊 최소 사양 파악: {json.dumps(state['min_specs'], ensure_ascii=False)}")
        terminal_logger.capture(f"로그 추가: 📊 권장 사양 파악: {json.dumps(state['recommended_specs'], ensure_ascii=False)}")
        
        return state
        
    except Exception as e:
        error_msg = f"권장사양 분석 오류: {str(e)}"
        state["errors"].append(error_msg)
        terminal_logger.capture(f"로그 추가: ❌ {error_msg}")
        
        # 기본 사양 설정
        state["min_specs"] = {
            "cpu": "Intel Core i5-6600K 또는 AMD Ryzen 5 1600",
            "gpu": "NVIDIA GTX 1060 3GB 또는 AMD RX 570 4GB",
            "ram": "8GB DDR4",
            "storage": "SSD 30GB"
        }
        state["recommended_specs"] = {
            "cpu": "Intel Core i7-8700K 또는 AMD Ryzen 7 2700X",
            "gpu": "NVIDIA RTX 2060 또는 AMD RX 5700",
            "ram": "16GB DDR4",
            "storage": "SSD 50GB"
        }
        
        return state

# 쿼리 최적화 함수 개선
def optimize_search_query(state: PCCompatibilityState) -> PCCompatibilityState:
    """검색 쿼리 최적화 - AI 모델 사용하지 않는 간단한 버전"""
    logger.info("Starting node: optimize_search_query")
    
    # 디버그용 직접 출력 추가
    print("====================== OPTIMIZE QUERIES START ======================")
    
    try:
        # 쿼리 가져오기
        queries = state.get("queries", {})
        
        # 쿼리가 없는 경우
        if not queries:
            print("⚠️ 최적화할 쿼리가 없습니다.")
            logger.warning("최적화할 쿼리가 없습니다.")
            state["errors"].append("최적화할 쿼리가 없습니다.")
            return state
        
        # 최적화된 쿼리 저장 딕셔너리
        optimized_queries = {}
        
        # 최적화 로직: 간단하게 각 쿼리를 그대로 리스트로 변환
        for table, query in queries.items():
            print(f"테이블 {table} 쿼리 최적화: {query[:100]}...")
            optimized_queries[table] = [query]
            
        # 저장
        state["optimized_queries"] = optimized_queries
        print(f"최적화된 쿼리 수: {len(optimized_queries)}")
        logger.info(f"최적화된 쿼리 수: {len(optimized_queries)}")
        
        print("====================== OPTIMIZE QUERIES END ======================")
        return state
    except Exception as e:
        error_msg = f"쿼리 최적화 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        state["errors"].append(error_msg)
        print("====================== OPTIMIZE QUERIES ERROR ======================")
        return state

# 3. SQL 쿼리 실행 노드
def execute_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """SQL 쿼리 실행 및 결과 처리 - 결과 없을 시 자동 대안 검색"""
    print("====================== EXECUTE QUERIES START ======================")
    start_time = time.time()
    
    terminal_logger.capture("로그 추가: 🔍 쿼리 실행 시작")
    
    # 쿼리 목록 가져오기
    queries = state.get("queries", {})
    print(f"쿼리 목록: {list(queries.keys())}")
    
    conn = None
    try:
        # DB 연결
        print(f"DB 연결 시도: {DB_PATH}")
        conn = duckdb.connect(DB_PATH)
        print(f"✅ DB 연결 성공: {DB_PATH}")
        
        # DB 테이블 목록 확인
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 DB 테이블 목록: {tables}")
        
        # 테이블별 스키마 확인 (디버깅용)
        for table_name in queries.keys():
            if table_name in tables:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [row[1] for row in cursor.fetchall()]
                print(f"{table_name.upper()} 테이블 컬럼: {columns}")
        
        # 쿼리 실행 결과 저장
        results = {}
        
        # 각 쿼리 실행
        for table, query in queries.items():
            print(f"\n===== 테이블 {table} 쿼리 실행 =====")
            
            # 쿼리를 리스트로 변환 (최적화된 쿼리 지원)
            query_list = []
            if isinstance(query, list):
                query_list = query
            else:
                query_list = [query]
            
            table_results = []
            query_index = 0
            
            # 각 쿼리 순차 실행
            for q in query_list:
                query_index += 1
                print(f"쿼리 #{query_index}: {q}")
                
                try:
                    # 쿼리 실행
                    q_start = time.time()
                    cursor.execute(q)
                    
                    # 결과 가져오기
                    rows = cursor.fetchall()
                    
                    # 결과를 딕셔너리로 변환
                    column_names = [column[0] for column in cursor.description]
                    
                    for row in rows:
                        result_dict = {}
                        for i, value in enumerate(row):
                            result_dict[column_names[i]] = value
                        table_results.append(result_dict)
                    
                    # 쿼리 실행 결과 기록
                    q_time = time.time() - q_start
                    print(f"✅ 결과 ({table}): {len(rows)}행 (실행시간: {q_time:.2f}초)")
                    
                    # 샘플 데이터 출력 (최대 3개)
                    if len(rows) > 0:
                        print(f"📋 샘플 데이터 ({table}, 최대 3개):")
                        for i, row_dict in enumerate(table_results[:3]):
                            print(f"{i+1}. {', '.join([f'{k}: {v}' for k, v in list(row_dict.items())[:5]])}...")
                    
                    # 충분한 결과가 있으면 다음 쿼리로 넘어가기
                    if len(table_results) >= 5:
                        break
                        
                except Exception as e:
                    error_msg = f"쿼리 실행 오류 ({table}): {str(e)}"
                    print(f"❌ {error_msg}")
                    state["errors"].append(error_msg)
            
            # 결과 저장
            results[table] = table_results
            
            # 결과가 없으면 대안 검색
            if len(table_results) == 0:
                print(f"⚠️ 테이블 {table}의 검색 결과가 없습니다. 대안 검색 시도...")
                
                # 대안 쿼리 생성 및 실행
                fallback_query = f"SELECT * FROM {table} LIMIT 15"
                try:
                    cursor.execute(fallback_query)
                    fallback_rows = cursor.fetchall()
                    
                    # 결과가 있으면 저장
                    if len(fallback_rows) > 0:
                        print(f"✅ 대안 쿼리로 {len(fallback_rows)}개 결과 찾음")
                        
                        # 결과를 딕셔너리로 변환
                        column_names = [column[0] for column in cursor.description]
                        fallback_results = []
                        
                        for row in fallback_rows:
                            result_dict = {}
                            for i, value in enumerate(row):
                                result_dict[column_names[i]] = value
                            fallback_results.append(result_dict)
                        
                        # 결과 저장
                        results[table] = fallback_results
                    else:
                        print(f"⚠️ 대안 쿼리로도 결과를 찾을 수 없습니다.")
                
                except Exception as e:
                    error_msg = f"대안 쿼리 실행 오류 ({table}): {str(e)}"
                    print(f"❌ {error_msg}")
                    state["errors"].append(error_msg)
    
    except Exception as e:
        error_msg = f"DB 연결 오류: {str(e)}"
        print(f"❌ {error_msg}")
        state["errors"].append(error_msg)
        
    finally:
        # DB 연결 종료
        if conn:
            conn.close()
            print("DB 연결 종료")
        
        # 결과 요약
        print("\n===== 쿼리 실행 결과 요약 =====")
        for table, table_results in results.items():
            print(f"테이블 {table}: {len(table_results)}개 결과")
        
        # 결과 저장
        state["results"] = results
        
        # 실행 시간 기록
        execution_time = time.time() - start_time
        print(f"쿼리 실행 시간: {execution_time:.2f}초")
        print("====================== EXECUTE QUERIES END ======================")
    
    return state

# 4. 결과 설명 생성 노드
def generate_explanation(state: PCCompatibilityState) -> PCCompatibilityState:
    """쿼리 결과를 바탕으로 설명 생성 - 웹 검색 결과 및 권장사양 명시"""
    print("====================== GENERATE EXPLANATION START ======================")
    start_time = time.time()
    
    # 결과 종합
    query_results = state.get("results", {})
    
    # 프로그램 요구사항 추출
    program_requirements = state.get("program_requirements", "")
    
    # 최소/권장 사양 추출
    min_specs = state.get("min_specs", {})
    recommended_specs = state.get("recommended_specs", {})
    
    # 각 부품 유형별 결과 확인 및 검색된 모델명 추출
    actual_products = {}
    error_messages = []
    
    for part_type in ["cpu", "gpu", "memory", "storage", "power_supply", "case_chassis", "motherboard"]:
        results = query_results.get(part_type, [])
        if results and len(results) > 0:
            actual_products[part_type] = []
            for item in results[:3]:  # 최대 3개 제품만 표시
                model_name = item.get("model_name", "")
                if model_name:
                    # 제조사와 모델명 추출
                    manufacturer = item.get("manufacturer", "")
                    if manufacturer and manufacturer in model_name:
                        model_info = model_name
                    else:
                        model_info = f"{manufacturer} {model_name}" if manufacturer else model_name
                    
                    # 추가 정보 수집
                    extra_info = ""
                    if part_type == "cpu":
                        cores = item.get("cores", "")
                        threads = item.get("threads", "")
                        socket = item.get("socket_type", "")
                        if cores and threads:
                            extra_info = f" ({cores}코어/{threads}스레드, 소켓:{socket})"
                    elif part_type == "gpu":
                        memory = item.get("memory_capacity", "")
                        if memory:
                            extra_info = f" ({memory}GB)"
                    elif part_type == "memory":
                        capacity = item.get("capacity", "")
                        memory_capacity = item.get("memory_capacity", "")
                        clock = item.get("clock", "")
                        if capacity:
                            extra_info = f" ({capacity}GB"
                            if clock:
                                extra_info += f"/{clock}MHz"
                            extra_info += ")"
                        elif memory_capacity:
                            extra_info = f" ({memory_capacity}"
                            if clock:
                                extra_info += f"/{clock}MHz"
                            extra_info += ")"
                    elif part_type == "storage":
                        capacity = item.get("capacity", "")
                        if capacity:
                            extra_info = f" ({capacity}GB)"
                    elif part_type == "power_supply":
                        wattage = item.get("wattage", "")
                        if wattage:
                            extra_info = f" ({wattage}W)"
                    
                    actual_products[part_type].append(model_info + extra_info)
        else:
            error_messages.append(f"⚠️ {part_type.upper()} 검색 결과가 없습니다. 더 일반적인 조건으로 검색이 필요합니다.")
    
    # 실제 검색된 제품 정보 구성
    product_info = "\n\n## 검색된 실제 제품 모델:\n"
    for part, products in actual_products.items():
        product_info += f"\n### {part.upper()}:\n"
        for product in products:
            product_info += f"- {product}\n"
    
    # 오류 메시지 추가
    if error_messages:
        product_info += "\n\n## 검색 오류:\n"
        for error in error_messages:
            product_info += f"- {error}\n"
        
        product_info += "\n⚠️ 일부 부품 검색 결과가 없으므로, 일반적인 추천 모델로 대체됩니다.\n"
    
    # 호환성 정보 추가
    compatibility_info = "\n\n## 호환성 정보:\n"
    
    # CPU와 메인보드 소켓 호환성
    if "cpu" in actual_products and len(actual_products["cpu"]) > 0 and len(query_results["cpu"]) > 0:
        first_cpu = query_results["cpu"][0]
        cpu_socket = first_cpu.get("socket_type", "")
        if cpu_socket:
            compatibility_info += f"\n- CPU 소켓 타입: {cpu_socket}\n"
            compatibility_info += f"  - 이 소켓과 호환되는 메인보드가 필요합니다.\n"
    
    # GPU 전력 요구사항
    if "gpu" in actual_products and len(actual_products["gpu"]) > 0 and len(query_results["gpu"]) > 0:
        first_gpu = query_results["gpu"][0]
        gpu_power = first_gpu.get("power_consumption", "")
        recommended_psu = first_gpu.get("recommended_psu", "")
        if gpu_power or recommended_psu:
            compatibility_info += f"\n- GPU 전력 요구사항: "
            if gpu_power:
                compatibility_info += f"소비전력 {gpu_power}W"
            if recommended_psu:
                compatibility_info += f", 권장 파워 {recommended_psu}W"
            compatibility_info += "\n"
    
    # 게임 사양 정보 요약
    game_name = state.get("question", "").replace("추천해줘", "").replace("PC 구성", "").strip()
    if not game_name:
        game_name = "게임"
        
    game_specs_info = f"""
## {game_name} 권장 사양 정보:

### 최소 사양:
{json.dumps(min_specs, ensure_ascii=False, indent=2)}

### 권장 사양:
{json.dumps(recommended_specs, ensure_ascii=False, indent=2)}

위 권장 사양을 기반으로 PC 구성을 추천합니다.
"""
    
    # 프롬프트 구성 - 웹 검색 결과, 권장사양, 실제 검색된 부품을 명확히 표시
    prompt = f"""
    당신은 PC 하드웨어 전문가입니다. 다음 정보를 바탕으로 게임용 PC 구성을 추천해 주세요.
    
    # 사용자 질문
    {state['question']}
    
    {game_specs_info}
    
    {product_info}
    
    {compatibility_info}
    
    위 정보를 바탕으로 다음 내용을 포함한 답변을 작성해주세요:
    
    1. 위에 명시된 게임 권장사양과 최소사양을 먼저 요약하여 제시
    2. 저가형, 중가형, 고가형 구성 추천
    3. 각 구성에 대한 구체적인 부품 목록 (실제 검색된 모델명 사용)
    4. 각 구성으로 게임 실행 시 예상 성능 (FPS, 설정 등)
    5. 호환성 고려사항 및 선택 이유 설명
    
    ## 중요: 
    1. 위에 제시된 실제 제품 모델명만 사용하여 구체적인 답변을 작성하세요!
    2. 답변은 한국어로 작성해주세요.
    3. 답변의 시작에 반드시 게임 권장사양과 최소사양을 먼저 언급하세요.
    """
    
    try:
        # LLM 호출
        result = llm.invoke(prompt)
        
        # 정규 표현식 검사 - 특정 제품명 포함 여부
        has_specific_models = False
        
        # 각 부품 유형별로 하나 이상의 제품이 언급되었는지 확인
        for part_type, products in actual_products.items():
            if products:
                # 첫 번째 제품의 일부 텍스트만 추출하여 검색
                first_product = products[0]
                model_text = first_product.split("(")[0].strip()  # 괄호 전 텍스트만 추출
                
                # 모델명이 충분히 길면 일부만 검색
                if len(model_text) > 10:
                    model_text = model_text[:10]
                
                if model_text and model_text in result:
                    has_specific_models = True
                    break
        
        # 제품명이 부족하거나 가격 정보가 없는 경우 보완
        if not has_specific_models or "성능" not in result:
            terminal_logger.capture("로그 추가: ⚠️ 구체적인 제품명이 부족합니다. 보완 중...")
            
            # 응답 보완을 위한 추가 프롬프트
            supplement_prompt = f"""
            앞서 제공한 답변에는 구체적인 제품 모델명이 부족합니다.
            
            아래 게임 사양과 검색된 제품 모델명을 사용하여 다시 답변을 작성해주세요:
            
            {game_specs_info}
            
            {product_info}
            
            각 가격대(저가형/중가형/고가형)별 PC 구성에 대해 다음을 포함하세요:
            1. 정확한 모델명 (위 목록에서 선택)
            2. 각 부품별 예상 성능
            3. 총 구성 완성도
            4. 게임 성능 예상치 (FPS, 해상도, 그래픽 설정)
            
            답변은 한국어로 구체적으로 작성해주세요.
            반드시 답변의 시작에 게임 권장사양과 최소사양을 먼저 언급하세요.
            """
            
            # 보완 요청
            supplement_result = llm.invoke(supplement_prompt)
            if len(supplement_result) > 500:  # 유효한 응답인지 확인
                result = supplement_result
        
        # 실행 시간 기록
        execution_time = time.time() - start_time
        
        # 결과 저장
        state["final_result"] = {
            "explanation": result,
            "execution_time": execution_time,
            "min_specs": min_specs,
            "recommended_specs": recommended_specs,
            "actual_products": actual_products
        }
        
        terminal_logger.capture("로그 추가: ✅ 답변 생성 완료")
        
    except Exception as e:
        error_msg = str(e)
        print(f"LLM 호출 오류: {error_msg}")
        state["errors"].append(f"설명 생성 오류: {error_msg}")
        state["final_result"] = {
            "explanation": f"PC 구성 추천을 생성하는 중 오류가 발생했습니다: {error_msg}",
            "errors": [error_msg]
        }
    
    print("====================== GENERATE EXPLANATION END ======================")
    return state

# 상태 흐름 그래프 구축 - 로깅 추가
def build_graph():
    """LangGraph 그래프 빌드"""
    logger.info("PC 호환성 검사 그래프 구축 시작")
    
    # 로깅과 함께 노드 추가하는 헬퍼 함수
    def add_node_with_logging(name, func):
        def logged_func(state):
            logger.info(f"Starting node: {name}")
            try:
                result = func(state)
                # 상태 로깅을 축소하고 중요한 정보만 로깅
                logger.debug(f"Node {name} completed")
                return result
            except Exception as e:
                logger.error(f"Error in node {name}: {str(e)}")
                raise
        graph.add_node(name, logged_func)
    
    # 그래프 생성
    graph = StateGraph(PCCompatibilityState)
    
    # 노드 추가
    add_node_with_logging("analyze_question", analyze_question)
    add_node_with_logging("generate_queries", generate_queries)
    add_node_with_logging("optimize_search_query", optimize_search_query)
    add_node_with_logging("execute_queries", execute_queries)
    add_node_with_logging("generate_explanation", generate_explanation)
    
    # 엣지 추가
    graph.add_edge("analyze_question", "generate_queries")
    graph.add_edge("generate_queries", "optimize_search_query")
    graph.add_edge("optimize_search_query", "execute_queries")
    graph.add_edge("execute_queries", "generate_explanation")
    graph.add_edge("generate_explanation", END)
    
    # 시작 노드 설정
    graph.set_entry_point("analyze_question")
    
    logger.info("PC 호환성 검사 그래프 구축 완료")
    
    # 컴파일 및 반환
    return graph.compile()

# 그래프 구축
pc_compatibility_graph = build_graph()

# PC 호환성 쿼리 처리 함수 - 이 함수가 호출될 때 로그 캡처가 시작됩니다
def process_pc_compatibility_query(question: str, input_state=None) -> Dict:
    """PC 호환성 쿼리 처리 함수 - 단일 진입점"""
    # 로그 초기화 - 새 질문마다 로그를 초기화합니다
    terminal_logger.clear()
    terminal_logger.capture(f"로그 추가: 🚀 에이전트 처리 시작 - 질문: '{question}' ({datetime.now().strftime('%H:%M:%S')})")
    
    # 초기 상태 설정 - 이 부분은 필수입니다
    initial_state = {
        "question": question,
        "search_keywords": [],
        "part_types": [],
        "queries": {},
        "optimized_queries": {},
        "results": {},
        "errors": [],
        "query_logs": [],
        "analysis_logs": [],
        "program_requirements": "",
        "has_program_requirements": False,
        "components": []
    }
    
    # 입력 상태가 제공된 경우 병합
    if input_state:
        for key, value in input_state.items():
            initial_state[key] = value
    
    # 기본 처리 유형 설정 (LLM 분석 후 업데이트됨)
    processing_type = "PC 부품 호환성 분석"
    
    # 로깅 추가
    logger.info(f"초기 상태: keywords={initial_state['search_keywords']}, part_types={initial_state['part_types']}")
    
    # 그래프 실행 - LLM이 질문을 분석하고 적절한 처리 방식을 결정합니다
    final_state = pc_compatibility_graph.invoke(initial_state)
    
    # 질문 유형에 따라 처리 유형 결정 (LLM 분석 결과 기반)
    if final_state.get("query_type") == "game_pc_recommendation":
        processing_type = "게임 PC 구성 추천"
    elif final_state.get("query_type") == "program_requirements":
        processing_type = "프로그램 요구사항 분석"
    elif final_state.get("existing_parts") and any(final_state["existing_parts"].values()):
        processing_type = "기존 부품 호환 PC 구성 추천"
        
    # 최종 결과 반환
    if "final_result" in final_state and final_state["final_result"]:
        # 결과 정보 추가
        final_state["final_result"]["detailed_query_logs"] = final_state.get("query_logs", [])
        final_state["final_result"]["analysis_logs"] = final_state.get("analysis_logs", [])
        
        # 쿼리 결과 요약 추가
        query_summary = []
        for table, results in final_state.get("results", {}).items():
            query_summary.append(f"테이블 {table}: {len(results)}개 결과")
        
        final_state["final_result"]["query_summary"] = query_summary
        
        # 캡처된 로그 추가
        final_state["final_result"]["terminal_logs"] = terminal_logger.get_logs()
        
        # 중요: processing_type 정보 추가
        final_state["final_result"]["processing_type"] = processing_type
        
        return final_state["final_result"]
    else:
        # 오류 처리
        return {
            "explanation": "질문 처리에 실패했습니다. 다른 질문으로 다시 시도해 주세요.",
            "errors": final_state.get("errors", []),
            "query_logs": final_state.get("query_logs", []),
            "analysis_logs": final_state.get("analysis_logs", []),
            "terminal_logs": terminal_logger.get_logs(),
            "processing_type": "오류"
        }

