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
import logging
from datetime import datetime

# Ollama 모델 초기화 - 온도 추가
llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1  # 낮은 온도로 더 일관된 응답 유도
)
DB_PATH = './cs_agent/db/pc_parts.db'

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
    except Exception as e:
        print(f"Error getting samples: {str(e)}")
    
    return samples

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

# LLM 호출 로깅 함수 추가
def log_llm_call(prompt, response, prompt_name=""):
    """LLM 프롬프트와 응답을 로깅하는 함수"""
    logger = logging.getLogger("PCCompatibilityGraph")
    logger.debug(f"====== LLM PROMPT ({prompt_name}) ======")
    logger.debug(prompt)
    logger.debug(f"====== LLM RESPONSE ({prompt_name}) ======")
    logger.debug(response)
    logger.debug(f"====== END LLM CALL ({prompt_name}) ======")

# 상태 로깅 함수 추가
def log_state(state: PCCompatibilityState, node_name: str = ""):
    """상태 객체의 내용을 로깅하는 함수"""
    logger = logging.getLogger("PCCompatibilityGraph")
    logger.debug(f"===== STATE ({node_name}) =====")
    # 각 필드별로 로깅 (큰 데이터는 요약)
    logger.debug(f"question: {state['question']}")
    logger.debug(f"search_keywords: {state.get('search_keywords', [])}")
    logger.debug(f"part_types: {state.get('part_types', [])}")
    
    # 쿼리는 개수만 로깅
    if 'queries' in state and state['queries']:
        logger.debug(f"queries: {len(state['queries'])} 개")
        # 쿼리 이름 목록도 로깅
        if state['queries']:
            logger.debug(f"query_keys: {list(state['queries'].keys())}")
    else:
        logger.debug("queries: 없음")
    
    # 결과는 개수만 로깅
    if 'results' in state and state['results']:
        results_summary = {}
        for key, value in state['results'].items():
            results_summary[key] = f"{len(value)} 개 결과"
        logger.debug(f"results: {results_summary}")
    else:
        logger.debug("results: 없음")
    
    # 오류 로깅
    if 'errors' in state and state['errors']:
        logger.debug(f"errors: {state.get('errors', [])}")
    else:
        logger.debug("errors: 없음")
    
    logger.debug("===== END STATE =====")
    
    return state

# 1. 질문 분석 노드
def analyze_question(state: PCCompatibilityState) -> PCCompatibilityState:
    """사용자 질문을 분석하여 관련된 부품 유형과 키워드 추출"""
    logger = logging.getLogger("PCCompatibilityGraph")
    question = state["question"]
    
    # 상태 로깅 (함수 시작)
    log_state(state, "analyze_question_start")
    
    # 모델 키워드 추출 프롬프트
    extract_prompt = """
    You are an expert in PC hardware compatibility analysis. Extract key information from the user's question.
    
    User Question: {question}
    
    Analyze the question and extract:
    1. The specific PC part models mentioned (e.g., RTX 4070, Ryzen 7800X3D)
    2. The types of PC parts involved in the question (e.g., GPU, CPU, motherboard, case, memory, cpu_cooler, power_supply, storage)
    
    Return your analysis as a JSON object with these fields:
    - search_keywords: List of model keywords for search (most specific to least specific)
    - part_types: List of PC part types mentioned in the question
    
    Example formats for search keywords:
    - For GPU: ["RTX 4070", "RTX4070", "4070"] #Searching only for RTX or RX may yield too many results
    - For CPU: ["Ryzen 7800X3D", "7800X3D", "7800X"]
    - For Memory: ["DDR4", "DDR5"]
    
    Valid part types include: gpu, cpu, motherboard, case_chassis, memory, cpu_cooler, power_supply, storage
    
    IMPORTANT: Use English for JSON field names, but you can use Korean or English for the values.
    
    JSON response:
    """
    
    # 프롬프트 로깅
    logger.debug(f"분석 프롬프트:\n{extract_prompt.format(question=question)}")
    
    # 키워드 및 부품 유형 추출
    result = llm.invoke(extract_prompt.format(question=question))
    
    # 응답 로깅
    log_llm_call(extract_prompt.format(question=question), result, "analyze_question")
    
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
        # GPU 패턴
        gpu_match = re.search(r'(RTX|GTX|RX)\s*(\d{3,4})(\s*Ti)?', question, re.IGNORECASE)
        if gpu_match:
            full_match = gpu_match.group(0)
            search_keywords.append(full_match)
            search_keywords.append(gpu_match.group(2))  # 숫자만
            if gpu_match.group(1):  # 접두사(RTX, GTX, RX)
                search_keywords.append(gpu_match.group(1))
            part_types.append("gpu")
        
        # CPU 패턴 (Ryzen 또는 Intel)
        cpu_match = re.search(r'(Ryzen|Intel)\s*([\d]+)[\s\-]*([\w\d]+)', question, re.IGNORECASE)
        if cpu_match:
            full_match = cpu_match.group(0)
            search_keywords.append(full_match)
            if cpu_match.group(3):  # 모델 번호 (예: 7800X3D)
                search_keywords.append(cpu_match.group(3))
            if cpu_match.group(1):  # 브랜드 (Ryzen, Intel)
                search_keywords.append(cpu_match.group(1))
            part_types.append("cpu")
        
        # 다양한 부품 유형 확인
        part_type_patterns = {
            "motherboard": r'(메인보드|마더보드|motherboard)',
            "case_chassis": r'(케이스|case)',
            "memory": r'(메모리|램|ram|memory)',
            "cpu_cooler": r'(쿨러|cooler)',
            "power_supply": r'(파워|psu|power)',
            "storage": r'(저장장치|ssd|hdd|storage)'
        }
        
        for part_type, pattern in part_type_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                part_types.append(part_type)
    
    # 상태 업데이트
    state["search_keywords"] = search_keywords
    state["part_types"] = part_types
    
    # 상태 로깅 (함수 종료)
    log_state(state, "analyze_question_end")
    
    return state

# 2. 쿼리 생성 노드
def generate_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """부품 유형 및 키워드에 기반한 SQL 쿼리 생성"""
    logger = logging.getLogger("PCCompatibilityGraph")
    question = state["question"]
    search_keywords = state["search_keywords"]
    part_types = state["part_types"]
    
    # 함수 시작시 상태 로깅
    log_state(state, "generate_queries_start")
    
    # 키워드가 없는 경우 기본 처리
    if not search_keywords:
        state["errors"].append("No search keywords found in the question.")
        return state
    
    # 키워드 처리 개선 - 더 유연한 검색 패턴 사용
    keyword = search_keywords[0]
    search_patterns = [
        f"%{keyword}%",  # 기본 패턴
    ]
    
    # GPU 모델 패턴 개선 (예: RX 7900)
    if any(gpu_brand in keyword.upper() for gpu_brand in ["RTX", "GTX", "RX"]):
        # 브랜드와 모델 번호 분리 시도
        gpu_pattern = re.search(r'(RTX|GTX|RX)\s*(\d{3,4})(\s*Ti)?', keyword, re.IGNORECASE)
        if gpu_pattern:
            brand = gpu_pattern.group(1)
            model = gpu_pattern.group(2)
            search_patterns.extend([
                f"%{brand}%{model}%",       # RX 7900
                f"%라데온%{brand}%{model}%", # 라데온 RX 7900
                f"%{model}%",               # 7900만
            ])
    
    # CPU 모델 패턴 개선 (예: 5600X)
    if "cpu" in part_types:
        cpu_pattern = re.search(r'(\d{4})(\s*[Xx]\d?)?', keyword, re.IGNORECASE)
        if cpu_pattern:
            model = cpu_pattern.group(1)
            suffix = cpu_pattern.group(2) or ""
            suffix = suffix.strip()
            search_patterns.extend([
                f"%{model}{suffix}%",      # 5600X
                f"%라이젠%{model}{suffix}%", # 라이젠 5600X
                f"%RYZEN%{model}{suffix}%", # RYZEN 5600X
            ])
    
    # 관계 매핑
    relations = []
    
    # 모든 가능한 부품 간 관계 정의 - GPU-메인보드 호환성 우선순위 낮춤
    compatibility_relations = {
        # GPU 관련 호환성 - 케이스와 전원 우선
        ("gpu", "case_chassis"): "gpu_case",
        ("gpu", "power_supply"): "gpu_psu",
        # GPU-메인보드 호환성은 대부분 PCIe로 해결되므로 우선순위 낮춤
        ("gpu", "motherboard"): "gpu_motherboard",
        
        # 다른 호환성 관계
        ("cpu", "motherboard"): "cpu_motherboard",
        ("cpu", "cpu_cooler"): "cpu_cooler",
        ("motherboard", "case_chassis"): "motherboard_case",
        ("motherboard", "memory"): "motherboard_memory",
        ("motherboard", "storage"): "motherboard_storage",
        ("power_supply", "case_chassis"): "psu_case",
        ("cpu_cooler", "case_chassis"): "cooler_case"
    }
    
    # 언급된 부품 타입 간의 모든 가능한 호환성 관계 추가
    for i, type1 in enumerate(part_types):
        for type2 in part_types[i+1:]:
            relation_key = tuple(sorted([type1, type2]))
            if relation_key in compatibility_relations or (relation_key[1], relation_key[0]) in compatibility_relations:
                if relation_key in compatibility_relations:
                    relations.append(compatibility_relations[relation_key])
                else:
                    relations.append(compatibility_relations[(relation_key[1], relation_key[0])])
    
    # 부품 타입이 있지만 관계가 없는 경우, 주요 호환성 관계 추가
    if part_types and not relations:
        primary_part = part_types[0]
        if primary_part == "gpu":
            # GPU의 경우 케이스 및 전원 호환성을 우선시
            relations.append("gpu_case")
            relations.append("gpu_psu")
            # 메인보드 호환성은 마지막에 추가
            relations.append("gpu_motherboard")
        elif primary_part == "cpu":
            relations.append("cpu_motherboard")
            relations.append("cpu_cooler")
        elif primary_part == "motherboard":
            relations.append("motherboard_case")
            relations.append("motherboard_memory")
            relations.append("motherboard_storage")
        elif primary_part == "power_supply":
            relations.append("psu_case")
    
    # 관계가 여전히 없는 경우, 키워드를 기반으로 추정
    if not relations:
        # GPU 키워드 확인
        if any(kw in keyword.upper() for kw in ["RTX", "GTX", "RX", "RADEON", "GEFORCE"]):
            # GPU의 경우 케이스 및 전원 호환성을 우선시
            relations.append("gpu_case")
            relations.append("gpu_psu")
            # 메인보드 호환성은 마지막에 추가
            relations.append("gpu_motherboard")
        # CPU 키워드 확인
        elif any(kw in keyword.upper() for kw in ["RYZEN", "INTEL", "CORE", "I7", "I9", "5800X", "7800X"]):
            relations.append("cpu_motherboard")
            relations.append("cpu_cooler")
        # 메모리 키워드 확인
        elif any(kw in keyword.upper() for kw in ["DDR4", "DDR5", "RAM", "GB", "VENGEANCE", "DOMINATOR"]):
            relations.append("motherboard_memory")
    
    # SQL 쿼리 생성
    queries = {}
    
    # 테이블과 쿼리 매핑을 명확히 정의
    query_templates = {
        # GPU와 케이스 호환성 (수정됨)
        "gpu_case": """
            WITH RankedCases AS (
                SELECT 
                    g.model_name AS gpu_model, 
                    g.length AS gpu_length, 
                    c.model_name AS case_model,
                    c.vga_length AS available_gpu_length,
                    g.manufacturer AS gpu_manufacturer,
                    c.manufacturer AS case_manufacturer,
                    (c.vga_length - g.length) AS space_difference,
                    ROW_NUMBER() OVER (PARTITION BY g.model_name ORDER BY (c.vga_length - g.length)) AS rank
                FROM 
                    gpu g, case_chassis c
                WHERE 
                    g.model_name LIKE '%{pattern}%' 
                    AND g.length <= c.vga_length
            )
            SELECT 
                gpu_model, 
                gpu_length, 
                case_model,
                available_gpu_length,
                gpu_manufacturer,
                case_manufacturer,
                space_difference
            FROM 
                RankedCases
            WHERE 
                rank = 1
            ORDER BY 
                CASE WHEN gpu_model LIKE '%Ti%' THEN 1 ELSE 0 END,
                gpu_model
            LIMIT 10
        """,
        
        # GPU와 전원 호환성 (수정됨)
        "gpu_psu": """
            SELECT 
                g.model_name AS gpu_model, 
                g.power_consumption AS gpu_power,
                p.model_name AS psu_model,
                p.wattage AS psu_wattage,
                g.manufacturer AS gpu_manufacturer,
                p.manufacturer AS psu_manufacturer
            FROM 
                gpu g, power_supply p
            WHERE 
                g.model_name LIKE '{pattern}' 
                AND g.power_consumption <= (p.wattage * 0.7)
            LIMIT 10
        """,
        
        # CPU와 메인보드 호환성 (수정됨)
        "cpu_motherboard": """
            SELECT 
                c.model_name AS cpu_model, 
                c.socket_type AS cpu_socket, 
                m.model_name AS motherboard_model,
                m.socket_type AS mb_socket,
                c.manufacturer AS cpu_manufacturer,
                m.manufacturer AS mb_manufacturer
            FROM 
                cpu c, motherboard m
            WHERE 
                (c.model_name LIKE '{pattern}' OR c.socket_type = '{pattern}')
                AND c.socket_type = m.socket_type
            LIMIT 10
        """,
        
        # 메인보드와 케이스 호환성 (수정됨)
        "motherboard_case": """
            SELECT 
                m.model_name AS motherboard_model, 
                m.form_factor AS mb_form_factor, 
                c.model_name AS case_model,
                c.supported_mb_types AS case_supported_mb_types,
                m.manufacturer AS mb_manufacturer,
                c.manufacturer AS case_manufacturer
            FROM 
                motherboard m, case_chassis c
            WHERE 
                m.model_name LIKE '{pattern}'
                AND (
                    (m.form_factor = 'ATX' AND c.supported_mb_types LIKE '%ATX%') OR
                    (m.form_factor = 'mATX' AND c.supported_mb_types LIKE '%mATX%') OR
                    (m.form_factor = 'ITX' AND c.supported_mb_types LIKE '%ITX%')
                )
            LIMIT 10
        """,
        
        # 기본 쿼리 템플릿 (변경 없음)
        "default": """
            SELECT * FROM {table_name} 
            WHERE model_name LIKE '{pattern}'
            LIMIT 10
        """
    }

    
    # 각 관계에 대한 쿼리 생성
    for relation in relations:
        # 각 검색 패턴에 대해 시도
        query = None
        for pattern in search_patterns:
            # 쿼리 템플릿 선택
            if relation in query_templates:
                query = query_templates[relation].format(pattern=pattern)
            else:
                # 관계에 해당하는 테이블 찾기
                table_name = relation
                # 매핑된 테이블 이름 확인
                if table_name in table_mapping:
                    table_name = table_mapping[table_name]
                
                # 테이블이 존재하는지 확인
                if table_name in db_schema:
                    query = query_templates["default"].format(
                        table_name=table_name,
                        pattern=pattern
                    )
                else:
                    # 테이블이 없으면 다음 패턴으로 넘어감
                    continue
            
            # 쿼리가 생성되었으면 저장하고 루프 종료
            if query:
                queries[relation] = query
                logger.debug(f"생성된 SQL 쿼리 ({relation}): \n{query}")
                break
    
    logger.info(f"생성된 쿼리 관계: {list(queries.keys())}")
    
    # 상태 업데이트
    state["queries"] = queries
    
    # 함수 종료시 상태 로깅
    log_state(state, "generate_queries_end")
    
    return state

# 쿼리 최적화 함수 개선
def optimize_search_query(state: PCCompatibilityState) -> PCCompatibilityState:
    """SQL 쿼리를 최적화하고 데이터베이스에 맞게 조정"""
    logger = logging.getLogger("PCCompatibilityGraph")
    part_types = state["part_types"]
    queries = state["queries"]
    search_keywords = state["search_keywords"]
    
    # 함수 시작시 상태 로깅
    log_state(state, "optimize_search_query_start")
    
    # 쿼리가 없는 경우 기본 처리
    if not queries:
        state["errors"].append("No queries generated.")
        return state
    
    # 최적화된 쿼리 저장
    optimized_queries = {}
    errors = []
    
    # LLM을 이용한 쿼리 최적화
    for relation, query in queries.items():
        try:
            # 이미 쿼리가 잘 구성되어 있다면 그대로 사용
            optimized_queries[relation] = query
            logger.debug(f"최적화 쿼리 ({relation}):\n{query}")
        except Exception as e:
            error_msg = f"쿼리 최적화 중 오류 발생: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
    
    # 상태 업데이트
    state["queries"] = optimized_queries
    if errors:
        state["errors"].extend(errors)
    
    # 로깅
    if optimized_queries:
        logger.info(f"Identified part types: {part_types}")
        logger.info(f"Extracted keywords: {search_keywords}")
        logger.info(f"Generated {len(optimized_queries)} queries for relations: {list(optimized_queries.keys())}")
    
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    
    # 함수 종료시 상태 로깅
    log_state(state, "optimize_search_query_end")
    
    return state

# 쿼리 실행 함수 개선
def execute_queries(state: PCCompatibilityState) -> PCCompatibilityState:
    """SQL 쿼리 실행 및 결과 처리"""
    logger = logging.getLogger("PCCompatibilityGraph")
    queries = state["queries"]
    
    # 함수 시작시 상태 로깅
    log_state(state, "execute_queries_start")
    
    logger.info("쿼리 실행 시작")
    logger.info(f"실행할 쿼리: {list(queries.keys())}")
    
    results = {}
    errors = []
    
    # 결과가 충분한지 확인하는 임계값
    SUFFICIENT_RESULTS = 5
    
    # 데이터베이스 연결
    try:
        with duckdb.connect(database=DB_PATH, read_only=True) as conn:
            for relation, query in queries.items():
                try:
                    logger.debug(f"===== 쿼리 실행 ({relation}) =====")
                    logger.debug(f"SQL:\n{query}")
                    
                    # 쿼리 실행
                    df = conn.execute(query).fetchdf()
                    
                    # 결과 변환
                    if not df.empty:
                        records = df.to_dict('records')
                        results[relation] = records
                        logger.debug(f"쿼리 결과 ({relation}): {len(records)} 개 레코드")
                        
                        # 처음 몇 개 결과 샘플 로깅
                        for i, record in enumerate(records[:3]):
                            logger.debug(f"  결과 {i+1}: {record}")
                    else:
                        logger.debug(f"쿼리 결과 ({relation}): 결과 없음")
                        
                except Exception as e:
                    error_msg = f"Error executing query for {relation}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    logger.exception(f"쿼리 실행 중 예외 발생 ({relation})")
    except Exception as e:
        error_msg = f"데이터베이스 연결 오류: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    # 상태 업데이트
    state["results"] = results
    if errors:
        state["errors"].extend(errors)
    
    # 함수 종료시 상태 로깅
    log_state(state, "execute_queries_end")
    
    return state

# 4. 결과 설명 생성 노드
def generate_explanation(state: PCCompatibilityState) -> PCCompatibilityState:
    """결과 통합 및 상세한 추천 PC 구성 설명 생성 (추천 구성 자동 생성 추가)"""
    logger = logging.getLogger("PCCompatibilityGraph")
    question = state["question"]
    results = state["results"]
    errors = state["errors"]
    search_keywords = state["search_keywords"]
    part_types = state["part_types"]
    
    # 상태 로깅
    log_state(state, "generate_explanation_start")
    
    # 실제 검색 키워드를 기반으로 인식된 모델 확인
    recognized_models = {}
    for keyword in search_keywords:
        if "gpu" in part_types and any(gpu_term in keyword.lower() for gpu_term in ["rtx", "rx", "geforce", "radeon"]):
            recognized_models["gpu"] = keyword
        elif "cpu" in part_types and any(cpu_term in keyword.lower() for cpu_term in ["ryzen", "core", "intel", "amd"]):
            recognized_models["cpu"] = keyword
    
    # 추천 구성 자동 생성: 각 부품별 후보를 쿼리 결과에서 추출
    recommended_config = {}
    
    # GPU 및 케이스: gpu_case 쿼리 결과 활용
    if "gpu_case" in results and results["gpu_case"]:
        first_row = results["gpu_case"][0]
        recommended_config["GPU"] = first_row.get("gpu_model", recognized_models.get("gpu", "적합한 GPU"))
        recommended_config["케이스"] = first_row.get("case_model", "적합한 케이스")
        
        # 공간 효율성 정보 추가
        space_diff = first_row.get("space_difference")
        if space_diff is not None and space_diff > 0:
            recommended_config["케이스 여유 공간"] = f"{space_diff}mm"
    else:
        recommended_config["GPU"] = recognized_models.get("gpu", "적합한 GPU")
        recommended_config["케이스"] = "적합한 케이스 (GPU 길이 확인 필요)"

    # CPU와 메인보드: cpu_motherboard 결과 활용
    if "cpu_motherboard" in results and results["cpu_motherboard"]:
        first_row = results["cpu_motherboard"][0]
        recommended_config["CPU"] = recognized_models.get("cpu", "적합한 CPU")
        recommended_config["메인보드"] = first_row.get("motherboard_model", "호환되는 메인보드")
    elif "motherboard_case" in results and results["motherboard_case"]:
        first_row = results["motherboard_case"][0]
        recommended_config["CPU"] = recognized_models.get("cpu", "적합한 CPU")
        recommended_config["메인보드"] = first_row.get("motherboard_model", "호환되는 메인보드")
    else:
        recommended_config["CPU"] = recognized_models.get("cpu", "적합한 CPU")
        recommended_config["메인보드"] = "호환되는 메인보드"

    # PSU: gpu_psu 결과 활용
    if "gpu_psu" in results and results["gpu_psu"]:
        first_row = results["gpu_psu"][0]
        recommended_config["PSU"] = first_row.get("psu_model", "충분한 용량의 PSU")
    else:
        recommended_config["PSU"] = "충분한 용량의 PSU"
    
    # CPU 쿨러: cpu_cooler 결과 활용 (컬럼 이름: cooler_model)
    if "cpu_cooler" in results and results["cpu_cooler"]:
        first_row = results["cpu_cooler"][0]
        recommended_config["CPU 쿨러"] = first_row.get("cooler_model", "")
    # (없으면 선택하지 않음)
    
    # 요약 데이터에 추천 구성 포함
    summary = {
        "추천구성": recommended_config
    }
    
    # 쿼리 결과를 텍스트로 변환
    results_str = ""
    if not results:
        results_str = "데이터베이스에서 관련 정보를 찾을 수 없습니다. 호환성 데이터가 없습니다."
    else:
        for relation, result in results.items():
            relation_kr = relation.replace("gpu_motherboard", "GPU와 메인보드")
            relation_kr = relation_kr.replace("gpu_case", "GPU와 케이스")
            relation_kr = relation_kr.replace("cpu_motherboard", "CPU와 메인보드")
            relation_kr = relation_kr.replace("motherboard_case", "메인보드와 케이스")
            
            results_str += f"{relation_kr} 호환성:\n"
            if result:
                for item in result[:10]:
                    results_str += f"- {', '.join([f'{k}: {v}' for k, v in item.items()])}\n"
                if len(result) > 10:
                    results_str += f"... 그리고 {len(result) - 10}개 더 있음\n"
            else:
                results_str += "호환되는 부품을 찾을 수 없습니다.\n"
            results_str += "\n"
    
    errors_str = "\n".join(errors) if errors else "오류 없음."
    search_keywords_str = ", ".join(search_keywords) if search_keywords else "인식된 키워드 없음"
    
    # 추천 구성 문자열 생성
    recommended_config_str = "추천 PC 구성:\n"
    for comp, model in recommended_config.items():
        recommended_config_str += f"- {comp}: {model}\n"
    
    # 개선된 LLM 프롬프트 (호환성 결과와 추천 구성을 명시적으로 포함)
    explanation_prompt = f"""
    당신은 PC 하드웨어 호환성 전문가입니다. 사용자에게 호환성 결과와 추천 PC 구성을 상세하게 설명해 주세요.
    
    사용자 질문: {question}
    실제 검색한 키워드: {search_keywords_str}
    분석된 부품 유형: {part_types}
    
    --- 호환성 결과 ---
    {results_str}
    
    --- 추천 PC 구성 ---
    {recommended_config_str}
    
    --- 오류 ---
    {errors_str}
    
    위 정보를 바탕으로 다음 내용을 포함하여 상세히 설명해 주세요:
    1. 요약: 호환성 결과와 추천 구성에 대해 간결히 요약.
    2. 추천 PC 구성: 각 부품(예: GPU, CPU, 메인보드, 케이스, PSU 등)의 구체적인 모델명을 추천하고, 선택한 이유를 설명.
    3. 데이터 부족 시 주의사항을 명시할 것.
    
    설명:
    """
    
    explanation = llm.invoke(explanation_prompt)
    
    # 로깅
    log_llm_call(explanation_prompt, explanation, "generate_explanation")
    
    final_result = {
        "summary": summary,
        "recommended_config": recommended_config,
        "detailed_results": results,
        "explanation": explanation,
        "errors": errors
    }
    
    state["final_result"] = final_result
    log_state(state, "generate_explanation_end")
    return state


# 상태 흐름 그래프 구축 - 로깅 추가
def build_graph():
    # 로깅 디렉토리 설정
    log_dir = "./cs_agent/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"langgraph_log_{timestamp}.log")
    
    # 로깅 설정 - DEBUG로 변경
    logging.basicConfig(
        level=logging.DEBUG,  # INFO에서 DEBUG로 변경
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 기존 로거가 있으면 핸들러 정리
    logger = logging.getLogger("PCCompatibilityGraph")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 추가 - DEBUG 레벨로 설정
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # DEBUG 레벨로 설정
    
    # 콘솔 핸들러 추가 - INFO 레벨 유지
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    console_handler.setLevel(logging.INFO)  # 콘솔은 INFO 유지
    
    # 로거에 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)  # 로거 자체는 DEBUG 레벨로 설정
    
    # 로깅과 함께 노드 추가하는 헬퍼 함수
    def add_node_with_logging(name, func):
        def logged_func(state):
            logger.info(f"Starting node: {name}")
            try:
                result = func(state)
                log_state(result, name)  # 상태 로깅 (반환값 무시)
                return result         # 원래의 결과를 반환
            except Exception as e:
                logger.error(f"Error in node {name}: {str(e)}")
                logger.exception(f"Exception details for {name}:")
                raise
        graph.add_node(name, logged_func)
    
    # 그래프 생성
    graph = StateGraph(PCCompatibilityState)
    
    # 로깅과 함께 노드 추가
    add_node_with_logging("analyze_question", analyze_question)
    add_node_with_logging("generate_queries", generate_queries)
    add_node_with_logging("optimize_search_query", optimize_search_query)
    add_node_with_logging("execute_queries", execute_queries)
    add_node_with_logging("generate_explanation", generate_explanation)
    
    # 엣지 추가 (노드 간 연결)
    graph.add_edge("analyze_question", "generate_queries")
    graph.add_edge("generate_queries", "optimize_search_query")
    graph.add_edge("optimize_search_query", "execute_queries")
    graph.add_edge("execute_queries", "generate_explanation")
    graph.add_edge("generate_explanation", END)
    
    # 시작 노드 설정
    graph.set_entry_point("analyze_question")
    
    logger.info("Graph compiled successfully")
    
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


# # 테스트 질문
test_query = "RTX 3080이랑 호환되는 메인보드랑 케이스 알려줘 CPU는 5600x 사용하고싶어"
# test_query = "RTX 3080 제품 스펙을 자세히 알고싶어"

# 함수 호출
result = process_pc_compatibility_query(test_query)

# 결과 출력
print("\nExplanation:")
print(result["explanation"])