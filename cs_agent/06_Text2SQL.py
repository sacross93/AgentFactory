from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import duckdb
import pandas as pd
import re

# Ollama 모델 초기화
llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434"
)

# 데이터베이스 연결
conn = duckdb.connect('pc_parts.db')

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

# 테이블 이름 매핑 (일반적으로 사용되는 이름 -> 실제 테이블 이름)
table_mapping = {
    "cpu_support": "cpu_mb_compatibility",
    "motherboard_compatibility": "cpu_mb_compatibility",
    "cpu_motherboard": "cpu_mb_compatibility",
    "mb": "motherboard",
    "mainboard": "motherboard",
    "case": "case_chassis",
    "cooler": "cpu_cooler",
    "psu": "power_supply",
    "ram": "memory",
    "gpu_compatibility": "gpu_case_compatibility",
    "cpu_compatibility": "cpu_mb_compatibility"
}

# 스키마 정보를 문자열로 변환
schema_str = "Database Schema:\n"
for table, columns in db_schema.items():
    schema_str += f"Table: {table}\n"
    schema_str += f"Columns: {', '.join(columns)}\n\n"

# 테이블 이름 검증 및 수정 함수
def validate_and_fix_table_names(query):
    # 모든 테이블 이름 목록
    all_tables = list(db_schema.keys())
    
    # 쿼리에서 FROM과 JOIN 절 이후의 테이블 이름 찾기
    table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables_in_query = re.findall(table_pattern, query, re.IGNORECASE)
    
    # 수정된 쿼리
    modified_query = query
    
    for table in tables_in_query:
        # 테이블이 존재하지 않는 경우
        if table not in all_tables:
            # 매핑에서 대체 테이블 찾기
            if table in table_mapping and table_mapping[table] in all_tables:
                # 테이블 이름 대체
                modified_query = re.sub(
                    r'(\b)' + table + r'(\b)', 
                    table_mapping[table], 
                    modified_query
                )
            else:
                # 유사한 테이블 이름 찾기
                similar_tables = find_similar_tables(table, all_tables)
                if similar_tables:
                    modified_query = re.sub(
                        r'(\b)' + table + r'(\b)', 
                        similar_tables[0], 
                        modified_query
                    )
    
    return modified_query

# 유사한 테이블 이름 찾기 함수
def find_similar_tables(table_name, all_tables, threshold=0.6):
    similar_tables = []
    
    # 간단한 유사도 계산 (부분 문자열 포함 여부)
    for existing_table in all_tables:
        # 테이블 이름이 서로 포함 관계인 경우
        if table_name.lower() in existing_table.lower() or existing_table.lower() in table_name.lower():
            similar_tables.append(existing_table)
    
    # 유사한 테이블이 없으면 호환성 테이블 중에서 찾기
    if not similar_tables:
        for existing_table in all_tables:
            if "compatibility" in existing_table.lower():
                similar_tables.append(existing_table)
    
    return similar_tables

# SQL 쿼리 실행 함수
def execute_sql_query(query):
    try:
        # 마크다운 코드 블록 제거
        query = clean_sql_query(query)
        
        # 테이블 이름 검증 및 수정
        query = validate_and_fix_table_names(query)
        
        result = conn.execute(query).fetchdf()
        return result
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# SQL 쿼리 정리 함수
def clean_sql_query(query):
    # 마크다운 코드 블록 제거 (```sql과 ``` 제거)
    query = re.sub(r'```sql\s*', '', query)
    query = re.sub(r'```\s*', '', query)
    
    # 앞뒤 공백 제거
    query = query.strip()
    
    return query

# 자연어를 SQL로 변환하는 프롬프트 템플릿
sql_prompt_template = """
You are an SQL expert for a PC parts compatibility database.
Convert the user's question into an SQL query.

{schema_str}

IMPORTANT: Use ONLY the exact table and column names from the schema above. Do not invent new tables or columns.

User Question: {question}

Follow these rules:
1. For compatibility-related questions, use the appropriate compatibility tables (e.g., cpu_mb_compatibility, gpu_case_compatibility).
2. When part names or model numbers are mentioned, use the LIKE operator with just the model number part.
   For example, if user mentions "Ryzen 7800X3D", search for "%7800X3D%" rather than "%Ryzen 7800X3D%".
3. Select only necessary columns to make the results easy to understand.
4. Include part names and models in the query results.
5. Return only the SQL query without any markdown formatting or code blocks.

SQL Query:
"""

# 프롬프트 템플릿 생성
sql_prompt = PromptTemplate(
    template=sql_prompt_template,
    input_variables=["schema_str", "question"]
)

# 자연어를 SQL로 변환하는 체인
def text_to_sql(question):
    prompt = sql_prompt.format(schema_str=schema_str, question=question)
    sql_query = llm.invoke(prompt)
    return clean_sql_query(sql_query)

# 결과 설명 프롬프트 템플릿
explanation_prompt_template = """
Please explain the following SQL query results in a way that's easy for the user to understand:

SQL Query: {query}

Query Results:
{result}

User Question: {question}

Provide a clear and concise explanation without using markdown formatting or code blocks.

Explanation:
"""

# 프롬프트 템플릿 생성
explanation_prompt = PromptTemplate(
    template=explanation_prompt_template,
    input_variables=["query", "result", "question"]
)

# 결과를 설명하는 체인
def explain_results(query, result, question):
    prompt = explanation_prompt.format(query=query, result=result, question=question)
    explanation = llm.invoke(prompt)
    return clean_explanation(explanation)

# 설명 정리 함수
def clean_explanation(explanation):
    # 마크다운 형식 제거
    explanation = re.sub(r'```.*?\n', '', explanation)
    explanation = re.sub(r'```', '', explanation)
    return explanation.strip()

# 키워드 추출 함수
def extract_model_keywords(question):
    """모델명에서 핵심 키워드 추출 (숫자와 관련 문자)"""
    # 정규식을 사용하여 모델 번호 패턴 찾기 (예: 7800X3D, i9-13900K, RTX 4090 등)
    model_patterns = re.findall(r'\b\w*\d+\w*[-]?\w*\d*\w*\b', question)
    return model_patterns

# 대체 쿼리 생성 함수
def create_fallback_query(original_query, question):
    """결과가 없을 때 사용할 대체 쿼리 생성"""
    # 원본 쿼리가 WHERE 절을 포함하는지 확인
    if 'WHERE' not in original_query.upper():
        return original_query
    
    # 키워드 추출
    keywords = extract_model_keywords(question)
    if not keywords:
        return original_query
    
    # 가장 유력한 키워드 (일반적으로 가장 긴 것)
    main_keyword = max(keywords, key=len)
    
    # WHERE 절을 대체
    parts = re.split(r'(WHERE\s+)', original_query, flags=re.IGNORECASE)
    where_idx = -1
    for i, part in enumerate(parts):
        if re.match(r'WHERE\s+', part, re.IGNORECASE):
            where_idx = i
            break
    
    if where_idx >= 0 and where_idx + 1 < len(parts):
        # WHERE 절 이후 AND 이전 조건을 변경
        conditions = parts[where_idx + 1].split('AND')
        conditions[0] = f" model_name LIKE '%{main_keyword}%' "
        parts[where_idx + 1] = 'AND'.join(conditions)
        return ''.join(parts)
    
    return original_query

# Function calling을 위한 함수 정의
def process_pc_compatibility_query(query: str) -> dict:
    """
    Process user questions about PC parts compatibility.
    
    Args:
        query: User's question (e.g., "Find motherboards compatible with Ryzen 7800X3D")
        
    Returns:
        dict: Dictionary containing the processing results
    """
    # 1. 자연어 질문을 SQL로 변환
    sql_query = text_to_sql(query)
    
    # 2. SQL 쿼리 실행
    result = execute_sql_query(sql_query)
    
    # 3. 결과가 없거나 오류인 경우 대체 쿼리 시도
    if isinstance(result, pd.DataFrame) and result.empty:
        fallback_query = create_fallback_query(sql_query, query)
        if fallback_query != sql_query:
            result = execute_sql_query(fallback_query)
            sql_query = fallback_query  # 성공한 쿼리로 업데이트
    
    # 4. 결과 설명 생성
    if isinstance(result, pd.DataFrame):
        if result.empty:
            explanation = "No results found."
        else:
            # DataFrame을 문자열로 변환
            result_str = result.to_string()
            explanation = explain_results(sql_query, result_str, query)
    else:
        explanation = result  # 오류 메시지
    
    return {
        "sql_query": sql_query,
        "result": result if isinstance(result, str) else result.to_dict(orient="records"),
        "explanation": explanation
    }

# 함수 스키마 정의
function_schema = {
    "name": "process_pc_compatibility_query",
    "description": "Process PC parts compatibility questions and generate SQL queries to return results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User's question about PC parts compatibility"
            }
        },
        "required": ["query"]
    }
}


test_query = "라이젠 7800X3D에 호환되는 메인보드 찾아줘"

# 함수 호출
result = process_pc_compatibility_query(test_query)

# 결과 출력
print("SQL Query:")
print(result["sql_query"])
print("\nExplanation:")
print(result["explanation"])