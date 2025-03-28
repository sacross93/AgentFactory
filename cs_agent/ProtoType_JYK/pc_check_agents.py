from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
import sys
import json
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from pc_check_func import *
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv
dotenv.load_dotenv('./.env')
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY_JY")

explainer_llm = OllamaLLM(
    model="gemma3:27b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1
)

llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-exp-03-25",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

flash_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class TableInfo(BaseModel):
    user_meaning: str = Field(description="Interpretation of the user's query intent")
    reason: str = Field(description="Reason for selecting these tables")
    table_names: List[str] = Field(description="Table names needed to resolve the user query")

class QueryInfo(BaseModel):
    reason: str = Field(description="Explain why you selected these columns and why you wrote the query this way.")
    selected_columns: List[str] = Field(description="Select columns needed to address the user query.")
    query: str = Field(description="Write a SQL query that addresses the user's question.")

class QueryModify(BaseModel):
    reason: str = Field(description="Explain why the query is not working and how to fix it.")
    queries: List[Dict[str, str]] = Field(description="List of queries, each with a type and SQL statement.")

class CheckResult(BaseModel):
    reason: str = Field(description="Explanation of why the result is suitable or not suitable for answering the user's question.")
    result: bool = Field(description="Whether this data is sufficient to answer the user's question.")
    answer: str = Field(description="The answer to the user's query based on the provided data.")
    issues: List[str] = Field(description="Specific issues or missing information in the result.")

def table_abstract_agent(user_query):
    # 실제 존재하는 테이블 정보 가져오기
    db_tables = get_db_table()
    table_list = [table[0] for table in db_tables]
    
    table_abstract_prompt = PromptTemplate.from_template("""
    You are an expert who analyzes user queries and extracts table names needed to resolve those queries.
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    Important guidelines:
    1. When dealing with compatibility queries, don't select only the compatibility table. For example, if checking CPU and motherboard compatibility, you need to select the CPU table, motherboard table, AND the CPU-motherboard compatibility table.
    2. Compatibility tables typically only contain IDs and compatibility status, not detailed product information like names or specifications.
    3. You must only select tables that are explicitly listed in the provided Tables section. Do not reference tables that don't exist in the database.
    4. If a component is marked as "Not specified" in the user query, ignore that component completely and do not include its related tables in your selection.
    5. Only consider components that have actual values specified in the query.
    6. IMPORTANT: Only select from the tables that actually exist in the database - they are explicitly listed below.

    Your response must be a JSON object with three keys:
        user_meaning:
            - Interpretation of the user's query intent
        reason:
            - Explanation of why these tables were selected for the query
        table_names:
            - Tables selected to address the user's query
            - Only include tables from this exact list: {table_list}
            - Do not include tables for components marked as "Not specified"
        
    Here is the output schema:
        {{"properties":{{"user_meaning": {{"title": "User Meaning", "description": "Interpretation of the user's query intent", "type": "string"}}, "reason": {{"title": "Reason", "description": "Explanation of why these tables were selected for the query", "type": "string"}}, "table_names": {{"title": "Table Names", "description": "Table names needed to resolve the user query", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["user_meaning", "reason", "table_names"]}}
        
    Available tables in database: {table_list}
    User query: ```{input}```
    """)

    table_abstract_chain = table_abstract_prompt | explainer_llm | JsonOutputParser(pydantic_object=TableInfo)
    table_info = table_abstract_chain.invoke({"input": user_query, "table_list": table_list})
    return table_info

def query_optimize_agent(user_query, table_info):
    # 실제 DB 스키마 정보 가져오기
    all_tables = [table[0] for table in get_db_table()]
    
    # 사용자가 요청한 테이블 중 실제 존재하는 테이블만 필터링
    requested_tables = table_info['table_names']
    valid_tables = [t for t in requested_tables if t in all_tables]
    
    # 존재하지 않는 테이블 로깅
    invalid_tables = [t for t in requested_tables if t not in all_tables]
    if invalid_tables:
        print(f"[경고] 존재하지 않는 테이블이 요청됨: {', '.join(invalid_tables)}")
    
    # 유효한 테이블의 컬럼 정보 가져오기
    table_columns = get_db_samples(valid_tables)
    
    query_optimize_prompt = PromptTemplate.from_template("""
    You are an SQL query expert. Write a query to resolve the user's question using available database tables.
    
    Important guidelines:
    1. Start with simple queries for each product type instead of complex joins.
    2. Only use tables and columns that actually exist in the database (shown below).
    3. For RAM queries, use the 'capacity' column to filter by GB size (e.g., WHERE capacity >= 16 for 16GB or more).
    4. For partial name matches, use LIKE with wildcards (e.g., '%5600%' not '5600').
    5. Avoid complex join structures until you've verified basic queries work.
    6. Limit each query result to 5 items unless specified otherwise.
    7. Don't try to join tables that don't have clear relationships.
    8. Focus on finding components that match the user's specifications first.
    
    Available tables and their columns:
    ```
    {table_columns}
    ```
    
    Valid tables for this query: {valid_tables}
    
    Your response must be a JSON object with these keys:
        reason:
            - Explain your query strategy
            - Describe how these queries resolve the user's question
        selected_columns:
            - List columns you selected for your query
        query:
            - A properly formatted SQL query that addresses the user's question
            
    Here is the output schema:
        {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation for column selection and query design", "type": "string"}}, "selected_columns": {{"title": "Selected Columns", "description": "Columns selected to address the user query", "type": "array", "items": {{"type": "string"}}}}, "query": {{"title": "Query", "description": "SQL query to resolve the user's question", "type": "string"}}}}, "required": ["reason", "selected_columns", "query"]}}
        
    user query: ```{user_query}```
    user meaning: ```{user_meaning}```
    """)

    query_optimize_chain = query_optimize_prompt | explainer_llm | JsonOutputParser(pydantic_object=QueryInfo)
    query_info = query_optimize_chain.invoke({
        "user_query": user_query, 
        "user_meaning": table_info["user_meaning"], 
        "table_columns": table_columns,
        "valid_tables": valid_tables
    })
    
    fixed_query = fix_union_query(query_info['query'])
    result = sql(fixed_query)
    return query_info, result

def query_modify_agent(user_query, table_info, query_info, result, attempt_history):
    # 결과 분석
    if isinstance(result, list) and len(result) == 0:
        problem = "No results were returned."
    elif isinstance(result, str) and "error" in result.lower():
        problem = f"Query error: {result}"
    else:
        problem = "The result does not adequately address the user's query."
    
    # 실제 DB 스키마 정보 가져오기
    all_tables = [table[0] for table in get_db_table()]
    
    # 사용자가 요청한 테이블 중 실제 존재하는 테이블만 필터링
    requested_tables = table_info['table_names']
    valid_tables = [t for t in requested_tables if t in all_tables]
    
    # 존재하지 않는 테이블 로깅
    invalid_tables = [t for t in requested_tables if t not in all_tables]
    if invalid_tables:
        print(f"[경고] 존재하지 않는 테이블이 요청됨: {', '.join(invalid_tables)}")
    
    # 유효한 테이블의 컬럼 정보 가져오기
    table_columns = get_db_samples(valid_tables)

    query_modify_prompt = PromptTemplate.from_template("""
    You are an SQL query writing and optimization expert.

    The previous query did not produce expected results. Create improved separate queries, one for each product type.

    Important guidelines:
    1. Start with basic SELECT statements without complex joins
    2. Only use tables and columns that actually exist in the database
    3. For each component category, create a separate simple query
    4. Use LIKE with wildcards for flexible matching (e.g., '%3060%' not '3060')
    5. For RAM queries, use the 'capacity' column to filter by GB size (e.g., WHERE capacity >= 16 for 16GB or more)
    6. For RAM compatibility with motherboards, check the memory_standard or memory_type columns
    7. Create separate queries for each component type (e.g., one for motherboards, one for cases)
    8. CRITICAL: DO NOT apply LOWER() function to ID fields (cpu_id, mb_id, gpu_id, case_id) as they are INTEGER type
    9. Verify all column names are correct for each table before using them

    Available tables and their columns:
    ```
    {table_columns}
    ```
    
    Valid tables for this query: {valid_tables}

    Your response must be a JSON object with two keys:
        reason:
            - Explain why the query is not working and how to fix it.
            - Verify that all elements mentioned in the user query are properly reflected in the queries.
            - Confirm that the queries return all the information types requested by the user.
        queries: 
            - A list of query objects, each with 'type' and 'sql' keys.
            - Example: [{{"type": "CPU", "sql": "SELECT 'CPU' AS product_type, model_name FROM cpu WHERE LOWER(model_name) LIKE '%5600%' LIMIT 5;"}}]

    Here is the output schema:
        {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation of why the query is not working and how to fix it", "type": "string"}}, "queries": {{"title": "Queries", "description": "List of queries, each with a type and SQL statement", "type": "array", "items": {{"type": "object", "properties": {{"type": {{"type": "string"}}, "sql": {{"type": "string"}}}}, "required": ["type", "sql"]}}}}}}, "required": ["reason", "queries"]}}
        
    user query: ```{user_query}```
    user meaning: ```{user_meaning}```
    selected table reason: ```{reasons}```
    previous query: ```{previous_query}```
    result: ```{result}```
    problem: ```{problem}```
    attempt_history: ```{attempt_history}```
    """)

    query_modify_chain = query_modify_prompt | gemini_llm | JsonOutputParser(pydantic_object=QueryModify)
    query_modify_info = query_modify_chain.invoke({
        "user_query": user_query,
        "user_meaning": table_info['user_meaning'],
        "reasons": table_info['reason'] + query_info['reason'],
        "previous_query": query_info['query'],
        "result": result,
        "problem": problem,
        "attempt_history": attempt_history,
        "table_columns": table_columns,
        "valid_tables": valid_tables
    })
    results = multiple_sql(query_modify_info['queries'])
    return query_modify_info, results

def check_result_agent(user_query, result):
    check_result_prompt = PromptTemplate.from_template("""
    You are an expert at analyzing query results and providing answers to user questions about computer components.
    
    Examine the query results and determine if they contain the information needed to address the user's requirements.
    
    Important guidelines:
    1. The user query may be in JSON format containing PC component specifications. Treat this as a request to find matching or compatible components.
    2. If the query is in JSON format, focus on the components that are specified (not marked as "Not specified").
    3. Provide comprehensive information about all matching components found in the results.
    4. If results only contain information about some components but not others, still provide what's available.
    5. For each component found in the results, include model names, specifications, and prices if available.
    
    Your response must be a JSON object with four keys:
        reason:
            - Explain why the results are suitable or not suitable for addressing the user's requirements.
            - Analyze the completeness and relevance of the data.
        result:
            - Indicate with TRUE or FALSE whether the data is sufficient to provide useful information.
        answer:
            - If result is TRUE: Provide a comprehensive answer listing all matching components with their details.
            - If result is FALSE: Explain that the necessary data couldn't be found and suggest possible reasons.
        issues:
            - Specific issues with the query results or missing information
            - Leave empty if no issues are found
            
    Here is the output schema:
        {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation of why the result is suitable or not suitable for addressing the user's requirements", "type": "string"}}, "result": {{"title": "Result", "description": "Whether the data is sufficient to provide useful information", "type": "boolean"}}, "answer": {{"title": "Answer", "description": "The comprehensive answer listing all matching components with their details", "type": "string"}}, "issues": {{"title": "Issues", "description": "Specific issues or missing information in the result", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["reason", "result", "answer", "issues"]}}
        
    user query: ```{user_query}```
    result: ```{result}```
    """)

    check_result_chain = check_result_prompt | explainer_llm | JsonOutputParser(pydantic_object=CheckResult)
    check_info = check_result_chain.invoke({"user_query": user_query, "result": result})
    return check_info

def summary_answer_agent(check_info, user_query):
    summary_answer_prompt = PromptTemplate.from_template("""
    당신은 결과들을 모아서 사용자이 질의를 해결하는 답변을 잘하는 전문가입니다.
    
    해당 답변들을 보고 사용자의 질의를 해결하기 위해 최선을 다해서 답변해주세요.
    1. 절대 결과에 있는 정보 외의 다른 정보를 사용하면 안됩니다.
    2. 각종 오류에 관한 말은 최대한 일반적이고 돌려서 작성해주세요.
    3. 컴퓨터에 대해 잘 모르는 사람도 이해할 수 있도록 작성해주세요.
    4. 답변은 반드시 한글로 해주세요. MUST USE KOREAN.
    
    결과는 아래와 같습니다.
    pc check result: ```{pc_check_result}```
    
    사용자 질의는 아래와 같습니다.
    user question: ```{user_question}```
    """)
    
    chain = summary_answer_prompt | flash_llm
    result = chain.invoke({"pc_check_result": check_info, "user_question": user_query})
    return result