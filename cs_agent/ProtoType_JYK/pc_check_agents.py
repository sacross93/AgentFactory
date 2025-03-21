from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from pc_check_func import *

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

## 사용자 질의를 보고 해당 질의에 맞는 Table 정보를 추출하는 Json 구조 (pydantic)
class TableInfo(BaseModel):
    user_meaning: str = Field(description="Interpretation of the user's query intent")
    reason: str = Field(description="Reason for selecting these tables")
    table_names: List[str] = Field(description="Table names needed to resolve the user query")

## 테이블 column을 보고 사용자 질의에 맞는 Query 작성하는 Json 구조(pydantic)
class QueryInfo(BaseModel):
    reason: str = Field(description="Explain why you selected these columns and why you wrote the query this way.")
    selected_columns: List[str] = Field(description="Select columns needed to address the user query.")
    query: str = Field(description="Write a SQL query that addresses the user's question.")
    
## Query가 제대로 동작하지 않는 경우 Json 구조(pydantic)
class QueryModify(BaseModel):
    reason: str = Field(description="Explain why the query is not working and how to fix it.")
    queries: List[Dict[str, str]] = Field(description="List of queries, each with a type and SQL statement.")
    
## 결과가 잘 나왔는지 확인하는 Json 구조(pydantic)
class CheckResult(BaseModel):
    reason: str = Field(description="Explanation of why the result is suitable or not suitable for answering the user's question.")
    result: bool = Field(description="Whether this data is sufficient to answer the user's question.")
    answer: str = Field(description="The answer to the user's query based on the provided data.")

# 사용자 질문을 해결할 수 있게 테이블 이름을 추출하는 Agent
def table_abstract_agent(user_query):
    table_abstract_prompt = PromptTemplate.from_template("""
    You are an expert who analyzes user queries and extracts table names needed to resolve those queries.
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    Important guidelines:
    1. When dealing with compatibility queries, don't select only the compatibility table. For example, if checking CPU and motherboard compatibility, you need to select the CPU table, motherboard table, AND the CPU-motherboard compatibility table.
    2. Compatibility tables typically only contain IDs and compatibility status, not detailed product information like names or specifications.

    Your response must be a JSON object with three keys:
        user_meaning:
            - Interpretation of the user's query intent
        reason:
            - Explanation of why these tables were selected for the query
        table_names:
            - Tables selected to address the user's query
        
    Here is the output schema:
    ```
    {{"properties":{{"user_meaning": {{"title": "User Meaning", "description": "Interpretation of the user's query intent", "type": "string"}}, "reason": {{"title": "Reason", "description": "Explanation of why these tables were selected for the query", "type": "string"}}, "table_names": {{"title": "Table Names", "description": "Table names needed to resolve the user query", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["user_meaning", "reason", "table_names"]}}
    ```

    Based on this guidance, analyze the user's actual query and provide your response in the specified JSON format.

    Tables: ```{tables}```
    User query: ```{input}```
    """)

    tables = get_db_table()

    table_abstract_chain = table_abstract_prompt | explainer_llm | JsonOutputParser(pydantic_object=TableInfo)

    table_info = table_abstract_chain.invoke({"input": user_query, "tables": tables})
    return table_info


# 테이블 column을 보고 사용자 질의에 맞는 Query 작성하는 Agent
def query_optimize_agent(user_query, table_info):
    table_columns = get_db_samples(table_info['table_names'])
    query_optimize_prompt = PromptTemplate.from_template("""
    You are an SQL query writing and optimization expert.

    Examine the user query and the table names with their columns to create a query that can resolve the user's question.
    You must create a single SQL query that addresses all aspects of the user's question.

    Your response must be a JSON object with three keys:
        reason:
            - Explain why you selected these columns and why you wrote the query this way.
        selected_columns:
            - Select columns needed to address the user query.
        query:
            - Write a SQL query that addresses the user's question.
            - Do not use UNION ALL operator.

    Here is the output schema:
    ```
    {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation of why these columns were selected and how the query was constructed", "type": "string"}}, "selected_columns": {{"title": "Selected Columns", "description": "Columns selected to address the user query", "type": "array", "items": {{"type": "string"}}}}, "query": {{"title": "Query", "description": "SQL query to resolve the user's question", "type": "string"}}}}, "required": ["reason", "selected_columns", "query"]}}
    ```

    table columns: ```{table_columns}```
    user query: ```{user_query}```
    """)

    query_optimize_chain = query_optimize_prompt | explainer_llm | JsonOutputParser(pydantic_object=QueryInfo)

    query_info = query_optimize_chain.invoke({"table_columns": table_columns, "user_query": user_query})
    fixed_query = fix_union_query(query_info['query'])
    result = sql(fixed_query)
    
    return query_info, result

# Query 수정 Agent
def query_modify_agent(user_query, table_info, query_info, result):
    query_modify_prompt = PromptTemplate.from_template("""
    You are an SQL query writing and optimization expert.

    Examine the previous query that didn't produce the expected results and fix it to properly address the user's question.
    Instead of creating a single complex query with UNION ALL, create separate queries for each product type.

    When analyzing why the query might not be working, consider common user input variations:
    0. You must create queries that resolve the user's question.
    1. Users may refer to product names with different capitalization
    2. Users might use different spacing in product names
    3. Users could reverse the order of components in product names
    4. Users might abbreviate or use partial product names (using LIKE operator in SQL can help with this)

    Important considerations for your queries:
    1. Use IN operator instead of = when comparing with subqueries that might return multiple rows
    2. Add a column to identify product types in the results
    3. Create separate queries for each product type (e.g., one for motherboards, one for cases)
    4. Keep queries simple and focused on one product type each
    5. Verify table names are correct based on the database schema
    6. Use appropriate compatibility tables for each component relationship
    7. Simplify join structures when possible to avoid unnecessary complexity
    8. Ensure compatibility conditions directly check relationships between components, not just their existence
    9. CRITICAL: DO NOT apply LOWER() function to ID fields (cpu_id, mb_id, gpu_id, case_id) as they are INTEGER type. 
    - CORRECT: WHERE cpu_id IN (SELECT cpu_id FROM cpu WHERE LOWER(model_name) LIKE '%5600g%')
    - INCORRECT: WHERE LOWER(cpu_id) IN (SELECT cpu_id FROM cpu WHERE LOWER(model_name) LIKE '%5600g%')
    10. When searching for specific product models, use broader search terms (e.g., '3060' instead of '3060ti') to account for variations
    11. Only use tables that are mentioned in the previous query or in the table_columns information

    Your response must be a JSON object with two keys:
        reason:
            - Explain why the query is not working and how to fix it.
            - Verify that all elements mentioned in the user query are properly reflected in the queries.
            - Confirm that the queries return all the information types requested by the user.
        queries: 
            - A list of query objects, each with 'type' and 'sql' keys.
            - Example: [{{"type": "Motherboard", "sql": "SELECT 'Motherboard' AS product_type, m.model_name FROM motherboard AS m JOIN cpu_mb_compatibility AS cmb ON m.mb_id = cmb.mb_id WHERE cmb.cpu_id IN (SELECT cpu_id FROM cpu WHERE LOWER(model_name) LIKE '%5600g%') LIMIT 5;"}}]

    Here is the output schema:
    ```
    {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation of why the query is not working and how to fix it", "type": "string"}}, "queries": {{"title": "Queries", "description": "List of queries, each with a type and SQL statement", "type": "array", "items": {{"type": "object", "properties": {{"type": {{"type": "string"}}, "sql": {{"type": "string"}}}}, "required": ["type", "sql"]}}}}}}, "required": ["reason", "queries"]}}
    ```

    user query: ```{user_query}```
    user meaning: ```{user_meaning}```
    selected table reason: ```{reasons}```
    previous query: ```{previous_query}```
    result: ```{result}```
    """)

    query_modify_chain = query_modify_prompt | llm | JsonOutputParser(pydantic_object=QueryModify)

    query_modify_info = query_modify_chain.invoke({"user_query":user_query, "user_meaning":table_info['user_meaning'], "reasons":table_info['reason'] + query_info['reason'], "previous_query":query_info['query'], "result":result})
    results = multiple_sql(query_modify_info['queries'])
    
    return query_modify_info, results

# 결과가 잘 나왔는지 확인하는 Agent
def check_result_agent(user_query, result):
    check_result_prompt = PromptTemplate.from_template("""
    You are an expert at analyzing query results and providing answers to user questions.
    
    Examine the query results and determine if they contain the information needed to answer the user's question.
    
    Your response must be a JSON object with three keys:
        reason:
            - Explain why the results are suitable or not suitable for answering the user's question.
            - Analyze the completeness and relevance of the data.
        result:
            - Indicate with TRUE or FALSE whether the data is sufficient to answer the user's question.
        answer:
            - If result is TRUE: Provide a comprehensive answer to the user's question based on the data.
            - If result is FALSE: Explain that the necessary data couldn't be found and suggest possible reasons.
            
    Here is the output schema:
    ```
    {{"properties":{{"reason": {{"title": "Reason", "description": "Explanation of why the result is suitable or not suitable for answering the user's question", "type": "string"}}, "result": {{"title": "Result", "description": "Whether the data is sufficient to answer the user's question", "type": "boolean"}}, "answer": {{"title": "Answer", "description": "The answer to the user's query based on the provided data", "type": "string"}}}}, "required": ["reason", "result", "answer"]}}
    ```

    user query: ```{user_query}```
    result: ```{result}```
    """)

    check_result_chain = check_result_prompt | explainer_llm | JsonOutputParser(pydantic_object=CheckResult)
    check_result_info = check_result_chain.invoke({"user_query":user_query, "result":result})
    return check_result_info



# 테스트 코드
# user_query = "cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘"
# table_info = table_abstract_agent(user_query)
# query_info, result = query_optimize_agent(user_query, table_info)
# query_modify_info, results = query_modify_agent(user_query, table_info, query_info, result)
# final_answer = check_result_agent(user_query, results)

# print(query_modify_info)
# print(results)
# print(final_answer)