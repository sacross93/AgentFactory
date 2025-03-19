from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from web_search_engine import enhanced_search

# LLM 초기화
llm = OllamaLLM(model="gemma3:27b", base_url="http://192.168.110.102:11434")
# llm = OllamaLLM(model="qwen2.5:32b", base_url="http://192.168.110.102:11434")

## 검색어 추천 Json 구조 (pydantic)
class recomended_search_query(BaseModel):
    reason: str = Field(description="Explain briefly (one sentence) why the suggested search query best fits the user's original question.")
    recomended_value: str = Field(description="Provide exactly one recommended internet search term.")

## 웹 검색 정보 검증 Json 구조 (pydantic)
class SearchResultVerification(BaseModel):
    relevance_assessment: str = Field(description="Assessment: Briefly explain how helpful the search results are for the user's original question.")
    is_sufficient: bool = Field(description="Sufficiency: Whether the search results are sufficient to answer the user's question (True/False)")

# 검색어 추천 Agent
def recomended_search_Agent(user_query):
    rsq_output_parser = JsonOutputParser(pydantic_object=recomended_search_query)
        
    advanced_search_prompt = PromptTemplate.from_template("""
    You are an expert who analyzes user queries and provides the most suitable single internet search query to resolve them. 
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    Your response must be a JSON object with two keys:
        reasoning:
            - Provide one brief sentence explaining why your recommended search query best fits the user's original question.
        recomended_value:
            - Suggest exactly one concise and clear internet search term explicitly based on the user's query.
            - DO NOT include any additional numerical values, specifications, or details that are not explicitly stated in the user's question.
        
    Here is the output schema:
    ```
    {{"properties":{{"reason": {{"title": "Reason", "description": "Explain briefly (one sentence) why the suggested search query best fits the user's original question.", "type": "string"}}, "recomended_value": {{"title": "Recomended Value", "description": "Provide exactly one recommended internet search term.", "type": "string"}}}}, "required": ["reason", "recomended_value"]}}
    ```

    Based on this guidance, analyze the user's actual query and provide your response in the specified JSON format.

    User query: ```{input}```
    """)

    chain = advanced_search_prompt | llm | rsq_output_parser
    rsq = chain.invoke({"input": user_query})
    return rsq

# 검색 답변 Agent
def answer_Agent(user_query, web_infor):
    web_infor = enhanced_search(web_infor)

    answer_prompt = PromptTemplate.from_template("""
    You are an expert who provides accurate and helpful answers to user questions.
    You must answer the user's question based on the provided web search results.

    Please follow these instructions carefully:
    1. Use only information found in the web search results.
    2. Do not make assumptions about information not present in the search results.
    3. Structure your answer as follows:
    - First, create a "Reference Information:" section that summarizes the key information from the web search results.
    - Then, provide a "Final Answer:" section with a direct answer to the user's question based on the reference information.
        - If the user mentions specific hardware, you MUST directly compare it to the recommended specifications and clearly state whether it meets or exceeds requirements.
        - For example, if the user mentions "CPU: 5600x and GPU: RTX 3080", explicitly state how these components compare to the game's requirements.
    - Finally, include a "Conclusion:" section at the very end that provides a clear, direct answer to the user's original question. This should be a concise summary that directly addresses what the user wanted to know.
    4. If the search results do not contain relevant information, honestly respond with "I could not find relevant information in the provided search results."

    Web search results: ```{web_information}```

    User question: ```{input}```
    
    Please make sure MUST answer in korean.
    """)

    answer = llm.invoke(answer_prompt.format(input=user_query, web_information=web_infor))
    
    return answer, web_infor

# 검색 검증 Agent
def search_verification_Agent(user_query, answer):
    srv_output_parser = JsonOutputParser(pydantic_object=SearchResultVerification)

    ## 프롬프트 정의
    search_agent_prompt = PromptTemplate.from_template("""
    You are an expert who evaluates whether the generated answer sufficiently addresses the user's original query.
    The output should be formatted as a JSON instance that strictly conforms to the JSON schema below.

    Your response must be a JSON object with exactly two keys:
        relevance_assessment:
            - If the answer is sufficient (is_sufficient=true), explain why it successfully addresses the user's question.
            - If the answer is insufficient (is_sufficient=false), explain specifically what information is missing or inadequate.
            - Keep your explanation brief and focused on one or two key points.
        is_sufficient:
            - Clearly state true if the provided answer is sufficient to address the user's query, or false otherwise.

    Here is the exact output schema your response must follow:
    ```
    {{"properties": {{"relevance_assessment": {{"title": "Assessment", "description": "Briefly explain why the answer is sufficient or insufficient for the user's original question.", "type": "string"}}, "is_sufficient": {{"title": "Is Sufficient", "description": "Whether the answer is sufficient to address the user's question (true/false).", "type": "boolean"}}}}, "required": ["relevance_assessment", "is_sufficient"]}}
    ```

    Based on this guidance, analyze the information and provide your response in the specified JSON format.

    User query: ```{input}```
    Generated answer: ```{answer}```
    """)

    chain = search_agent_prompt | llm | srv_output_parser
    srv = chain.invoke({"input": user_query, "answer": answer})
    
    return srv


# 웹 검색어 더 개선
def improved_search_Agent(user_query, previous_search_query, relevance_assessment):
    re_output_parser = JsonOutputParser(pydantic_object=recomended_search_query)
    re_search_prompt = PromptTemplate.from_template("""
    You are an expert at refining search queries when initial search results were insufficient to answer a user's question.
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    The user's original question and a previous search query were used, but the results did not fully address the user's needs. Your task is to create an improved search query that will yield more relevant information.

    Please follow these guidelines:
    1. Analyze why the previous search query might have been insufficient
    2. Create a more specific, targeted search query that addresses the core of the user's question
    3. Focus on extracting key technical terms or specifications from the user's question
    4. Avoid overly broad terms that might dilute search results
    5. Provide exactly ONE improved search query (no lists or multiple options)

    Your response must be a JSON object with two keys:
        reasoning:
            - Provide one brief sentence explaining why your recommended search query best fits the user's original question.
        recomended_value:
            - Suggest exactly one concise and clear internet search term explicitly based on the user's query.
            - DO NOT include any additional numerical values, specifications, or details that are not explicitly stated in the user's question.
        
    Here is the output schema:
    ```
    {{"properties":{{"reason": {{"title": "Reason", "description": "Explain briefly (one sentence) why the suggested search query best fits the user's original question.", "type": "string"}}, "recomended_value": {{"title": "Recomended Value", "description": "Provide exactly one recommended internet search term.", "type": "string"}}}}, "required": ["reason", "recomended_value"]}}
    ```

    User's original question: ```{input}```
    Previous search query: ```{previous_search_query}```
    Reason for insufficiency: ```{relevance_assessment}```

    Based on this guidance, analyze the user's actual query and provide your response in the specified JSON format.
    """)

    chain = re_search_prompt | llm | re_output_parser
    re_search = chain.invoke({"input": user_query, "previous_search_query": previous_search_query, "relevance_assessment": relevance_assessment})
    
    return re_search