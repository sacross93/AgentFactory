# agents.py
"""에이전트 정의 모듈: 검색, 검증, 쿼리 제안, 답변 생성 에이전트를 포함."""

from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
import json
import re

# LLM 초기화
llm = OllamaLLM(model="exaone3.5:32b", base_url="http://192.168.110.102:11434")
search_tool = DuckDuckGoSearchResults()

# 검색 에이전트
search_agent_prompt = PromptTemplate.from_template("""
You are a search agent tasked with finding information based on the given query.
Search Query: {input}
Use the tool: {tools}
Format:
Thought: What to do
Action: [{tool_names}]
Action Input: Tool input
Observation: Tool result
Thought: I have the results
Final Answer: Summarize results concisely
{agent_scratchpad}
""")

search_agent = create_react_agent(llm, [search_tool], search_agent_prompt)
search_executor = AgentExecutor(agent=search_agent, tools=[search_tool], verbose=True, handle_parsing_errors=True)

# 검증 에이전트
verification_prompt = PromptTemplate.from_template("""
You are a verification agent checking if collected info answers the question.
Original Question: {original_question}
Collected Information: {collected_information}
Analyze the question for:
1. Main topic
2. Specific requirements
3. Needed info
Evaluate if info is sufficient. Return JSON:
{{
    "verification_reason": "Analysis of info vs question",
    "is_sufficient": true/false
}}
""")

def verification_agent(collected_information, original_question):
    """검증 에이전트: 수집된 정보가 질문에 충분한지 평가."""
    response = llm.invoke(verification_prompt.format(
        original_question=original_question,
        collected_information="\n".join(collected_information)
    ))
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"verification_reason": "Unable to verify", "is_sufficient": False}
    except Exception as e:
        return {"verification_reason": f"Error: {e}", "is_sufficient": False}

# 쿼리 제안 에이전트
query_suggestion_prompt = PromptTemplate.from_template("""
You are a query suggestion agent.
Original Question: {original_question}
Current Search Query: {current_search_query}
Collected Information: {collected_information}
Identify missing info and suggest 1-3 English queries. Return JSON:
{{
    "suggested_queries": ["query1", "query2", "query3"]
}}
""")

def query_suggestion_agent(original_question, current_search_query, collected_information):
    """쿼리 제안 에이전트: 추가 검색 쿼리 생성."""
    response = llm.invoke(query_suggestion_prompt.format(
        original_question=original_question,
        current_search_query=current_search_query,
        collected_information="\n".join(collected_information)
    ))
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"suggested_queries": ["AMD 5600G performance"]}
    except Exception:
        return {"suggested_queries": ["AMD 5600G performance"]}

# 최종 답변 에이전트
final_answer_prompt = PromptTemplate.from_template("""
You are an AI assistant providing a comprehensive answer in Korean.
Original Question: {original_question}
Collected Information: {collected_information}
Chat History: {chat_history}
Provide a detailed, accurate answer in Korean. Address all aspects of the question.
""")

def final_answer_agent(original_question, collected_information, chat_history):
    """최종 답변 에이전트: 한국어로 답변 생성."""
    response = llm.invoke(final_answer_prompt.format(
        original_question=original_question,
        collected_information="\n".join(collected_information),
        chat_history=chat_history
    ))
    return response