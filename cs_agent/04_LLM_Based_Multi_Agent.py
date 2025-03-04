from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json

# Ollama 모델 초기화
llm = OllamaLLM(
    model="qwen2.5:32b",
    base_url="http://192.168.110.102:11434"
)

# 검색 도구 초기화
search_tool = DuckDuckGoSearchResults()


class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[str]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int
    
create_query_prompt = PromptTemplate.from_template("""
You are an expert at finding internet search terms to solve user questions.

You should do your best to help users solve problems.
Please look at the user's question and create the right search terms to solve it.

Original Question: {original_question}
""")

advanced_create_query_prompt = PromptTemplate.from_template("""
You are an expert at finding Internet search terms to solve user questions.

Original Question: {original_question}
current_search_query: {current_search_query}
""")