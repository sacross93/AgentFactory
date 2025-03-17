from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Dict, Any, Optional
import json
import re

# LLM 초기화
llm = OllamaLLM(model="gemma3:27b", base_url="http://192.168.110.102:11434")
search_tool = DuckDuckGoSearchResults(output_format="list")

search_tool.invoke("배틀그라운드 권장사양")

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[str]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int
    chat_history: str
    
advanced_search_prompt = PromptTemplate.from_template("""

""")
    
    
# 프롬프트 정의
search_agent_prompt = PromptTemplate.from_template("""
You are a search agent. Your task is to search for information based on the given query.

Search Query: {input}

Use the following tool to search for information:
{tools}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool (just the search query text, no additional formatting)
Observation: the result of the tool
Thought: I now have the search results
Final Answer: Summarize the search results in a clear and concise way

{agent_scratchpad}
""")
