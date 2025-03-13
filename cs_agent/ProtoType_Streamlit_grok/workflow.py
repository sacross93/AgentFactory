# workflow.py
"""LangGraph를 사용한 워크플로우 정의 모듈."""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from agents import search_executor, verification_agent, query_suggestion_agent, final_answer_agent

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

def search_node(state: AgentState) -> AgentState:
    """검색 노드: 쿼리로 정보 검색."""
    state["iteration_count"] += 1
    result = search_executor.invoke({
        "input": state["current_search_query"],
        "tools": [search_executor.tools[0]],
        "tool_names": ["DuckDuckGoSearchResults"],
        "agent_scratchpad": ""
    })
    search_results = result["output"]
    state["search_results"].append(search_results)
    state["collected_information"].append(f"Search for '{state['current_search_query']}': {search_results}")
    return state

def verify_node(state: AgentState) -> AgentState:
    """검증 노드: 정보 충분성 확인."""
    result = verification_agent(state["collected_information"], state["original_question"])
    state["is_sufficient"] = result["is_sufficient"]
    return state

def suggest_queries_node(state: AgentState) -> AgentState:
    """쿼리 제안 노드: 추가 쿼리 생성."""
    result = query_suggestion_agent(state["original_question"], state["current_search_query"], state["collected_information"])
    state["suggested_queries"] = result["suggested_queries"]
    return state

def select_next_query_node(state: AgentState) -> AgentState:
    """다음 쿼리 선택 노드."""
    if state["suggested_queries"]:
        state["current_search_query"] = state["suggested_queries"][0]
        state["suggested_queries"] = state["suggested_queries"][1:]
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    """답변 생성 노드."""
    state["final_answer"] = final_answer_agent(state["original_question"], state["collected_information"], state["chat_history"])
    return state

def router(state: AgentState) -> str:
    """라우터: 다음 단계 결정."""
    if state["iteration_count"] >= 5 or state["is_sufficient"]:
        return "generate_answer"
    elif state["suggested_queries"]:
        return "select_next_query"
    return "suggest_queries"

# 워크플로우 정의
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("verify", verify_node)
workflow.add_node("suggest_queries", suggest_queries_node)
workflow.add_node("select_next_query", select_next_query_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_edge("search", "verify")
workflow.add_conditional_edges("verify", router, {
    "generate_answer": "generate_answer",
    "select_next_query": "select_next_query",
    "suggest_queries": "suggest_queries"
})
workflow.add_edge("suggest_queries", "select_next_query")
workflow.add_edge("select_next_query", "search")
workflow.add_edge("generate_answer", END)
workflow.set_entry_point("search")

graph = workflow.compile()