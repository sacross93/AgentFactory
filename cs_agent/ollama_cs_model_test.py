from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json

# Ollama 모델 초기화
llm = OllamaLLM(
    model="exaone3.5:32b",
    base_url="http://192.168.110.102:11434"
)

# 검색 도구 초기화
search_tool = DuckDuckGoSearchResults()

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[str]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int  # 반복 횟수를 추적하기 위한 필드 추가

# 검색 에이전트 (Agent 1)
search_agent_prompt = PromptTemplate.from_template("""
You are a search agent. Your task is to search for information based on the given query.

Search Query: {input}

Use the following tool to search for information:
{tools}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
Thought: I now have the search results
Final Answer: Summarize the search results in a clear and concise way

{agent_scratchpad}
""")

search_agent = create_react_agent(llm, [search_tool], search_agent_prompt)
search_executor = AgentExecutor(
    agent=search_agent, 
    tools=[search_tool], 
    verbose=True,
    handle_parsing_errors=True  # 파싱 오류 처리 옵션 추가
)

# 검증 에이전트 (Agent 2)
verification_prompt = PromptTemplate.from_template("""
You are a verification agent. Your task is to determine if the collected information is sufficient to answer the original question.

Original Question: {original_question}
Collected Information:
{collected_information}

Analyze if the collected information is sufficient to provide a complete answer to the original question.
Return your analysis in JSON format:
{{
    "is_sufficient": true/false,
    "reasoning": "Your reasoning here"
}}
""")

def verification_agent(state):
    collected_info = "\n".join(state["collected_information"])
    response = llm.invoke(verification_prompt.format(
        original_question=state["original_question"],
        collected_information=collected_info
    ))
    
    try:
        result = json.loads(response)
        return {"is_sufficient": result["is_sufficient"]}
    except:
        # 파싱 오류 시 기본값 반환
        return {"is_sufficient": False}

# 쿼리 제안 에이전트 (Agent 3)
query_suggestion_prompt = PromptTemplate.from_template("""
You are a query suggestion agent. Your task is to suggest new search queries to find more information.

Original Question: {original_question}
Current Search Query: {current_search_query}
Collected Information:
{collected_information}

What we still need to know:
1. What aspects of the original question are not yet answered?
2. What specific information is missing?

Suggest 1-3 new search queries that would help gather the missing information.
Return your suggestions in JSON format:
{{
    "suggested_queries": ["query1", "query2", "query3"]
}}
""")

def query_suggestion_agent(state):
    collected_info = "\n".join(state["collected_information"])
    response = llm.invoke(query_suggestion_prompt.format(
        original_question=state["original_question"],
        current_search_query=state["current_search_query"],
        collected_information=collected_info
    ))
    
    try:
        result = json.loads(response)
        return {"suggested_queries": result["suggested_queries"]}
    except:
        # 파싱 오류 시 기본값 반환
        return {"suggested_queries": [f"more about {state['original_question']}"]}

# 최종 답변 생성 에이전트
final_answer_prompt = PromptTemplate.from_template("""
You are a helpful korean assistant. Your task is to provide a comprehensive answer to the original question based on the collected information.

Original Question: {original_question}
Collected Information:
{collected_information}

MUST Provide a comprehensive answer in Korean. Make sure to address all aspects of the original question.
""")

def final_answer_agent(state):
    collected_info = "\n".join(state["collected_information"])
    response = llm.invoke(final_answer_prompt.format(
        original_question=state["original_question"],
        collected_information=collected_info
    ))
    
    return {"final_answer": response}

# 검색 노드
def search_node(state):
    # 반복 횟수 증가
    state["iteration_count"] += 1
    
    result = search_executor.invoke({
        "input": state["current_search_query"],
        "tools": [search_tool],
        "tool_names": ["DuckDuckGoSearchResults"],
        "agent_scratchpad": ""
    })
    
    search_results = result["output"]
    state["search_results"].append(search_results)
    state["collected_information"].append(f"Search for '{state['current_search_query']}': {search_results}")
    
    return state

# 다음 쿼리 선택 노드
def select_next_query(state):
    if state["suggested_queries"]:
        state["current_search_query"] = state["suggested_queries"][0]
        state["suggested_queries"] = state["suggested_queries"][1:]
    return state

# 라우터 함수
def router(state):
    # 최대 5번 반복 제한 추가
    if state["iteration_count"] >= 5:
        return "generate_answer"
    
    if state["is_sufficient"]:
        return "generate_answer"
    elif state["suggested_queries"]:
        return "select_next_query"
    else:
        return "suggest_queries"

# 그래프 정의
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("search", search_node)
workflow.add_node("verify", verification_agent)
workflow.add_node("suggest_queries", query_suggestion_agent)
workflow.add_node("select_next_query", select_next_query)
workflow.add_node("generate_answer", final_answer_agent)

# 엣지 추가
workflow.add_edge("search", "verify")
# 조건부 엣지 추가 - verify 노드에서 라우터 함수를 사용하여 다음 노드 결정
workflow.add_conditional_edges(
    "verify",
    router,
    {
        "generate_answer": "generate_answer",
        "select_next_query": "select_next_query",
        "suggest_queries": "suggest_queries"
    }
)
workflow.add_edge("suggest_queries", "select_next_query")
workflow.add_edge("select_next_query", "search")
workflow.add_edge("generate_answer", END)

# 시작점 추가
workflow.set_entry_point("search")

# 그래프 컴파일
graph = workflow.compile()

# 실행
def run_agent_workflow(question):
    initial_state = {
        "original_question": question,
        "current_search_query": question,
        "search_results": [],
        "collected_information": [],
        "is_sufficient": False,
        "suggested_queries": [],
        "final_answer": None,
        "iteration_count": 0  # 반복 횟수 초기화
    }
    
    result = graph.invoke(initial_state)
    return result["final_answer"]

# 테스트
if __name__ == "__main__":
    # question = "코어i9-12900KF와 ASUS PRIME A520M-A II 대원CTS 제품의 호환성을 체크해줘"
    # question = "Stable diffusion을 실행하기 위해 필요한 사양을 알려주고, 실행되는 컴퓨터를 구매하려면 얼마정도 써야하는지 알려줘"
    question = "리그오브레전드를 하려고 하는데 필요한 컴퓨터 사양을 알려줘 나는 GTX 1660ti랑 CPU는 i5-12400f를 사용하고 있어 RAM은 16GB야"
    # question = "제플몰의 운영시간을 알려줘"
    answer = run_agent_workflow(question)
    print("\n최종 답변:")
    print(answer)
