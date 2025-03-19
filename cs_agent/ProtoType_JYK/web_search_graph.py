from typing import TypedDict, List, Dict, Any, Optional
import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from web_search_agents import recomended_search_Agent, answer_Agent, search_verification_Agent, improved_search_Agent

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int
    chat_history: str

# 웹 검색 그래프 정의
def web_search_graph(user_query: str) -> AgentState:
    # 초기 상태 설정
    state: AgentState = {
        "original_question": user_query,
        "current_search_query": "",
        "search_results": [],
        "collected_information": [],
        "answers": [],
        "is_sufficient": False,
        "suggested_queries": [],
        "final_answer": None,
        "iteration_count": 0,
        "chat_history": f"원래 질문: {user_query}\n\n"
    }

    # 최대 반복 횟수 설정 (무한 루프 방지)
    max_iterations = 3

    # 답변이 충분하거나 최대 반복 횟수에 도달할 때까지 반복
    while not state["is_sufficient"] and state["iteration_count"] < max_iterations:
        state["iteration_count"] += 1
        state["chat_history"] += f"--- 반복 {state['iteration_count']} ---\n"

        # 첫 번째 반복: 초기 검색어 생성
        if state["iteration_count"] == 1:
            rsq = recomended_search_Agent(user_query)
            state["current_search_query"] = rsq["recomended_value"]
            state["suggested_queries"].append(rsq["recomended_value"])
            state["chat_history"] += f"검색어: {rsq['recomended_value']}\n"
            state["chat_history"] += f"검색어 선택 이유: {rsq['reason']}\n"
        # 이후 반복: 검색어 개선
        else:
            previous_search_query = state["current_search_query"]
            relevance_assessment = state["answers"][-1]["verification"]["relevance_assessment"]
            re_search = improved_search_Agent(user_query, previous_search_query, relevance_assessment)
            state["current_search_query"] = re_search["recomended_value"]
            state["suggested_queries"].append(re_search["recomended_value"])
            state["chat_history"] += f"이전 검색어: {previous_search_query}\n"
            state["chat_history"] += f"개선된 검색어: {re_search['recomended_value']}\n"
            state["chat_history"] += f"검색어 개선 이유: {re_search['reason']}\n"

        # 웹 검색 및 답변 생성
        answer, web_info = answer_Agent(user_query, state["current_search_query"])
        state['collected_information'].append(web_info)

        # 답변 검증
        srv = search_verification_Agent(user_query, answer)
        state["is_sufficient"] = srv["is_sufficient"]
        state["answers"].append({"answer": answer, "verification": srv})
        
        # chat_history에 답변과 검증 결과 추가
        state["chat_history"] += f"\n답변:\n{answer}\n\n"
        state["chat_history"] += f"검증 결과: {srv['relevance_assessment']}\n"
        state["chat_history"] += f"충분한 답변인가?: {'예' if srv['is_sufficient'] else '아니오'}\n\n"

        # 답변이 충분하면 최종 답변 설정
        if state["is_sufficient"]:
            state["final_answer"] = answer
            state["chat_history"] += f"--- 최종 답변 ---\n{answer}\n"
        elif state["iteration_count"] == max_iterations:
            state["chat_history"] += f"--- 최대 반복 횟수 도달 ---\n"

    # 최대 반복 횟수 도달 시 메시지 설정
    if state["final_answer"] is None:
        state["final_answer"] = "최대 반복 횟수 동안 충분한 답변을 찾지 못했습니다."
        state["chat_history"] += f"최종 결론: {state['final_answer']}\n"

    return state

# 테스트 코드
test_user_query = "배틀그라운드 권장사양을 알고싶어. 난 CPU는 5600x 사용하고 있고 GPU는 RTX 3080사용하고 있는데 잘 돌아갈지 궁금하거든"
result = web_search_graph(test_user_query)
print("최종 답변:")
print(result["final_answer"])
print("\n채팅 기록:")
print(result["chat_history"])