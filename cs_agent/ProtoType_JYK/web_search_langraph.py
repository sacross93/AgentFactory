from typing import TypedDict, List, Dict, Any, Optional
import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from web_search_agents import recomended_search_Agent, answer_Agent, search_verification_Agent, improved_search_Agent
from langgraph.graph import StateGraph, END

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    answers: List[Dict[str, Any]]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int
    chat_history: str

# 검색어 생성 노드
def generate_search_query(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 검색어 생성 에이전트 (반복 {state['iteration_count'] + 1})")
    print(f"{'='*50}")
    
    user_query = state["original_question"]
    
    # 첫 번째 반복: 초기 검색어 생성
    if state["iteration_count"] == 0:
        print("[작업] 초기 검색어 생성 중...")
        rsq = recomended_search_Agent(user_query)
        search_query = rsq["recomended_value"]
        reason = rsq["reason"]
        print(f"[결과] 검색어: {search_query}")
        print(f"[결과] 이유: {reason}")
        
        # 채팅 기록 업데이트
        chat_update = f"검색어: {search_query}\n"
        chat_update += f"검색어 선택 이유: {reason}\n"
    
    # 이후 반복: 검색어 개선
    else:
        print("[작업] 검색어 개선 중...")
        previous_search_query = state["current_search_query"]
        relevance_assessment = state["answers"][-1]["verification"]["relevance_assessment"]
        re_search = improved_search_Agent(user_query, previous_search_query, relevance_assessment)
        search_query = re_search["recomended_value"]
        reason = re_search["reason"]
        print(f"[결과] 이전 검색어: {previous_search_query}")
        print(f"[결과] 개선된 검색어: {search_query}")
        print(f"[결과] 이유: {reason}")
        
        # 채팅 기록 업데이트
        chat_update = f"이전 검색어: {previous_search_query}\n"
        chat_update += f"개선된 검색어: {search_query}\n"
        chat_update += f"검색어 개선 이유: {reason}\n"
    
    # 상태 업데이트
    return {
        **state,
        "current_search_query": search_query,
        "suggested_queries": state["suggested_queries"] + [search_query],
        "chat_history": state["chat_history"] + f"--- 반복 {state['iteration_count'] + 1} ---\n" + chat_update
    }

# 검색 및 답변 생성 노드
def search_and_answer(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 웹 검색 및 답변 생성 에이전트 (반복 {state['iteration_count'] + 1})")
    print(f"{'='*50}")
    print(f"[작업] 검색어 '{state['current_search_query']}'로 웹 검색 중...")
    
    user_query = state["original_question"]
    search_query = state["current_search_query"]
    
    # 검색 및 답변 생성
    answer, web_info = answer_Agent(user_query, search_query)
    
    # 답변 일부 출력 (너무 길 수 있으므로 처음 200자만)
    preview = answer[:200] + "..." if len(answer) > 200 else answer
    print(f"[결과] 답변 생성 완료 (미리보기): {preview}")
    
    # 상태 업데이트
    return {
        **state,
        "search_results": state["search_results"] + [web_info],
        "chat_history": state["chat_history"] + f"\n답변:\n{answer}\n\n",
        "answers": state["answers"] + [{"answer": answer, "verification": None}]
    }

# 답변 검증 노드
def verify_answer(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 답변 검증 에이전트 (반복 {state['iteration_count'] + 1})")
    print(f"{'='*50}")
    print("[작업] 답변의 충분성 검증 중...")
    
    user_query = state["original_question"]
    current_answer = state["answers"][-1]["answer"]
    
    # 검증 수행
    srv = search_verification_Agent(user_query, current_answer)
    
    print(f"[결과] 검증 결과: {srv['relevance_assessment']}")
    print(f"[결과] 충분한 답변인가?: {'예' if srv['is_sufficient'] else '아니오'}")
    
    # 현재 답변의 검증 정보 업데이트
    updated_answers = state["answers"].copy()
    updated_answers[-1]["verification"] = srv
    
    # 채팅 기록 업데이트
    chat_update = f"검증 결과: {srv['relevance_assessment']}\n"
    chat_update += f"충분한 답변인가?: {'예' if srv['is_sufficient'] else '아니오'}\n\n"
    
    # 최종 답변 설정 여부 결정
    final_answer = current_answer if srv["is_sufficient"] else state["final_answer"]
    
    # 최종 답변이 설정되었다면 채팅 기록에 추가
    if srv["is_sufficient"]:
        print(f"\n[완료] 충분한 답변 찾음 - 프로세스 종료")
        chat_update += f"--- 최종 답변 ---\n{current_answer}\n"
    
    # 반복 횟수 업데이트 및 최대 반복 횟수 검사
    iteration_count = state["iteration_count"] + 1
    if iteration_count >= 3 and not srv["is_sufficient"]:
        print(f"\n[완료] 최대 반복 횟수 도달 - 프로세스 종료")
        final_answer = "최대 반복 횟수 동안 충분한 답변을 찾지 못했습니다."
        chat_update += f"--- 최대 반복 횟수 도달 ---\n최종 결론: {final_answer}\n"
    
    # 상태 업데이트
    return {
        **state,
        "answers": updated_answers,
        "is_sufficient": srv["is_sufficient"],
        "final_answer": final_answer,
        "iteration_count": iteration_count,
        "chat_history": state["chat_history"] + chat_update
    }

# 상태에 따른 다음 단계 결정
def should_continue(state: AgentState) -> str:
    # 답변이 충분하거나 최대 반복 횟수에 도달하면 종료
    if state["is_sufficient"] or state["iteration_count"] >= 3:
        decision = "end"
    else:
        decision = "continue"
        
    print(f"\n{'='*50}")
    print(f"[결정] 다음 단계: {'종료' if decision == 'end' else '계속 검색'}")
    print(f"{'='*50}")
    return decision

# 웹 검색 그래프 정의
def web_search_langraph(user_query: str) -> AgentState:
    print(f"\n{'*'*60}")
    print(f"시작: 웹 검색 그래프 (질문: {user_query})")
    print(f"{'*'*60}")
    
    # 초기 상태 설정
    state: AgentState = {
        "original_question": user_query,
        "current_search_query": "",
        "search_results": [],
        "answers": [],
        "is_sufficient": False,
        "suggested_queries": [],
        "final_answer": None,
        "iteration_count": 0,
        "chat_history": f"원래 질문: {user_query}\n\n"
    }
    
    # 그래프 정의
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("generate_query", generate_search_query)
    workflow.add_node("search_answer", search_and_answer)
    workflow.add_node("verify", verify_answer)
    
    # 엣지 추가
    workflow.add_edge("generate_query", "search_answer")
    workflow.add_edge("search_answer", "verify")
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "verify",
        should_continue,
        {
            "continue": "generate_query",
            "end": END
        }
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("generate_query")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    print("그래프 컴파일 완료. 실행 시작...")
    
    # 그래프 실행
    result = app.invoke(state)
    
    print(f"\n{'*'*60}")
    print(f"완료: 웹 검색 그래프")
    print(f"{'*'*60}")
    
    return result

# 실행 함수
def run_web_search(user_query: str) -> str:
    result = web_search_langraph(user_query)
    return result["final_answer"] or "답변을 생성하지 못했습니다."

# 테스트
if __name__ == "__main__":
    # user_query = "배틀그라운드 권장사양을 알고싶어. 난 CPU는 5600x 사용하고 있고 GPU는 RTX 3080사용하고 있는데 잘 돌아갈지 궁금하거든"
    user_query = "배틀그라운드 권장사양을 알고싶어. 권장사양에 맞게 PC 구성을 하려고 하거든"
    result = web_search_langraph(user_query)
    print("최종 답변:")
    print(result["final_answer"])
    print("\n채팅 기록:")
    print(result["chat_history"]) 