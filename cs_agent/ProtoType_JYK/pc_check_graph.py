import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from pc_check_agents import *
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# graph 상태 정의
class AgentState(TypedDict):
    original_question: str
    user_meaning: str
    used_tables: List[str]
    selected_columns: List[str]
    queries: List[Dict[str, str]]
    previous_query: str
    results: Any
    iteration: int
    answer: str
    finished: bool
    
# 테이블 정보 추출 노드
def extract_tables(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 테이블 정보 추출 에이전트")
    print(f"{'='*50}")
    print("[작업] 사용자 질의에서 필요한 테이블 식별 중...")
    
    user_query = state["original_question"]
    table_info = table_abstract_agent(user_query)
    
    print(f"[결과] 사용자 의도: {table_info['user_meaning']}")
    print(f"[결과] 필요한 테이블: {', '.join(table_info['table_names'])}")
    
    return {
        **state,
        "user_meaning": table_info["user_meaning"],
        "used_tables": table_info["table_names"]
    }

# 쿼리 최적화 노드
def optimize_query(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 쿼리 최적화 에이전트")
    print(f"{'='*50}")
    print("[작업] 최적의 SQL 쿼리 생성 중...")
    
    user_query = state["original_question"]
    table_info = {
        "user_meaning": state["user_meaning"],
        "table_names": state["used_tables"],
        "reason": ""
    }
    
    try:
        query_info, result = query_optimize_agent(user_query, table_info)
        
        # 결과만 간단히 출력 (쿼리와 컬럼 출력 제거)
        print(f"[결과] 쿼리 생성 완료")
        print(f"[결과] 쿼리 결과 행 수: {len(result) if result else 0}")
        
        return {
            **state,
            "selected_columns": query_info["selected_columns"],
            "previous_query": query_info["query"],
            "results": result
        }
    except Exception as e:
        print(f"[오류] 쿼리 최적화 중 오류 발생: {e}")
        return {
            **state,
            "selected_columns": [],
            "previous_query": "",
            "results": []
        }

# 결과 확인 노드
def check_results(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 결과 확인 에이전트 (반복 {state['iteration']})")
    print(f"{'='*50}")
    print("[작업] 쿼리 결과 평가 중...")
    
    user_query = state["original_question"]
    check_info = check_result_agent(user_query, state["results"])
    
    print(f"[결과] 결과 적합성: {'적합' if check_info['result'] else '부적합'}")
    print(f"[결과] 평가 이유: {check_info['reason']}")
    if check_info['result']:
        print(f"[결과] 최종 답변 생성 완료")
    
    return {
        **state,
        "answer": check_info["answer"],
        "finished": check_info["result"]
    }

# 쿼리 수정 노드
def modify_query(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 쿼리 수정 에이전트 (반복 {state['iteration']})")
    print(f"{'='*50}")
    print("[작업] 쿼리 개선 중...")
    
    user_query = state["original_question"]
    table_info = {
        "user_meaning": state["user_meaning"],
        "table_names": state["used_tables"],
        "reason": ""
    }
    query_info = {
        "reason": "",
        "selected_columns": state["selected_columns"],
        "query": state["previous_query"]
    }
    
    try:
        query_modify_info, results = query_modify_agent(user_query, table_info, query_info, state["results"])
        
        # 간략한 결과만 출력 (쿼리 내용 출력 제거)
        print(f"[결과] 수정 이유: {query_modify_info['reason']}")
        print(f"[결과] 생성된 쿼리 수: {len(query_modify_info['queries'])}")
        print(f"[결과] 쿼리 결과 행 수: {len(results) if results else 0}")
        
        # 수정된 쿼리 정보로 상태 업데이트
        return {
            **state,
            "queries": query_modify_info["queries"],
            "results": results,
            "iteration": state["iteration"] + 1
        }
    except Exception as e:
        print(f"[오류] 쿼리 수정 중 오류 발생: {e}")
        return {
            **state,
            "iteration": state["iteration"] + 1
        }

# 조건 라우터: 결과가 만족스러운지 확인
def should_continue(state: AgentState) -> str:
    # 결과가 만족스럽거나 최대 반복 횟수에 도달하면 종료
    if state["finished"] or state["iteration"] >= 5:
        decision = "end"
    else:
        decision = "modify"
    
    print(f"\n{'='*50}")
    print(f"[결정] 다음 단계: {'종료' if decision == 'end' else '쿼리 수정'}")
    if state["finished"]:
        print(f"[이유] 만족스러운 결과를 찾았습니다.")
    elif state["iteration"] >= 5:
        print(f"[이유] 최대 반복 횟수({state['iteration']})에 도달했습니다.")
    else:
        print(f"[이유] 결과가 불충분합니다. 쿼리를 수정합니다.")
    print(f"{'='*50}")
    
    return decision

# pc check graph 정의
def pc_check_graph(user_query: str) -> AgentState:
    print(f"\n{'*'*60}")
    print(f"시작: PC 체크 그래프 (질문: {user_query})")
    print(f"{'*'*60}")
    
    # 초기 상태 설정
    state: AgentState = {
        "original_question": user_query,
        "user_meaning": "",
        "used_tables": [],
        "selected_columns": [],
        "queries": [],
        "previous_query": "",
        "results": None,
        "iteration": 0,
        "answer": "",
        "finished": False
    }
    
    # 그래프 정의
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("extract_tables", extract_tables)
    workflow.add_node("optimize_query", optimize_query)
    workflow.add_node("check_results", check_results)
    workflow.add_node("modify_query", modify_query)
    
    # 엣지 추가
    workflow.add_edge("extract_tables", "optimize_query")
    workflow.add_edge("optimize_query", "check_results")
    workflow.add_conditional_edges(
        "check_results",
        should_continue,
        {
            "end": END,
            "modify": "modify_query"
        }
    )
    workflow.add_edge("modify_query", "check_results")
    
    # 시작 노드 설정
    workflow.set_entry_point("extract_tables")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    print("그래프 컴파일 완료. 실행 시작...")
    
    # 그래프 실행
    try:
        final_state = app.invoke(state)
        print(f"\n{'*'*60}")
        print(f"완료: PC 체크 그래프")
        print(f"{'*'*60}")
        return final_state
    except Exception as e:
        print(f"\n{'*'*60}")
        print(f"오류: PC 체크 그래프 실행 중 예외 발생")
        print(f"오류 내용: {e}")
        print(f"{'*'*60}")
        # 오류 발생 시 현재 상태 반환
        return state

# 테스트 함수
def run_pc_check(user_query: str) -> str:
    result = pc_check_graph(user_query)
    if result["answer"]:
        return result["answer"]
    else:
        return "처리 중 오류가 발생했거나 적절한 답변을 찾지 못했습니다."

# 실행 코드 예시
if __name__ == "__main__":
    user_query = "cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘"
    answer = run_pc_check(user_query)
    print("\n최종 답변:")
    print(answer)