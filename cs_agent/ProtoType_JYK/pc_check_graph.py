import sys
import json
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from cs_agent.ProtoType_JYK.pc_check_agents import *
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

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
    attempt_history: List[Dict[str, Any]]

def extract_tables(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 테이블 정보 추출 에이전트")
    print(f"{'='*50}")
    print("[작업] 사용자 질의에서 필요한 테이블 식별 중...")
    
    user_query = state["original_question"]
    table_info = table_abstract_agent(user_query)
    
    print(f"[결과] 사용자 의도: {table_info['user_meaning']}")
    print(f"[결과] 필요한 테이블: {', '.join(table_info['table_names'])}")
    
    # 실제 존재하는 테이블 확인
    all_tables = [table[0] for table in get_db_table()]
    valid_tables = [t for t in table_info['table_names'] if t in all_tables]
    
    # 존재하지 않는 테이블 로깅
    invalid_tables = [t for t in table_info['table_names'] if t not in all_tables]
    if invalid_tables:
        print(f"[경고] 존재하지 않는 테이블이 요청됨: {', '.join(invalid_tables)}")
    
    return {
        **state,
        "user_meaning": table_info["user_meaning"],
        "used_tables": valid_tables,
        "attempt_history": []
    }

def optimize_query(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 쿼리 최적화 에이전트")
    print(f"{'='*50}")
    print("[작업] 최적의 SQL 쿼리 생성 중...")
    
    user_query = state["original_question"]
    table_info = {"user_meaning": state["user_meaning"], "table_names": state["used_tables"], "reason": ""}
    
    try:
        query_info, result = query_optimize_agent(user_query, table_info)
        print(f"[결과] 쿼리 생성 완료")
        print(f"[결과] 쿼리 결과 행 수: {len(result) if isinstance(result, list) else 0}")
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
    
    # 새로운 attempt 생성 및 history 업데이트
    attempt = {
        "query": state["previous_query"], 
        "result": state["results"], 
        "issues": check_info["issues"]
    }
    attempt_history = state["attempt_history"] + [attempt]
    
    return {
        **state,
        "answer": check_info["answer"],
        "finished": check_info["result"],
        "attempt_history": attempt_history
    }

def summarize_answer(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 요약 답변 에이전트")
    print(f"{'='*50}")
    print("[작업] 최종 답변 생성 중...")
    
    user_query = state["original_question"]
    check_info = state.get("attempt_history", [])
    
    # check_info를 JSON 문자열로 변환
    # check_info_str = json.dumps(check_info, ensure_ascii=False)
    
    final_answer = summary_answer_agent(check_info, user_query)
    print(f"[결과] 최종 답변 생성 완료")
    
    return {
        **state,
        "answer": final_answer
    }

def modify_query(state: AgentState) -> AgentState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 쿼리 수정 에이전트 (반복 {state['iteration']})")
    print(f"{'='*50}")
    print("[작업] 쿼리 개선 중...")
    
    user_query = state["original_question"]
    table_info = {"user_meaning": state["user_meaning"], "table_names": state["used_tables"], "reason": ""}
    query_info = {"reason": "", "selected_columns": state["selected_columns"], "query": state["previous_query"]}
    
    try:
        query_modify_info, results = query_modify_agent(user_query, table_info, query_info, state["results"], state["attempt_history"])
        
        # 쿼리 실행 결과 로깅
        if isinstance(results, list):
            print(f"[결과] 수정 이유: {query_modify_info['reason']}")
            print(f"[결과] 생성된 쿼리 수: {len(query_modify_info['queries'])}")
            print(f"[결과] 쿼리 결과 행 수: {len(results)}")
        else:
            print(f"[결과] 수정 이유: {query_modify_info['reason']}")
            print(f"[결과] 생성된 쿼리 수: {len(query_modify_info['queries'])}")
            print(f"[결과] 쿼리 결과: 오류 또는 빈 결과")
        
        return {
            **state,
            "queries": query_modify_info["queries"],
            "results": results,
            "previous_query": query_modify_info["queries"][0]["sql"] if query_modify_info["queries"] else "",
            "iteration": state["iteration"] + 1
        }
    except Exception as e:
        print(f"[오류] 쿼리 수정 중 오류 발생: {e}")
        return {
            **state,
            "iteration": state["iteration"] + 1
        }

def should_continue(state: AgentState) -> str:
    if state["finished"]:
        decision = "summarize"
        reason = "만족스러운 결과를 찾았습니다"
    elif state["iteration"] >= 3:
        decision = "summarize"
        reason = f"최대 반복 횟수({state['iteration']})에 도달했습니다"
    elif len(state["attempt_history"]) >= 2 and state["attempt_history"][-1]["issues"] == state["attempt_history"][-2]["issues"]:
        decision = "summarize"
        reason = "연속해서 같은 문제가 발생했습니다"
    else:
        decision = "modify"
        reason = "결과가 불충분합니다. 쿼리를 수정합니다"
    
    print(f"\n{'='*50}")
    print(f"[결정] 다음 단계: {'요약 답변 생성' if decision == 'summarize' else '쿼리 수정'}")
    print(f"[이유] {reason}")
    print(f"{'='*50}")
    return decision

def pc_check_graph(user_query: str) -> AgentState:
    print(f"\n{'*'*60}")
    print(f"시작: PC 체크 그래프 (질문: {user_query})")
    print(f"{'*'*60}")
    
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
        "finished": False,
        "attempt_history": []
    }
    
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_tables", extract_tables)
    workflow.add_node("optimize_query", optimize_query)
    workflow.add_node("check_results", check_results)
    workflow.add_node("modify_query", modify_query)
    workflow.add_node("summarize_answer", summarize_answer)
    
    workflow.add_edge("extract_tables", "optimize_query")
    workflow.add_edge("optimize_query", "check_results")
    workflow.add_conditional_edges("check_results", should_continue, 
                                {"summarize": "summarize_answer", "modify": "modify_query"})
    workflow.add_edge("modify_query", "check_results")
    workflow.add_edge("summarize_answer", END)
    workflow.set_entry_point("extract_tables")
    
    app = workflow.compile()
    print("그래프 컴파일 완료. 실행 시작...")
    
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
        return state

def run_pc_check(user_query: str) -> str:
    result = pc_check_graph(user_query)
    if result["answer"]:
        return result["answer"].content
    else:
        return "처리 중 오류가 발생했거나 적절한 답변을 찾지 못했습니다."

if __name__ == "__main__":
    user_query = "cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘"
    answer = run_pc_check(user_query)
    print("\n최종 답변:")
    print(answer)