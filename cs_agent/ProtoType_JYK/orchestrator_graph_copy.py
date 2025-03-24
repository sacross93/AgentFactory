import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from cs_agent.ProtoType_JYK.pc_check_graph import run_pc_check
from web_search_langraph import run_web_search
from orchestrator_agent import orchestrator_agent, web_search_based_pc_check, sumary_answer_agent
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# 오케스트레이터 그래프 상태 정의 
class OrchestratorState(TypedDict):
    user_query: str                      # 사용자 원본 질의
    orchestrator_result: Dict[str, Any]  # 오케스트레이터 에이전트 결과
    pc_check_result: Optional[str]       # PC 체크 에이전트 결과
    web_search_result: Optional[str]     # 웹 검색 에이전트 결과
    web_search_extracted: Optional[Dict] # 웹 검색에서 추출된 컴포넌트 정보
    final_answer: str                    # 최종 통합 답변
    agents_to_run: List[str]             # 실행할 에이전트 목록
    current_agent: Optional[str]         # 현재 실행 중인 에이전트
    completed_agents: List[str]          # 실행 완료된 에이전트 목록
    requires_integration: bool           # 결과 통합 필요 여부

# 오케스트레이터 노드: 어떤 에이전트를 실행할지 결정
def run_orchestrator(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 오케스트레이터 에이전트")
    print(f"{'='*50}")
    print(f"[작업] 사용자 질의 분석 중: {state['user_query']}")
    
    # 오케스트레이터 에이전트 실행
    result = orchestrator_agent(state["user_query"])
    
    # 실행할 에이전트 목록 추출
    agents_to_run = [agent["agent_type"] for agent in result["agents"]]
    
    # 웹 검색과 PC 체크 모두 필요한지 확인
    requires_integration = "web_search" in agents_to_run and "pc_check" in agents_to_run
    
    print(f"[결과] 선택된 에이전트: {', '.join(agents_to_run)}")
    print(f"[결과] 선택 이유: {result['reasoning']}")
    print(f"[결과] 통합 처리 필요: {'예' if requires_integration else '아니오'}")
    
    return {
        **state,
        "orchestrator_result": result,
        "agents_to_run": agents_to_run,
        "completed_agents": [],
        "requires_integration": requires_integration
    }

# PC 체크 에이전트 노드
def run_pc_check_agent(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] PC 체크 에이전트")
    print(f"{'='*50}")
    print(f"[작업] PC 호환성 및 사양 확인 중...")
    
    # 통합 모드에서 웹 검색 결과가 있으면 이를 기반으로 PC 체크 실행
    if state["requires_integration"] and state["web_search_extracted"]:
        print(f"[정보] 웹 검색 결과를 기반으로 PC 체크 실행")
        result = run_pc_check(state["web_search_extracted"])
    else:
        # 일반 PC 체크 에이전트 실행
        result = run_pc_check(state["user_query"])
    
    print(f"[결과] PC 체크 완료")
    
    # 완료된 에이전트 목록에 추가
    completed_agents = state["completed_agents"] + ["pc_check"]
    
    return {
        **state,
        "pc_check_result": result,
        "current_agent": None,
        "completed_agents": completed_agents
    }

# 웹 검색 에이전트 노드
def run_web_search_agent(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 웹 검색 에이전트")
    print(f"{'='*50}")
    print(f"[작업] 웹에서 정보 검색 중...")
    
    # 웹 검색 에이전트 실행
    result = run_web_search(state["user_query"])
    
    print(f"[결과] 웹 검색 완료")
    
    # 통합 처리가 필요하면 웹 검색 결과에서 컴포넌트 정보 추출
    extracted_info = None
    if state["requires_integration"]:
        print(f"[작업] 웹 검색 결과에서 컴포넌트 정보 추출 중...")
        extracted_info = web_search_based_pc_check(result)
        print(f"[결과] 컴포넌트 정보 추출 완료: {extracted_info}")
    
    # 완료된 에이전트 목록에 추가
    completed_agents = state["completed_agents"] + ["web_search"]
    
    return {
        **state,
        "web_search_result": result,
        "web_search_extracted": extracted_info,
        "current_agent": None,
        "completed_agents": completed_agents
    }

# 결과 통합 노드
def integrate_results(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 결과 통합")
    print(f"{'='*50}")
    
    # 통합 모드에서 결과 통합
    if state["requires_integration"] and "web_search" in state["completed_agents"] and "pc_check" in state["completed_agents"]:
        print(f"[작업] 웹 검색과 PC 체크 결과 통합 중...")
        final_answer = sumary_answer_agent(
            state["web_search_result"],
            state["pc_check_result"],
            state["user_query"]
        )
        print(f"[결과] 통합 답변 생성 완료")
    else:
        # 단일 에이전트 실행 결과만 사용
        results = []
        if "pc_check" in state["completed_agents"] and state["pc_check_result"]:
            results.append(state["pc_check_result"])
        
        if "web_search" in state["completed_agents"] and state["web_search_result"]:
            results.append(state["web_search_result"])
        
        # 결과 통합
        if results:
            final_answer = "\n\n".join(results)
        else:
            final_answer = "적절한 결과를 찾지 못했습니다."
        
        print(f"[결과] 최종 답변 생성 완료")
    
    return {
        **state,
        "final_answer": final_answer
    }

# 다음 에이전트 선택 라우터
def select_next_agent(state: OrchestratorState) -> str:
    # 통합 실행 모드에서는 web_search를 먼저 실행
    if state["requires_integration"] and "web_search" not in state["completed_agents"]:
        print(f"\n{'='*50}")
        print(f"[결정] 다음 단계: web_search 에이전트 실행")
        print(f"[이유] 통합 모드에서는 웹 검색을 먼저 실행")
        print(f"{'='*50}")
        return "web_search"
    
    # 통합 모드에서 웹 검색 완료 후 pc_check 실행
    if state["requires_integration"] and "web_search" in state["completed_agents"] and "pc_check" not in state["completed_agents"]:
        print(f"\n{'='*50}")
        print(f"[결정] 다음 단계: pc_check 에이전트 실행")
        print(f"[이유] 웹 검색 결과를 기반으로 PC 체크 수행")
        print(f"{'='*50}")
        return "pc_check"
    
    # 모든 에이전트가 실행 완료되었는지 확인
    if set(state["agents_to_run"]).issubset(set(state["completed_agents"])):
        print(f"\n{'='*50}")
        print(f"[결정] 다음 단계: 결과 통합")
        print(f"[이유] 모든 에이전트 실행 완료")
        print(f"{'='*50}")
        return "integrate"
    
    # 아직 실행되지 않은 에이전트 중 하나 선택
    for agent in state["agents_to_run"]:
        if agent not in state["completed_agents"]:
            print(f"\n{'='*50}")
            print(f"[결정] 다음 단계: {agent} 에이전트 실행")
            print(f"[이유] 아직 실행되지 않은 에이전트")
            print(f"{'='*50}")
            return agent
    
    # 기본값 (도달하지 않아야 함)
    return "integrate"

# 오케스트레이터 그래프 정의
def orchestrator_graph(user_query: str) -> str:
    print(f"\n{'*'*60}")
    print(f"시작: 오케스트레이터 그래프 (질문: {user_query})")
    print(f"{'*'*60}")
    
    # 초기 상태 설정
    state: OrchestratorState = {
        "user_query": user_query,
        "orchestrator_result": {},
        "pc_check_result": None,
        "web_search_result": None,
        "web_search_extracted": None,
        "final_answer": "",
        "agents_to_run": [],
        "current_agent": None,
        "completed_agents": [],
        "requires_integration": False
    }
    
    # 그래프 정의
    workflow = StateGraph(OrchestratorState)
    
    # 노드 추가
    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("pc_check", run_pc_check_agent)
    workflow.add_node("web_search", run_web_search_agent)
    workflow.add_node("integrate", integrate_results)
    
    # 엣지 추가
    workflow.add_edge("orchestrator", select_next_agent)
    workflow.add_conditional_edges(
        "orchestrator",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "integrate": "integrate"
        }
    )
    
    workflow.add_conditional_edges(
        "pc_check",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "integrate": "integrate"
        }
    )
    
    workflow.add_conditional_edges(
        "web_search",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "integrate": "integrate"
        }
    )
    
    workflow.add_edge("integrate", END)
    
    # 시작 노드 설정
    workflow.set_entry_point("orchestrator")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    print("그래프 컴파일 완료. 실행 시작...")
    
    # 그래프 실행
    try:
        final_state = app.invoke(state)
        print(f"\n{'*'*60}")
        print(f"완료: 오케스트레이터 그래프")
        print(f"{'*'*60}")
        return final_state["final_answer"]
    except Exception as e:
        print(f"\n{'*'*60}")
        print(f"오류: 오케스트레이터 그래프 실행 중 예외 발생")
        print(f"오류 내용: {e}")
        print(f"{'*'*60}")
        return f"처리 중 오류가 발생했습니다: {str(e)}"

# 실행 코드 예시
if __name__ == "__main__":
    # 웹 검색만 필요한 경우
    # user_query = "배틀그라운드 권장사양을 알고싶어."
    
    # PC 체크만 필요한 경우
    # user_query = "cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘"
    
    # 둘 다 필요한 경우
    user_query = "배틀그라운드 권장사양을 알고싶어. 해당 권장사양을 기반으로 호환되는 실제 부품들을 알고 싶어"
    
    answer = orchestrator_graph(user_query)
    print("\n최종 답변:")
    print(answer) 