import sys
import time
import json
import logging
from datetime import datetime
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from cs_agent.ProtoType_JYK.pc_check_graph import run_pc_check
from web_search_langraph import run_web_search
from orchestrator_agent import orchestrator_agent, web_search_based_pc_check, sumary_answer_agent
from typing import TypedDict, List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("orchestrator.log"), logging.StreamHandler()]
)
logger = logging.getLogger("orchestrator")

# 오케스트레이터 그래프 상태 정의 
class OrchestratorState(TypedDict):
    user_query: str                           # 사용자 원본 질의
    orchestrator_result: Dict[str, Any]       # 오케스트레이터 에이전트 결과
    pc_check_result: Optional[str]            # PC 체크 에이전트 결과
    web_search_result: Optional[str]          # 웹 검색 에이전트 결과
    web_search_extracted: Optional[Dict]      # 웹 검색에서 추출된 컴포넌트 정보
    final_answer: str                         # 최종 통합 답변
    agents_to_run: List[str]                  # 실행할 에이전트 목록
    current_agent: Optional[str]              # 현재 실행 중인 에이전트
    completed_agents: List[str]               # 실행 완료된 에이전트 목록
    requires_integration: bool                # 결과 통합 필요 여부
    errors: Dict[str, str]                    # 각 단계별 발생한 오류
    metrics: Dict[str, Dict[str, Union[float, str]]]  # 성능 메트릭
    status_messages: List[str]                # 진행 상황 메시지
    retry_counts: Dict[str, int]              # 에이전트별 재시도 횟수

# 성능 측정 데코레이터
def measure_performance(agent_name):
    def decorator(func):
        def wrapper(state: OrchestratorState) -> OrchestratorState:
            # 시작 시간 기록
            start_time = time.time()
            
            # 에이전트 이름 기록
            state["current_agent"] = agent_name
            
            # 상태 메시지 추가
            state["status_messages"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} 실행 중...")
            
            try:
                # 함수 실행
                result_state = func(state)
                
                # 성공 시 메트릭 기록
                execution_time = time.time() - start_time
                if "metrics" not in result_state:
                    result_state["metrics"] = {}
                
                result_state["metrics"][agent_name] = {
                    "execution_time": execution_time,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                
                # 상태 메시지 추가
                result_state["status_messages"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} 완료 (소요 시간: {execution_time:.2f}초)"
                )
                
                logger.info(f"{agent_name} 실행 완료: {execution_time:.2f}초")
                return result_state
                
            except Exception as e:
                # 에러 발생 시 메트릭 기록
                execution_time = time.time() - start_time
                if "metrics" not in state:
                    state["metrics"] = {}
                    
                state["metrics"][agent_name] = {
                    "execution_time": execution_time,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                # 에러 정보 기록
                if "errors" not in state:
                    state["errors"] = {}
                state["errors"][agent_name] = str(e)
                
                # 상태 메시지 추가
                state["status_messages"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {agent_name} 실패: {str(e)}"
                )
                
                logger.error(f"{agent_name} 실행 실패: {str(e)}")
                
                # 재시도 카운트 업데이트
                if "retry_counts" not in state:
                    state["retry_counts"] = {}
                if agent_name not in state["retry_counts"]:
                    state["retry_counts"][agent_name] = 0
                
                return state
        return wrapper
    return decorator

# 오케스트레이터 노드: 어떤 에이전트를 실행할지 결정
@measure_performance("orchestrator")
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
@measure_performance("pc_check")
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
@measure_performance("web_search")
def run_web_search_agent(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 웹 검색 에이전트")
    print(f"{'='*50}")
    print(f"[작업] 웹에서 정보 검색 중...")
    
    # 웹 검색 에이전트 실행
    result = run_web_search(state["user_query"])
    
    print(f"[결과] 웹 검색 완료")
    
    # 통합 처리가 필요하면 웹 검색 결과에서 컴포넌트 정보 추출
    if state["requires_integration"]:
        print(f"[작업] 웹 검색 결과에서 컴포넌트 정보 추출 중...")
        extracted = run_extract_components(state)
        print(f"[결과] 컴포넌트 정보 추출 완료")
    else:
        extracted = None
    
    # 완료된 에이전트 목록에 추가
    completed_agents = state["completed_agents"] + ["web_search"]
    
    return {
        **state,
        "web_search_result": result,
        "web_search_extracted": extracted,
        "current_agent": None,
        "completed_agents": completed_agents
    }

# 웹 검색 결과에서 PC 부품 정보 추출
@measure_performance("extract_components")
def run_extract_components(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 컴포넌트 추출 에이전트")
    print(f"{'='*50}")
    print(f"[작업] 웹 검색 결과에서 컴포넌트 정보 추출 중...")
    
    if not state.get("web_search_result"):
        print(f"[오류] 웹 검색 결과가 없습니다.")
        state["errors"]["extract_components"] = "웹 검색 결과 없음"
        state["web_search_extracted"] = {"cpu": "Unknown", "gpu": "Unknown", "ram": "8GB", "motherboard": "Unknown"}
        return state
    
    try:
        # 컴포넌트 정보 추출
        extracted = web_search_based_pc_check(state["web_search_result"])
        print(f"[결과] 추출된 컴포넌트 정보: {json.dumps(extracted, ensure_ascii=False)}")
        
        state["web_search_extracted"] = extracted
    except Exception as e:
        error_msg = f"컴포넌트 정보 추출 실패: {str(e)}"
        print(f"[오류] {error_msg}")
        state["errors"]["extract_components"] = error_msg
        # 기본값 설정
        state["web_search_extracted"] = {"cpu": "Unknown", "gpu": "Unknown", "ram": "8GB", "motherboard": "Unknown"}
    
    return state

# 결과 통합 노드
@measure_performance("integrate")
def integrate_results(state: OrchestratorState) -> OrchestratorState:
    print(f"\n{'='*50}")
    print(f"[실행 중] 결과 통합 에이전트")
    print(f"{'='*50}")
    print(f"[작업] 에이전트 결과 통합 중...")
    
    # PC 체크와 웹 검색 결과가 모두 있는 경우
    if "pc_check" in state["completed_agents"] and "web_search" in state["completed_agents"]:
        print(f"[정보] PC 체크와 웹 검색 결과 모두 통합")
        final_answer = sumary_answer_agent(
            user_question=state["user_query"],
            pc_check_result=state["pc_check_result"] or "PC 체크 결과를 찾지 못했습니다.",
            web_search_result=state["web_search_result"] or "웹 검색 결과를 찾지 못했습니다."
        )
    
    # PC 체크 결과만 있는 경우
    elif "pc_check" in state["completed_agents"]:
        print(f"[정보] PC 체크 결과만 사용")
        final_answer = state["pc_check_result"] or "PC 체크 결과를 찾지 못했습니다."
    
    # 웹 검색 결과만 있는 경우
    elif "web_search" in state["completed_agents"]:
        print(f"[정보] 웹 검색 결과만 사용")
        final_answer = state["web_search_result"] or "웹 검색 결과를 찾지 못했습니다."
    
    # 결과가 없는 경우
    else:
        print(f"[정보] 실행된 에이전트 없음")
        final_answer = "질문 처리를 위한 에이전트를 실행하지 못했습니다."
    
    print(f"[결과] 최종 답변 생성 완료")
    
    # 처리 데이터 요약
    stats = {
        "completed_agents": state["completed_agents"],
        "errors": state["errors"] if "errors" in state else {},
        "metrics": state["metrics"] if "metrics" in state else {}
    }
    
    logger.info(f"처리 완료: {json.dumps(stats, default=str)}")
    
    return {
        **state,
        "final_answer": final_answer
    }

# 에이전트 선택 함수
def select_next_agent(state: OrchestratorState) -> str:
    # 이미 모든 에이전트가 실행되었으면 결과 통합으로 이동
    if set(state["agents_to_run"]).issubset(set(state["completed_agents"])):
        print(f"\n{'='*50}")
        print(f"[결정] 다음 단계: 결과 통합")
        print(f"[이유] 모든 처리 완료")
        print(f"{'='*50}")
        return "integrate"
    
    # 통합 모드에서는 실행 순서가 중요함 (웹 검색 -> 추출 -> PC 체크)
    if state["requires_integration"]:
        # 실패한 웹 검색 재시도 처리
        if "web_search" in state["agents_to_run"] and "web_search" not in state["completed_agents"]:
            if "web_search" in state.get("errors", {}) and should_retry(state, "web_search"):
                print(f"\n{'='*50}")
                print(f"[결정] 다음 단계: 웹 검색 (재시도 {state['retry_counts'].get('web_search', 0) + 1})")
                print(f"[이유] 이전 시도 실패: {state['errors']['web_search']}")
                print(f"{'='*50}")
                return "web_search"
            
            if "web_search" not in state.get("errors", {}):
                print(f"\n{'='*50}")
                print(f"[결정] 다음 단계: 웹 검색")
                print(f"[이유] 통합 처리를 위해 웹 검색 먼저 실행")
                print(f"{'='*50}")
                return "web_search"
        
        # 웹 검색이 완료되고 컴포넌트 추출이 필요한 경우
        if "web_search" in state["completed_agents"] and state.get("web_search_result") and not state.get("web_search_extracted"):
            # extract_components 노드로 이동
            print(f"\n{'='*50}")
            print(f"[결정] 다음 단계: 컴포넌트 추출")
            print(f"[이유] 웹 검색 결과에서 PC 부품 정보 추출 필요")
            print(f"{'='*50}")
            return "extract_components"
            
        # PC 체크 실행
        if "pc_check" in state["agents_to_run"] and "pc_check" not in state["completed_agents"] and state.get("web_search_extracted"):
            print(f"\n{'='*50}")
            print(f"[결정] 다음 단계: PC 체크")
            print(f"[이유] 추출된 컴포넌트 정보 기반 PC 체크 실행")
            print(f"{'='*50}")
            return "pc_check"
    else:
        # 단일 에이전트 모드: 첫 번째 에이전트만 실행
        agent = state["agents_to_run"][0]
        if agent not in state["completed_agents"]:
            print(f"\n{'='*50}")
            print(f"[결정] 다음 단계: {agent}")
            print(f"[이유] 단일 에이전트 실행")
            print(f"{'='*50}")
            return agent
    
    # 모든 처리가 완료되면 결과 통합
    print(f"\n{'='*50}")
    print(f"[결정] 다음 단계: 결과 통합")
    print(f"[이유] 모든 처리 완료")
    print(f"{'='*50}")
    return "integrate"

# 실패한 에이전트 재시도 로직
def should_retry(state: OrchestratorState, agent_name: str, max_retries: int = 2) -> bool:
    if agent_name not in state.get("retry_counts", {}):
        return True
    
    if state["retry_counts"][agent_name] < max_retries:
        return True
    
    return False

# 오케스트레이터 그래프 실행
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
        "requires_integration": False,
        "errors": {},
        "metrics": {},
        "status_messages": [f"[{datetime.now().strftime('%H:%M:%S')}] 처리 시작: {user_query}"],
        "retry_counts": {}
    }
    
    # 그래프 정의
    workflow = StateGraph(OrchestratorState)
    
    # 노드 추가
    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("pc_check", run_pc_check_agent)
    workflow.add_node("web_search", run_web_search_agent)
    workflow.add_node("extract_components", run_extract_components)
    workflow.add_node("integrate", integrate_results)
    
    # 엣지 추가
    workflow.add_conditional_edges(
        "orchestrator",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "extract_components": "extract_components",
            "integrate": "integrate"
        }
    )
    
    workflow.add_conditional_edges(
        "pc_check",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "extract_components": "extract_components",
            "integrate": "integrate"
        }
    )
    
    workflow.add_conditional_edges(
        "web_search",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "extract_components": "extract_components",
            "integrate": "integrate"
        }
    )
    
    workflow.add_conditional_edges(
        "extract_components",
        select_next_agent,
        {
            "pc_check": "pc_check",
            "web_search": "web_search",
            "extract_components": "extract_components",
            "integrate": "integrate"
        }
    )
    
    workflow.add_edge("integrate", END)
    
    # 시작 노드 설정
    workflow.set_entry_point("orchestrator")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    logger.info(f"질의 처리 시작: {user_query}")
    print("그래프 컴파일 완료. 실행 시작...")
    
    # 그래프 실행
    try:
        start_time = time.time()
        final_state = app.invoke(state)
        total_time = time.time() - start_time
        
        print(f"\n{'*'*60}")
        print(f"완료: 오케스트레이터 그래프 (소요 시간: {total_time:.2f}초)")
        print(f"상태 메시지:")
        for msg in final_state["status_messages"]:
            print(f" - {msg}")
        print(f"{'*'*60}")
        
        logger.info(f"질의 처리 완료: {user_query} (소요 시간: {total_time:.2f}초)")
        return final_state["final_answer"]
    except Exception as e:
        print(f"\n{'*'*60}")
        print(f"오류: 오케스트레이터 그래프 실행 중 예외 발생")
        print(f"오류 내용: {e}")
        print(f"{'*'*60}")
        
        logger.error(f"질의 처리 실패: {user_query} - {str(e)}")
        return f"처리 중 오류가 발생했습니다: {str(e)}"

# 실행 코드 예시
if __name__ == "__main__":
    # 테스트 케이스 선택
    test_cases = [
        "배틀그라운드 권장사양을 알고싶어.",  # 웹 검색만 필요한 경우
        "cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘",  # PC 체크만 필요한 경우
        "배틀그라운드 권장사양을 알고싶어. 해당 권장사양을 기반으로 호환되는 실제 부품들을 알고 싶어"  # 둘 다 필요한 경우
    ]
    
    # 세 번째 테스트 케이스 실행
    user_query = test_cases[1]
    
    answer = orchestrator_graph(user_query)
    print("\n최종 답변:")
    print(answer) 