import streamlit as st
from langchain_ollama import OllamaLLM
import json
import time
import re

from agents import AgentSystem
from utils import get_memory, apply_custom_css, wait_for_ollama_server
from ui_components import setup_page_config, create_sidebar, create_chat_ui, display_processing_status

# 새로운 통합 에이전트 import
from integrated_agent import create_integrated_agent

# 통합 로깅 시스템 적용
from logging_config import get_logger

# 앱 로거 가져오기
logger = get_logger("Streamlit")

def initialize_session_state():
    """
    세션 상태 초기화 함수
    
    Streamlit은 각 사용자 세션마다 상태를 유지해야 하므로,
    필요한 변수들을 세션 상태에 초기화합니다.
    """
    # 메시지 기록을 저장하는 리스트 (사용자와 AI의 대화 내용)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 현재 처리 중인지 여부를 나타내는 플래그
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # 현재 처리 중인 질문 저장
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
    # 처리 로그를 저장하는 리스트
    if "process_logs" not in st.session_state:
        st.session_state.process_logs = []
    
    # 통합 에이전트 인스턴스 저장
    if "integrated_agent" not in st.session_state:
        st.session_state.integrated_agent = None

def main():
    """
    메인 애플리케이션 함수
    
    Streamlit 앱의 전체 구조와 흐름을 정의합니다.
    사용자 입력 처리, UI 구성, 에이전트 호출 등을 관리합니다.
    """
    # 페이지 설정 (제목, 아이콘 등)
    setup_page_config()
    logger.info("Streamlit 앱 시작")
    
    # 커스텀 CSS 스타일 적용
    apply_custom_css()
    
    # 세션 상태 초기화 (메시지, 에이전트 상태 등)
    initialize_session_state()
    
    # Ollama 서버 연결 확인 - AI 모델 서버가 실행 중인지 체크
    server_available = wait_for_ollama_server()
    if not server_available:
        st.error("⚠️ Ollama AI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.")
        st.warning("서버가 실행 중이지 않으면 대화 기능이 제한됩니다.")
        st.session_state.server_error = True
    else:
        st.session_state.server_error = False
    
    # LangChain 메모리 초기화 (대화 기록 유지)
    memory = get_memory()
    
    # 사이드바 UI 생성 및 설정 옵션 가져오기
    sidebar_options = create_sidebar(memory)
    
    # 선택된 AI 모델 초기화
    llm = OllamaLLM(
        model=sidebar_options["selected_model"],
        base_url="http://192.168.110.102:11434"
    )
    
    # 통합 에이전트 초기화 (필요시 재생성)
    # 사용자가 모델을 변경한 경우 에이전트를 다시 생성합니다
    if (st.session_state.integrated_agent is None or 
        sidebar_options["selected_model"] != st.session_state.current_model):
        st.session_state.integrated_agent = create_integrated_agent(llm)
        st.session_state.current_model = sidebar_options["selected_model"]
    
    # 채팅 UI 생성 (메시지 표시 영역과 입력 필드)
    chat_container, user_input = create_chat_ui()
    
    # 처리가 완료되고 로그가 존재하는 경우 - 이전 프로세스의 결과 표시
    if st.session_state.get("process_complete", False) and st.session_state.get("final_answer", None):
        # 처리 유형 확인 (세션 상태에서 가져옴)
        processing_type = st.session_state.get("processing_type", st.session_state.get("query_type", "알 수 없음"))
        
        # 마지막 메시지(방금 추가된 답변)에 대해서만 expander 표시
        last_msg_idx = len(st.session_state.messages) - 1
        if last_msg_idx >= 0 and st.session_state.messages[last_msg_idx]["role"] == "assistant":
            # 생성 과정을 토글 형식으로 표시 - expander의 제목에 처리 유형을 표시합니다
            with st.expander(f"📊 답변 생성 과정 보기 - {processing_type}", expanded=False):
                # 로그 표시 - session_state에 저장된 로그를 가져와 표시합니다
                for log in st.session_state.process_logs:
                    st.markdown(f"- {log}")
            
            # 프로세스 완료 상태 초기화 (다음 질문을 위해)
            st.session_state.process_complete = False
            st.session_state.final_answer = None
    
    # 사용자 입력 처리 (서버 오류 상태면 비활성화)
    if st.session_state.server_error and user_input:
        with chat_container:
            with st.chat_message("assistant"):
                st.error("서버 연결 오류로 질문을 처리할 수 없습니다. 서버 상태를 확인해 주세요.")
        return
    
    # FAQ 질문 선택 처리 - 사이드바에서 FAQ를 선택한 경우
    if sidebar_options["selected_faq"] != "선택하세요...":
        st.session_state.current_question = sidebar_options["selected_faq"]
        st.session_state.processing = True
        st.rerun()  # 화면 새로고침으로 처리 시작
    
    # 사용자 입력 처리 - 채팅 입력란에 텍스트를 입력한 경우
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 처리 상태로 설정
        st.session_state.processing = True
        st.session_state.current_question = user_input
        
        # 화면 갱신 - 이렇게 하면 사용자 입력이 표시되고 처리가 시작됩니다
        st.rerun()
    
    # 처리 중인 경우 에이전트 실행
    if st.session_state.processing and st.session_state.current_question:
        # 처리 상태 표시 및 답변 생성 함수 호출
        response = display_integrated_processing(
            chat_container, 
            st.session_state.integrated_agent,
            st.session_state.current_question,
            memory,
            sidebar_options["conversation_style"]
        )
        
        # response 객체를 세션 상태에 저장 (rerun 후에도 유지되도록)
        st.session_state.last_response = response
        
        # 페이지 rerun - 처리 완료 후 표시를 위해
        st.rerun()
    else:
        # 처리 완료 후 expander 표시
        if st.session_state.get("process_complete", False) and st.session_state.get("last_response"):
            response = st.session_state.last_response
            
            # 가능한 모든 소스에서 처리 유형 결정
            processing_type = response.get("processing_type", None)
            if not processing_type:
                processing_type = st.session_state.get("query_type", "알 수 없음")
                # 게임 관련 처리인지 확인
                if "game" in processing_type.lower() or "게임" in str(st.session_state.get("current_question", "")):
                    processing_type = "게임 PC 구성 추천"
                elif "호환성" in processing_type:
                    processing_type = "PC 부품 호환성 분석"
            
            # expander 제목 설정
            expander_title = f"📊 답변 생성 과정 보기 - {processing_type}"
            
            # 로그 표시
            with st.expander(expander_title, expanded=False):
                logs_to_display = []
                
                # 1. 응답 객체의 로그 확인
                if "processing_logs" in response and response["processing_logs"]:
                    logs_to_display = response["processing_logs"]
                # 2. 터미널 로그 확인
                elif "terminal_logs" in response and response["terminal_logs"]:
                    logs_to_display = [log.replace("로그 추가: ", "") for log in response["terminal_logs"]]
                # 3. 세션 로그 확인
                elif st.session_state.get("process_logs"):
                    logs_to_display = st.session_state.process_logs
                
                # 로그 표시
                for log in logs_to_display:
                    st.markdown(f"- {log}")
                
            # 처리 완료 상태 초기화
            st.session_state.process_complete = False

def display_integrated_processing(chat_container, integrated_agent, question, memory, conversation_style):
    """
    통합 에이전트 응답 생성 및 처리 상태 표시
    """
    # 채팅 컨테이너 외부에 로그 표시 영역 생성
    log_display = st.empty()  # 메인 채팅 영역 외부에 로그 표시 영역
    
    # 로그 초기화 (처리 시작 시에만 초기화)
    # continuing_process 플래그를 확인하여 이미 진행 중인 프로세스인지 확인
    if not st.session_state.get("continuing_process", False):
        if "process_logs" not in st.session_state:
            st.session_state.process_logs = []
        st.session_state.process_logs = []  # 로그 초기화
        st.session_state.process_complete = False
    
    # 프로세스가 이미 완료되었다면, 최종 상태 표시로 바로 진행
    # 이 부분은 rerun 이후에 실행될 수 있음
    if st.session_state.get("process_complete", False):
        # 최종 결과 표시 - 완료 메시지와 안내 표시
        with log_display.container():
            st.success("✅ 처리가 완료되었습니다. 아래 답변을 확인하세요.")
            st.info("📊 자세한 로그는 답변 아래의 '답변 생성 과정 보기'를 클릭하면 확인할 수 있습니다.")
            # 작은 간격 추가
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        # 저장된 결과 반환
        return st.session_state.final_answer
    
    # 로그 업데이트 함수 - 새 로그를 추가하고 UI에 표시
    def add_log(message):
        """
        로그 메시지를 추가하고 UI에 표시하는 내부 함수
        
        Args:
            message: 추가할 로그 메시지
        """
        # 콘솔에 출력 (디버깅용)
        print(f"로그 추가: {message}")
        
        # 세션 상태에 로그 추가 - 이 로그는 나중에 expander에서 표시됨
        st.session_state.process_logs.append(message)
        
        # 로그 영역 업데이트 - 실시간으로 로그 표시
        with log_display.container():
            st.markdown("### 🔍 실시간 처리 로그")
            log_container = st.container()
            # 최근 로그는 항상 보이도록 스크롤 가능 컨테이너 생성
            with log_container.container():
                for i, log in enumerate(st.session_state.process_logs[-10:]):  # 최근 10개만 표시
                    log_style = "log-info"
                    if "❌" in log or "오류" in log:
                        log_style = "log-error"
                    elif "⚠️" in log:
                        log_style = "log-warning" 
                    elif "✅" in log or "완료" in log:
                        log_style = "log-success"
                        
                    st.markdown(f"<div class='log-entry {log_style}'>{i+1}. {log}</div>", unsafe_allow_html=True)
        
        # UI 업데이트를 위한 짧은 대기 - Streamlit의 비동기 업데이트 특성 때문에 필요
        time.sleep(0.05)
    
    # 처리 시작 로그 추가
    add_log(f"🚀 에이전트 처리 시작 - 질문: '{question}' ({time.strftime('%H:%M:%S')})")
    
    # 진행 상황 표시 컨테이너 - 프로그레스 바와 상태 메시지를 표시
    progress_container = st.empty()
    
    try:
        with progress_container:
            # 진행 상태 표시 컴포넌트
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 진행 상태 업데이트 함수
            def update_progress(progress, status):
                """
                진행 상태 및 상태 메시지 업데이트 함수
                
                Args:
                    progress: 0~1 사이의 진행률 값
                    status: 상태 텍스트
                """
                progress_bar.progress(progress)
                status_text.markdown(f"**{status}**")
                add_log(status)  # 로그에 상태 추가
            
            # 질문 분석 단계
            update_progress(0.1, "🔍 질문 분석 중...")
            
            # 메모리에서 대화 기록 가져오기
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", "")
            
            # 통합 워크플로우 실행
            update_progress(0.3, "🧠 질문 유형 분류 중...")
            
            try:
                # 통합 에이전트 실행 - 디버깅 추가
                add_log("🚀 통합 에이전트 실행 중...")
                
                # 실행 시간 측정 시작
                execution_start_time = time.time()
                
                try:
                    result = integrated_agent.run_workflow(question, chat_history)
                    
                    # 오류 여부 확인
                    if result.get("errors", []):
                        # 오류가 있지만 실행은 완료된 경우
                        error_msgs = result.get("errors", [])
                        add_log(f"⚠️ 처리 중 {len(error_msgs)}개의 오류가 발생했습니다")
                        for err in error_msgs[:3]:  # 처음 3개만 표시
                            add_log(f"⚠️ 오류 내용: {err}")
                        
                        # 부분 성공으로 표시
                        add_log("⚠️ 일부 오류와 함께 에이전트 실행 완료")
                    else:
                        # 완전 성공
                        add_log("✅ 에이전트 실행 성공적으로 완료")
                
                except Exception as e:
                    # 심각한 오류 - 실행이 완전히 실패한 경우
                    error_msg = str(e)
                    add_log(f"❌ 에이전트 실행 실패: {error_msg}")
                    
                    # 스택 트레이스 기록 (로그에만 표시)
                    import traceback
                    stack_trace = traceback.format_exc()
                    logger.error(f"에이전트 실행 중 오류:\n{stack_trace}")
                    
                    # 실행 시간 기록
                    execution_end_time = time.time()
                    execution_duration = execution_end_time - execution_start_time
                    add_log(f"⏱️ 오류 발생까지 경과 시간: {execution_duration:.2f}초")
                    
                    # 기본 결과 생성
                    result = {
                        "answer": f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {error_msg}",
                        "query_type": "error",
                        "collected_information": st.session_state.process_logs.copy(),
                        "errors": [error_msg]
                    }
                
                # 쿼리 유형 확인
                query_type = result.get("query_type", "unknown")
                add_log(f"🏷️ 분류된 질문 유형: {query_type}")
                
                # 수집된 정보 표시
                collected_info = result.get("collected_information", [])
                add_log(f"📊 수집된 정보 수: {len(collected_info)}개")
                
                # 모든 수집 정보 로그에 추가
                for info in collected_info:
                    # 이미 로그에 있는 항목은 중복 추가하지 않음
                    if info not in st.session_state.process_logs:
                        add_log(info)
                
                # 실행 시간 계산 및 표시
                execution_end_time = time.time()
                execution_duration = execution_end_time - execution_start_time
                add_log(f"⏱️ 총 실행 시간: {execution_duration:.2f}초")
                
                # 성능 관련 정보 추가
                if execution_duration > 30:
                    add_log("⚠️ 처리 시간이 30초를 초과했습니다. 성능 개선이 필요할 수 있습니다.")
                
                # 질문 유형별 진행 상황 표시
                if query_type == "web_search":
                    update_progress(0.7, "📝 검색 결과 분석 중...")
                elif query_type == "pc_compatibility":
                    update_progress(0.7, "📊 호환성 데이터 처리 중...")
                elif query_type == "hybrid":
                    update_progress(0.7, "🔄 통합 분석 수행 중...")
                
                # 최종 답변 생성
                update_progress(0.9, "📝 최종 답변 생성 중...")
                answer = result.get("answer", "답변을 생성할 수 없습니다.")

                # 중복 답변 처리 - 더 강력한 중복 검사 적용
                processed_answer = process_answer_for_duplicates(answer)

                # 중복 확인 결과 로깅
                if len(processed_answer) < len(answer):
                    add_log(f"⚠️ 중복 내용 감지: 답변 길이 {len(answer)}자 → {len(processed_answer)}자로 축소")
                else:
                    processed_answer = answer  # 중복이 없는 경우 원본 답변 사용

                # 캡처된 터미널 로그가 있는지 확인하고 표시
                if "terminal_logs" in result:
                    terminal_logs = result["terminal_logs"]
                    for log in terminal_logs:
                        add_log(log.replace("로그 추가: ", ""))  # "로그 추가: " 접두어 제거

            except Exception as e:
                # 전체 처리 과정에서의 예외 처리
                error_msg = str(e)
                add_log(f"❌ 예상치 못한 오류: {error_msg}")
                processed_answer = f"예상치 못한 오류가 발생했습니다: {error_msg}"
                query_type = "error"
            
            # 진행 완료
            update_progress(1.0, "✨ 답변 생성 완료!")
            add_log("✅ 응답 생성 완료")
            time.sleep(0.5)  # 완료 메시지를 잠시 표시
    
    except Exception as e:
        # 전체 처리 과정에서의 예외 처리
        error_msg = str(e)
        add_log(f"❌ 예상치 못한 오류: {error_msg}")
        processed_answer = f"예상치 못한 오류가 발생했습니다: {error_msg}"
        query_type = "error"
    
    # 결과를 세션 상태에 저장 - 이것들은 페이지 rerun 후에도 유지됨
    st.session_state.final_answer = processed_answer
    st.session_state.query_type = query_type
    st.session_state.process_complete = True
    st.session_state.process_logs = st.session_state.process_logs.copy()
    
    # 메모리에 대화 저장 - 다음 대화를 위한 컨텍스트 유지
    memory.save_context({"question": question}, {"answer": processed_answer})
    
    # 메시지 저장 - UI에 표시될 대화 내용
    st.session_state.messages.append({"role": "assistant", "content": processed_answer})
    
    # 처리 완료 상태로 설정
    st.session_state.processing = False
    st.session_state.current_question = None
    
    # 중요: 결과 객체에도 처리 타입 및 로그 저장 (expander 표시용)
    if "processing_type" not in result:
        # 응답에 처리 유형 정보가 없으면 추가
        if query_type == "game_pc_recommendation":
            result["processing_type"] = "게임 PC 구성 추천"
        elif query_type == "pc_compatibility":
            result["processing_type"] = "PC 부품 호환성 분석"
        elif "권장" in question or "사양" in question:
            result["processing_type"] = "프로그램 요구사항 분석"
        else:
            result["processing_type"] = query_type
    
    # 처리 유형을 세션 상태에도 저장 (rerun 후에도 유지되도록)
    st.session_state.processing_type = result.get("processing_type", query_type)

    # 처리 로그도 결과에 저장
    result["processing_logs"] = st.session_state.process_logs.copy()
    
    # 처리 완료 후 페이지 rerun - 이 rerun이 핵심
    st.rerun()
    
    return result

# 텍스트 유사도 비교 함수
def similar_text(text1, text2, threshold=0.8):
    """두 텍스트의 유사도를 계산 (0-1)"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return False
        
    common = words1.intersection(words2)
    similarity = len(common) / max(len(words1), len(words2))
    return similarity > threshold

# 반복되는 답변 문제 해결을 위한 최종 방안
def process_answer_for_duplicates(answer):
    """답변에서 중복된 내용을 제거하는 함수"""
    # 완전히 동일한 두 부분으로 나뉘어 있는지 확인 (가장 명확한 케이스)
    if len(answer) % 2 == 0:
        half_len = len(answer) // 2
        first_half = answer[:half_len]
        second_half = answer[half_len:]
        
        if first_half == second_half:
            logger.info("완전히 동일한, 두 개로 나뉜 답변 감지")
            return first_half
    
    # 줄바꿈으로 나눈 후 중복 단락 검사
    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        # 첫 번째 단락과 마지막 단락이 동일한지 확인
        if paragraphs[0] == paragraphs[-1]:
            logger.info("첫 번째와 마지막 단락이 동일함")
            # 중복 단락 찾기
            unique_paragraphs = []
            seen = set()
            for para in paragraphs:
                if para not in seen:
                    unique_paragraphs.append(para)
                    seen.add(para)
            return "\n\n".join(unique_paragraphs)
    
    # 문장 단위로 비교
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > 10:  # 충분한 문장이 있는 경우에만 검사
        half_point = len(sentences) // 2
        first_half_sentences = sentences[:half_point]
        second_half_sentences = sentences[half_point:]
        
        # 두 번째 부분이 첫 번째 부분을 포함하는지 확인
        if all(sent in second_half_sentences for sent in first_half_sentences[:5]):
            logger.info("문장 수준에서 중복 감지")
            return " ".join(first_half_sentences)
    
    return answer

if __name__ == "__main__":
    main()