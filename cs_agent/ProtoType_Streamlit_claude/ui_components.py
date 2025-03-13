import streamlit as st
import time
from utils import export_chat_history, save_feedback, check_ollama_server, wait_for_ollama_server
import json
from agents import final_answer_prompt 

def setup_page_config():
    """페이지 설정 초기화"""
    st.set_page_config(
        page_title="제플몰 종합 상담 봇",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def create_sidebar(memory):
    """사이드바 UI 생성"""
    st.sidebar.title("설정 및 옵션")
    st.sidebar.markdown("---")
    
    # 대화 기록 지우기 버튼
    if st.sidebar.button("대화 기록 지우기"):
        memory.clear()
        st.session_state.messages = []
        st.session_state.process_logs = []
        st.rerun()
    
    # 검색 설정 섹션
    st.sidebar.markdown("## 검색 설정")
    max_searches = st.sidebar.slider("최대 검색 횟수", min_value=1, max_value=10, value=5, step=1)
    show_search_process = st.sidebar.checkbox("검색 과정 자동으로 보여주기", value=False)
    
    # 모델 설정 섹션
    st.sidebar.markdown("## 모델 설정")
    model_options = ["gemma3:27b","exaone3.5:32b", "qwen2.5:32b", "deepseek-r1:32b"]
    selected_model = st.sidebar.selectbox("AI 모델 선택", model_options, index=0)
    
    # 대화 스타일 설정
    st.sidebar.markdown("## 대화 스타일")
    conversation_style = st.sidebar.selectbox(
        "답변 스타일",
        ["표준", "상세한 설명", "간결한 요약", "전문가 수준"],
        index=0
    )
    
    # 자주 묻는 질문
    st.sidebar.markdown("## 자주 묻는 질문")
    faq_questions = [
        "제플몰에서 가장 인기있는 CPU는 무엇인가요?",
        "게이밍 PC 구성 추천해주세요",
        "그래픽카드 없이 게임을 할 수 있나요?",
        "RAM은 얼마나 필요한가요?"
    ]
    selected_faq = st.sidebar.selectbox("질문 선택", ["선택하세요..."] + faq_questions)
    
    # 대화 내보내기/가져오기 기능
    st.sidebar.markdown("## 대화 관리")
    if st.sidebar.button("대화 내보내기"):
        href = export_chat_history(st.session_state.messages)
        st.sidebar.markdown(href, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("대화 가져오기", type="json")
    if uploaded_file is not None:
        try:
            imported_messages = json.loads(uploaded_file.read())
            st.session_state.messages = imported_messages
            st.sidebar.success("대화 내용을 성공적으로 가져왔습니다!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"파일 가져오기 오류: {e}")

    # 정보 섹션
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 정보")
    st.sidebar.markdown("""
    💬 **제플몰 종합 상담 봇**  
    🔄 버전: 1.0.0  
    👨‍💻 개발: AI 연구팀 김진영 책임
    """)

    # 피드백 섹션
    st.sidebar.markdown("## 피드백")
    feedback = st.sidebar.text_area("의견이나 버그 제보", height=100)
    if st.sidebar.button("제출"):
        if save_feedback(feedback):
            st.sidebar.success("피드백이 제출되었습니다. 감사합니다!")
        else:
            st.sidebar.warning("피드백을 입력해주세요.")
    
    return {
        "max_searches": max_searches,
        "show_search_process": show_search_process,
        "selected_model": selected_model,
        "conversation_style": conversation_style,
        "selected_faq": selected_faq
    }

def apply_custom_styles():
    """앱에 사용자 정의 스타일 적용"""
    st.markdown("""
    <style>
    /* 로그 항목 스타일 */
    .log-entry {
        padding: 8px;
        margin: 6px 0;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    .log-info {
        background-color: #f0f8ff;
        border-left: 3px solid #0066cc;
    }
    .log-warning {
        background-color: #fff8e6;
        border-left: 3px solid #ffc107;
    }
    .log-error {
        background-color: #fff0f0;
        border-left: 3px solid #dc3545;
    }
    .log-success {
        background-color: #f0fff0;
        border-left: 3px solid #28a745;
    }
    
    /* 로그 컨테이너 */
    .log-container {
        max-height: 400px;
        overflow-y: auto;
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

def create_chat_ui():
    """채팅 UI 및 사용자 입력 생성"""
    # 스타일 적용
    apply_custom_styles()
    
    # 채팅 UI 컨테이너
    chat_container = st.container()
    
    # 채팅 내역 표시 (처리 중인 현재 질문은 제외)
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            # 현재 처리 중인 질문이면 표시하지 않음 (중복 방지)
            if (st.session_state.processing and 
                st.session_state.current_question and 
                message["role"] == "user" and 
                message["content"] == st.session_state.current_question and
                i == len(st.session_state.messages) - 1):
                continue
                
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 현재 처리 중인 질문 표시 (처리 중일 때만)
    if st.session_state.processing and st.session_state.current_question:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(st.session_state.current_question)
    
    # 사용자 입력 처리
    if st.session_state.processing:
        # 처리 중일 때는 입력 비활성화
        user_input = None
    else:
        # 사용자 입력 받기
        user_input = st.chat_input("무엇을 도와드릴까요?", key="user_input")
    
    return chat_container, user_input

def display_processing_status(chat_container, agent_system, question, memory, conversation_style):
    """처리 상태 표시 및 에이전트 실행"""
    # 진행 상황 표시를 위한 컨테이너
    progress_container = st.empty()
    
    # 진행 상황 업데이트 함수
    def update_progress(progress, status_text):
        """진행 상황 업데이트 및 로그 저장"""
        progress_container.progress(progress, text=status_text)
        
        # 로그 저장 (세션 상태에 저장)
        if "process_logs" not in st.session_state:
            st.session_state.process_logs = []
        
        # 중복 로그 방지
        if len(st.session_state.process_logs) == 0 or st.session_state.process_logs[-1] != status_text:
            st.session_state.process_logs.append(status_text)
            # 디버깅을 위한 콘솔 출력
            print(f"진행 상황: {progress:.2f} - {status_text}")
    
    # 통합 에이전트 실행
    try:
        # 진행 상황 표시
        update_progress(0.1, "🔍 질문 분석 중...")
        
        # 세션 초기화 및 로그 저장
        if "process_logs" not in st.session_state:
            st.session_state.process_logs = []
        st.session_state.process_logs = []  # 로그 초기화
        
        # 통합 에이전트 가져오기
        integrated_agent = st.session_state.integrated_agent
        
        # 질문 유형 분류
        update_progress(0.3, "🧠 질문 유형 분류 중...")
        
        # 통합 에이전트 실행
        result = integrated_agent.run_workflow(question, chat_history)
        
        # 로그에 수집된 정보 추가
        collected_info = result.get("collected_information", [])
        for info in collected_info:
            update_progress(0.5, info)
        
        query_type = result.get("query_type", "web_search")
        
        # 질문 유형에 따른 상태 업데이트
        query_type_display = {
            "web_search": "🌐 웹 검색",
            "pc_compatibility": "🖥️ PC 부품 호환성 분석",
            "hybrid": "🔄 통합 분석 (웹 검색 + 호환성 분석)"
        }
        
        update_progress(0.7, f"💡 {query_type_display.get(query_type, '알 수 없음')} 기반으로 답변 생성 중...")
        update_progress(0.9, "📝 최종 답변 생성 중...")
        
        # 최종 결과 추출
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        
        # 진행 완료
        update_progress(1.0, "✨ 답변 생성 완료!")
        time.sleep(0.5)  # 완료 메시지를 잠시 표시
        
        # 결과 표시
        with chat_container.chat_message("assistant"):
            st.markdown(answer)
            
            # 쿼리 타입에 따른 표시 정보
            query_type_display = {
                "web_search": "🌐 웹 검색",
                "pc_compatibility": "🖥️ PC 부품 호환성 분석",
                "hybrid": "🔄 통합 분석 (웹 검색 + 호환성 분석)"
            }
            
            with st.expander(f"📊 답변 생성 과정 보기 - {query_type_display.get(query_type, '알 수 없음')}", expanded=False):
                st.markdown(f"**처리 유형:** {query_type_display.get(query_type, '알 수 없음')}")
                
                # 수집된 모든 로그 표시 (세션 상태에서 가져옴)
                if "process_logs" in st.session_state and st.session_state.process_logs:
                    for log in st.session_state.process_logs:
                        st.write(log)
                
                # 추가 정보: 수집된 정보 직접 표시
                st.markdown("### 🔍 수집된 모든 정보")
                if collected_info:
                    for i, info in enumerate(collected_info):
                        st.write(f"{i+1}. {info}")
                else:
                    st.warning("⚠️ 수집된 정보가 없습니다.")
            
            # 메모리에 대화 저장
            memory.save_context({"question": question}, {"answer": answer})
            
            # 메시지 저장
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # 처리 완료 상태로 설정
            st.session_state.processing = False
            st.session_state.current_question = None
            
            return answer

    except Exception as e:
        error_msg = str(e)
        st.error(f"통합 에이전트 실행 중 오류 발생: {error_msg}")
        st.session_state.processing = False
        st.session_state.current_question = None
        return None

def display_server_status(container, check_interval=60):
    """서버 상태 표시"""
    if "last_server_check" not in st.session_state:
        st.session_state.last_server_check = 0
    
    current_time = time.time()
    if current_time - st.session_state.last_server_check > check_interval:
        st.session_state.server_status = check_ollama_server()
        st.session_state.last_server_check = current_time
    
    if st.session_state.server_status:
        container.success("🟢 AI 서버 연결됨")
    else:
        container.error("🔴 AI 서버 연결 끊김")
        if container.button("서버 재연결 시도"):
            if wait_for_ollama_server(max_retries=1):
                st.session_state.server_status = True
                container.success("✅ 서버 연결 복구됨")
                time.sleep(1)
                st.rerun()
            else:
                container.error("❌ 서버에 연결할 수 없습니다")