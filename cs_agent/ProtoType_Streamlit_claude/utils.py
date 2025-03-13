from langchain.memory import ConversationBufferMemory
import streamlit as st
import base64
import json
import time
import requests

def get_memory():
    """Langchain 메모리 객체를 반환하는 함수"""
    if "langchain_memory" not in st.session_state:
        st.session_state.langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
    return st.session_state.langchain_memory

def apply_custom_css():
    """커스텀 CSS를 적용하는 함수 - UI 스타일 정의"""
    st.markdown("""
    <style>
        /* 전체 앱 스타일 - 배경색과 기본 텍스트 색상 */
        .main {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* 헤더 스타일 - 앱 상단의 제목 부분 */
        .stTitleContainer {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        
        /* 채팅 메시지 스타일 - 모든 채팅 메시지 공통 스타일 */
        .stChatMessage {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* 사용자 메시지 스타일 - 사용자가 보낸 메시지 */
        .stChatMessage[data-testid="stChatMessageUser"] {
            background-color: #e9f5ff;
        }
        
        /* 어시스턴트 메시지 스타일 - AI가 보낸 메시지 */
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: #f0f7ff;
        }
        
        /* 로그 컨테이너 스타일 - 로그 항목을 보여주는 영역 */
        .log-entry {
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* 일반 정보 로그 스타일 - 기본 로그 항목 */
        .log-info {
            background-color: #f0f8ff;
            border-left: 3px solid #0066cc;
        }
        
        /* 경고 로그 스타일 - 경고 메시지 */
        .log-warning {
            background-color: #fff8e6;
            border-left: 3px solid #ffc107;
        }
        
        /* 오류 로그 스타일 - 오류 메시지 */
        .log-error {
            background-color: #fff0f0;
            border-left: 3px solid #dc3545;
        }
        
        /* 성공 로그 스타일 - 성공 메시지 */
        .log-success {
            background-color: #f0fff0;
            border-left: 3px solid #28a745;
        }
        
        /* 로그 스크롤 영역 - 로그가 많을 때 스크롤 가능하게 */
        .log-scroll-area {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 10px;
            background-color: #fafafa;
        }
        
        /* 확장된 상태에서의 로그 영역 - expander 컴포넌트 스타일 */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            background-color: #ffffff;
        }
        
        /* 확장된 상태에서의 제목 스타일 - expander 헤더 스타일 */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #333;
            background-color: #f5f7f9;
            border-bottom: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)

def export_chat_history(messages):
    """대화 내용을 JSON으로 변환하고 다운로드 링크 생성"""
    chat_export = json.dumps(messages, ensure_ascii=False)
    b64 = base64.b64encode(chat_export.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="chat_export.json">대화 내용 다운로드</a>'
    return href

def save_feedback(feedback):
    """사용자 피드백을 파일에 저장"""
    if feedback.strip():
        with open("feedback_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {feedback}\n")
        return True
    return False

def check_ollama_server(base_url="http://192.168.110.102:11434"):
    """Ollama 서버 연결 상태 확인"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def wait_for_ollama_server(base_url="http://192.168.110.102:11434", max_retries=3, retry_delay=2):
    """Ollama 서버 연결 대기 (최대 시도 횟수와 지연 시간 지정)"""
    for attempt in range(max_retries):
        if check_ollama_server(base_url):
            return True
        time.sleep(retry_delay)
    return False

# 검색 결과 캐싱 시스템
class SearchCache:
    def __init__(self, cache_size=100):
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []  # LRU 구현을 위한 접근 순서 추적
    
    def get(self, query):
        """캐시에서 검색 결과 가져오기"""
        normalized_query = self._normalize_query(query)
        if normalized_query in self.cache:
            # 접근 순서 업데이트
            self.access_order.remove(normalized_query)
            self.access_order.append(normalized_query)
            return self.cache[normalized_query]
        return None
    
    def set(self, query, result):
        """검색 결과를 캐시에 저장"""
        normalized_query = self._normalize_query(query)
        
        # 캐시 크기 제한 관리
        if len(self.cache) >= self.cache_size and normalized_query not in self.cache:
            # 가장 오래된 항목 제거 (LRU)
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        # 새 항목 추가
        self.cache[normalized_query] = result
        if normalized_query in self.access_order:
            self.access_order.remove(normalized_query)
        self.access_order.append(normalized_query)
    
    def _normalize_query(self, query):
        """쿼리 정규화 (대소문자 무시, 공백 제거 등)"""
        return query.lower().strip()

# 전역 캐시 인스턴스
search_cache = SearchCache()