# app.py
"""Streamlit 기반 메인 애플리케이션."""

import streamlit as st
from langchain.memory import ConversationBufferMemory
from workflow import graph, AgentState
from utils import sanitize_input, log_error, cache_search_results
import time

# 페이지 설정
st.set_page_config(page_title="제플몰 종합 상담 봇", page_icon="🔍", layout="wide")
st.markdown("<style>/* CSS 생략 */</style>", unsafe_allow_html=True)

conversation_style = st.sidebar.selectbox(
    "답변 스타일",
    ["표준", "상세한 설명", "간결한 요약", "전문가 수준"],
    index=0
)

# 메모리 초기화
def get_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return st.session_state.memory

@cache_search_results
def run_workflow(question: str, memory) -> str:
    """워크플로우 실행."""
    try:
        chat_history = memory.load_memory_variables({}).get("chat_history", "")
        initial_state: AgentState = {
            "original_question": question,
            "current_search_query": "AMD 5600G performance",
            "search_results": [],
            "collected_information": [],
            "is_sufficient": False,
            "suggested_queries": [],
            "final_answer": None,
            "iteration_count": 0,
            "chat_history": chat_history
        }
        result = graph.invoke(initial_state)
        return result["final_answer"]
    except Exception as e:
        log_error(e, "Workflow execution")
        return "오류가 발생했습니다. 다시 시도해주세요."

def main():
    st.title("🔍 제플몰 종합 상담 봇")
    memory = get_memory()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 사이드바 설정 생략 (기존과 유사)
    user_input = st.chat_input("질문을 입력하세요...")

    with st.container():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if user_input and not st.session_state.processing:
        try:
            sanitized_input = sanitize_input(user_input)
            st.session_state.messages.append({"role": "user", "content": sanitized_input})
            st.session_state.processing = True

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    answer = run_workflow(sanitized_input, memory)
                    st.write(answer)
                    memory.save_context({"input": sanitized_input}, {"output": answer})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error("예상치 못한 오류가 발생했습니다.")
            log_error(e, "Main processing")
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()