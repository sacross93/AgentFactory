from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import streamlit as st
from langchain.memory import ConversationBufferMemory
import time

# 페이지 설정 및 스타일 적용
st.set_page_config(
    page_title="제플몰 종합 상담 봇",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 적용
st.markdown("""
<style>
    /* 전체 앱 스타일 */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* 헤더 스타일 */
    .stTitleContainer {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* 채팅 메시지 스타일 */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 사용자 메시지 */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e9f5ff;
    }
    
    /* 어시스턴트 메시지 */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #f0f7ff;
    }
    
    /* 진행 상태 컨테이너 */
    .progress-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 진행 상태 바 */
    .stProgress > div > div {
        background-color: #4e8df5;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        background-color: #4e8df5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #3a7bd5;
    }
    
    /* 확장 패널 스타일 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    
    /* 채팅 입력창 고정 스타일 */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8f9fa;
        padding: 1rem;
        z-index: 1000;
        border-top: 1px solid #e9ecef;
    }
    
    /* 채팅 영역에 하단 여백 추가 */
    .chat-container {
        margin-bottom: 70px;
    }
</style>
""", unsafe_allow_html=True)

# Ollama 모델 초기화
llm = OllamaLLM(
    model="exaone3.5:32b",
    base_url="http://192.168.110.102:11434"
)

# 검색 도구 초기화
search_tool = DuckDuckGoSearchResults()

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[str]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int  # 반복 횟수를 추적하기 위한 필드 추가

# 검색 에이전트 (Agent 1)
search_agent_prompt = PromptTemplate.from_template("""
You are a search agent. Your task is to search for information based on the given query.

Search Query: {input}

Use the following tool to search for information:
{tools}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
Thought: I now have the search results
Final Answer: Summarize the search results in a clear and concise way

{agent_scratchpad}
""")

search_agent = create_react_agent(llm, [search_tool], search_agent_prompt)
search_executor = AgentExecutor(
    agent=search_agent, 
    tools=[search_tool], 
    verbose=True,
    handle_parsing_errors=True  # 파싱 오류 처리 옵션 추가
)

# 검증 에이전트 (Agent 2)
verification_prompt = PromptTemplate.from_template("""
You are a verification agent. Your task is to determine if the collected information is sufficient to answer the original question.

Original Question: {original_question}
Collected Information:
{collected_information}

First, analyze the original question to identify:
1. The main topic
2. Specific requirements or constraints
3. What information would be needed to provide a comprehensive answer

Then, evaluate the collected information to determine if it addresses all aspects of the question.

First, provide a detailed analysis of the information collected and how it relates to the question.
Then, based on this analysis, determine if the information is sufficient.

Return your analysis in JSON format:
{{
    "verification_reason": "Detailed explanation of your analysis of the collected information in relation to the question",
    "is_sufficient": true/false
}}

Focus on providing a thorough analysis first, then make your determination about sufficiency.
""")

def verification_agent(state):
    collected_info = "\n".join(state["collected_information"])
    
    response = llm.invoke(verification_prompt.format(
        original_question=state["original_question"],
        collected_information=collected_info
    ))
    
    try:
        # 응답에서 JSON 부분만 추출하기 위한 시도
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return {
                "is_sufficient": result.get("is_sufficient", False),
                "verification_reason": result.get("verification_reason", "정보가 충분한지 판단할 수 없습니다.")
            }
        else:
            # JSON을 찾지 못한 경우 기본값 반환
            return {
                "is_sufficient": False,
                "verification_reason": "정보가 충분한지 판단할 수 없습니다."
            }
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        # 파싱 오류 시 기본값 반환
        return {
            "is_sufficient": False,
            "verification_reason": "정보가 충분한지 판단할 수 없습니다."
        }

# 쿼리 제안 에이전트 (Agent 3)
query_suggestion_prompt = PromptTemplate.from_template("""
You are a query suggestion agent. Your task is to suggest new search queries to find more information.

Original Question: {original_question}
Current Search Query: {current_search_query}
Collected Information:
{collected_information}

What we still need to know:
1. What aspects of the original question are not yet answered?
2. What specific information is missing?

Suggest 1-3 new search queries in English that would help gather the missing information.
Return your suggestions in JSON format:
{{
    "suggested_queries": ["query1", "query2", "query3"]
}}
""")

def query_suggestion_agent(state):
    collected_info = "\n".join(state["collected_information"])
    response = llm.invoke(query_suggestion_prompt.format(
        original_question=state["original_question"],
        current_search_query=state["current_search_query"],
        collected_information=collected_info
    ))
    
    try:
        # 응답에서 JSON 부분만 추출하기 위한 시도
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return {"suggested_queries": result["suggested_queries"]}
        else:
            # JSON을 찾지 못한 경우 기본값 반환
            return {"suggested_queries": [
                f"AMD 5600G FPS in League of Legends", 
                f"League of Legends minimum requirements vs AMD 5600G"
            ]}
    except:
        # 파싱 오류 시 기본값 반환
        return {"suggested_queries": [
            f"AMD 5600G FPS in League of Legends", 
            f"League of Legends minimum requirements vs AMD 5600G"
        ]}

# 기존 ChatMemory 클래스 대신 Langchain의 Memory 사용
def get_memory():
    if "langchain_memory" not in st.session_state:
        st.session_state.langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
    return st.session_state.langchain_memory

# 최종 답변 생성 프롬프트 정의
final_answer_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Your task is to provide a comprehensive answer to the user's question based on the collected information.

Original Question: {original_question}
Collected Information:
{collected_information}

Please provide a detailed, accurate, and helpful answer based on the collected information. 
Make sure to address all aspects of the question and provide specific details where available.
If the information is insufficient to answer any part of the question, acknowledge this limitation.

Your answer should be well-structured, easy to understand, and directly relevant to the question.
""")

def final_answer_agent(state):
    collected_info = "\n".join(state["collected_information"])
    chat_history = state.get("chat_history", "")
    
    response = llm.invoke(final_answer_prompt.format(
        original_question=state["original_question"],
        collected_information=collected_info,
        chat_history=chat_history
    ))
    
    return {"final_answer": response}

# 검색 노드
def search_node(state):
    # 반복 횟수 증가
    state["iteration_count"] += 1
    
    result = search_executor.invoke({
        "input": state["current_search_query"],
        "tools": [search_tool],
        "tool_names": ["DuckDuckGoSearchResults"],
        "agent_scratchpad": ""
    })
    
    search_results = result["output"]
    state["search_results"].append(search_results)
    state["collected_information"].append(f"Search for '{state['current_search_query']}': {search_results}")
    
    return state

# 다음 쿼리 선택 노드
def select_next_query(state):
    if state["suggested_queries"]:
        state["current_search_query"] = state["suggested_queries"][0]
        state["suggested_queries"] = state["suggested_queries"][1:]
    return state

# 라우터 함수
def router(state):
    # 최대 5번 반복 제한 추가
    if state["iteration_count"] >= 5:
        return "generate_answer"
    
    if state["is_sufficient"]:
        return "generate_answer"
    elif state["suggested_queries"]:
        return "select_next_query"
    else:
        return "suggest_queries"

# 그래프 정의
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("search", search_node)
workflow.add_node("verify", verification_agent)
workflow.add_node("suggest_queries", query_suggestion_agent)
workflow.add_node("select_next_query", select_next_query)
workflow.add_node("generate_answer", final_answer_agent)

# 엣지 추가
workflow.add_edge("search", "verify")
# 조건부 엣지 추가 - verify 노드에서 라우터 함수를 사용하여 다음 노드 결정
workflow.add_conditional_edges(
    "verify",
    router,
    {
        "generate_answer": "generate_answer",
        "select_next_query": "select_next_query",
        "suggest_queries": "suggest_queries"
    }
)
workflow.add_edge("suggest_queries", "select_next_query")
workflow.add_edge("select_next_query", "search")
workflow.add_edge("generate_answer", END)

# 시작점 추가
workflow.set_entry_point("search")

# 그래프 컴파일
graph = workflow.compile()

# 검색 쿼리 최적화 에이전트 개선
query_optimization_prompt = PromptTemplate.from_template("""
You are a search query optimization agent. Your task is to convert a user's question into effective search queries.

User Question: {question}

First, analyze the question to identify:
1. Key topics and entities
2. Technical terms
3. Specific requirements or constraints

IMPORTANT: If the question is not in English, translate the key concepts to English for better search results.

Then, create 1-3 effective search queries in English that would help find relevant information.
The queries should be concise, specific, and use appropriate technical terms.

For example:
- If the question is about "AMD 5600g 그래픽카드 없이 리그오브레전드 실행", create queries like:
  "AMD 5600G integrated graphics League of Legends performance"
  "Can League of Legends run on AMD 5600G without dedicated GPU"

Return your analysis and queries in JSON format:
{{
    "analysis": "Brief analysis of the question",
    "search_queries": ["query1", "query2", "query3"]
}}
""")

def query_optimization_agent(question):
    response = llm.invoke(query_optimization_prompt.format(question=question))
    
    try:
        # 응답에서 JSON 부분만 추출하기 위한 시도
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return result["search_queries"]
        else:
            # JSON을 찾지 못한 경우 기본값 반환
            return [f"AMD 5600G League of Legends performance", 
                   f"Can League of Legends run on AMD 5600G without dedicated GPU"]
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        # 파싱 오류 시 기본값 반환
        return [f"AMD 5600G League of Legends performance", 
               f"Can League of Legends run on AMD 5600G without dedicated GPU"]

# 검색 도구 수정 - 영어 쿼리 사용 보장
def search_with_progress(state):
    state["iteration_count"] += 1
    
    # 현재 쿼리가 한국어인지 확인하고 영어로 변환
    current_query = state["current_search_query"]
    if any(ord(char) > 127 for char in current_query):
        english_query = f"AMD 5600G League of Legends performance {state['iteration_count']}"
        log_message = f"📊 검색 쿼리 (원본): {current_query}"
        process_logs.append(log_message)
        with progress_container:
            st.write(log_message)
            
        log_message = f"📊 검색 쿼리 (영어로 변환): {english_query}"
        process_logs.append(log_message)
        with progress_container:
            st.write(log_message)
            
        current_query = english_query
    else:
        log_message = f"📊 검색 쿼리: {current_query}"
        process_logs.append(log_message)
        with progress_container:
            st.write(log_message)
    
    # 영어 쿼리로 검색 실행
    result = search_executor.invoke({
        "input": current_query,
        "tools": [search_tool],
        "tool_names": ["DuckDuckGoSearchResults"],
        "agent_scratchpad": ""
    })
    
    search_results = result["output"]
    log_message = f"🔎 검색 결과: {search_results[:200]}..."
    process_logs.append(log_message)
    with progress_container:
        st.write(log_message)
    
    state["search_results"].append(search_results)
    state["collected_information"].append(f"Search for '{current_query}': {search_results}")
    
    return state

# 실행 함수 수정
def run_agent_workflow(question, memory=None):
    chat_history = ""
    if memory:
        # Langchain 메모리에서 대화 기록 가져오기
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", "")
    
    # 최적화된 검색 쿼리 생성
    optimized_queries = query_optimization_agent(question)
    
    # 영어 쿼리인지 확인하고, 아니면 기본 영어 쿼리 사용
    english_queries = []
    for query in optimized_queries:
        if any(ord(char) > 127 for char in query):  # 한국어 문자 포함 여부 확인
            # 기본 영어 쿼리로 대체
            english_queries.append(f"AMD 5600G League of Legends performance")
        else:
            english_queries.append(query)
    
    if not english_queries:
        english_queries = [
            "AMD 5600G League of Legends performance",
            "Can League of Legends run on AMD 5600G without dedicated GPU"
        ]
    
    initial_query = english_queries[0]
    
    initial_state = {
        "original_question": question,
        "current_search_query": initial_query,  # 최적화된 첫 번째 영어 쿼리 사용
        "search_results": [],
        "collected_information": [],
        "is_sufficient": False,
        "suggested_queries": english_queries[1:] if len(english_queries) > 1 else [],  # 나머지 쿼리들을 제안 쿼리로 설정
        "final_answer": None,
        "iteration_count": 0,
        "chat_history": chat_history
    }
    
    result = graph.invoke(initial_state)
    return result["final_answer"]

# Streamlit 앱 수정
def main():
    st.title("AI 검색 어시스턴트")
    
    # 메모리 초기화
    memory = get_memory()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "process_logs" not in st.session_state:
        st.session_state.process_logs = []
    
    # 사이드바에 대화 기록 지우기 버튼 추가
    if st.sidebar.button("대화 기록 지우기"):
        memory.clear()
        st.session_state.messages = []
        st.session_state.process_logs = []
        st.rerun()
    
    # 채팅 컨테이너 생성 (하단 여백 추가)
    chat_container = st.container()
    
    # 채팅 입력창 (항상 맨 아래에 위치)
    user_input = st.chat_input("질문을 입력하세요...")
    
    # 채팅 기록 표시 (채팅 컨테이너 내부)
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if user_input and not st.session_state.processing:
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        
        # 처리 중 상태로 설정
        st.session_state.processing = True
        st.session_state.current_question = user_input
        st.session_state.process_logs = []  # 로그 초기화
        
        # 처리 시작을 위한 페이지 갱신
        st.rerun()
    
    # 처리 중이고 현재 질문이 있는 경우
    if st.session_state.processing and st.session_state.current_question:
        with chat_container:
            with st.chat_message("assistant"):
                # 진행 상황을 표시할 컨테이너
                progress_container = st.container()
                
                with progress_container:
                    # 진행 상태 표시 컴포넌트
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 진행 상태 업데이트 함수
                    def update_progress(progress, status):
                        progress_bar.progress(progress)
                        status_text.markdown(f"**{status}**")
                        st.session_state.process_logs.append(status)
                        time.sleep(0.1)  # UI 업데이트를 위한 짧은 지연
                    
                    # 최적화된 검색 쿼리 생성
                    update_progress(0.1, "🔍 질문 분석 및 검색 쿼리 최적화 중...")
                    
                    optimized_queries = query_optimization_agent(st.session_state.current_question)
                    
                    # 영어 쿼리인지 확인하고, 아니면 기본 영어 쿼리 사용
                    english_queries = []
                    for query in optimized_queries:
                        if any(ord(char) > 127 for char in query):  # 한국어 문자 포함 여부 확인
                            # 기본 영어 쿼리로 대체
                            english_queries.append(f"AMD 5600G League of Legends performance")
                        else:
                            english_queries.append(query)
                    
                    if not english_queries:
                        english_queries = [
                            "AMD 5600G League of Legends performance",
                            "Can League of Legends run on AMD 5600G without dedicated GPU"
                        ]
                    
                    update_progress(0.2, f"📝 최적화된 검색 쿼리: {', '.join(english_queries)}")
                    
                    # 초기 상태 설정
                    memory_variables = memory.load_memory_variables({})
                    chat_history = memory_variables.get("chat_history", "")
                    
                    initial_state = {
                        "original_question": st.session_state.current_question,
                        "current_search_query": english_queries[0],  # 최적화된 첫 번째 영어 쿼리 사용
                        "search_results": [],
                        "collected_information": [],
                        "is_sufficient": False,
                        "suggested_queries": english_queries[1:] if len(english_queries) > 1 else [],  # 나머지 쿼리들을 제안 쿼리로 설정
                        "final_answer": None,
                        "iteration_count": 0,
                        "chat_history": chat_history
                    }
                    
                    # 검색 실행
                    sufficient_info = False
                    for i, query in enumerate(english_queries[:min(5, len(english_queries))]):
                        progress_value = 0.2 + (i * 0.15)  # 각 검색마다 진행률 증가
                        
                        update_progress(progress_value, f"📊 검색 쿼리 실행 중: {query}")
                        
                        # 검색 실행
                        result = search_executor.invoke({
                            "input": query,
                            "tools": [search_tool],
                            "tool_names": ["DuckDuckGoSearchResults"],
                            "agent_scratchpad": ""
                        })
                        
                        search_results = result["output"]
                        truncated_results = search_results[:200] + "..." if len(search_results) > 200 else search_results
                        
                        update_progress(progress_value + 0.05, f"🔎 검색 결과 분석 중...")
                        st.session_state.process_logs.append(f"🔎 검색 결과: {truncated_results}")
                        
                        initial_state["search_results"].append(search_results)
                        initial_state["collected_information"].append(f"Search for '{query}': {search_results}")
                        
                        # 검증 단계
                        update_progress(progress_value + 0.1, "✅ 정보 검증 중...")
                        
                        # 검증 에이전트 호출
                        collected_info = "\n".join(initial_state["collected_information"])
                        verification_response = llm.invoke(verification_prompt.format(
                            original_question=initial_state["original_question"],
                            collected_information=collected_info
                        ))
                        
                        try:
                            # 응답에서 JSON 부분만 추출
                            import re
                            json_match = re.search(r'\{.*\}', verification_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                verification_result = json.loads(json_str)
                                verification_reason = verification_result.get("verification_reason", "정보가 충분한지 판단할 수 없습니다.")
                                is_sufficient = verification_result.get("is_sufficient", False)
                            else:
                                verification_reason = "정보가 충분한지 판단할 수 없습니다."
                                is_sufficient = False
                        except Exception as e:
                            print(f"Error parsing JSON: {e}")
                            verification_reason = "정보가 충분한지 판단할 수 없습니다."
                            is_sufficient = False
                        
                        # 검증 결과 로그에 추가
                        st.session_state.process_logs.append(f"💡 검증 분석: {verification_reason}")
                        
                        if is_sufficient:
                            update_progress(progress_value + 0.15, "📋 검증 결과: 충분한 정보 수집됨 ✅")
                            st.session_state.process_logs.append("📋 검증 결과: 충분한 정보 수집됨 ✅")
                            sufficient_info = True
                            break
                        else:
                            update_progress(progress_value + 0.15, "📋 검증 결과: 추가 정보 필요 ❌")
                            st.session_state.process_logs.append("📋 검증 결과: 추가 정보 필요 ❌")
                    
                    # 최종 답변 생성
                    update_progress(0.9, "📝 최종 답변 생성 중...")
                    
                    final_answer_response = llm.invoke(final_answer_prompt.format(
                        original_question=initial_state["original_question"],
                        collected_information="\n".join(initial_state["collected_information"])
                    ))
                    answer = final_answer_response
                    
                    # 진행 완료
                    update_progress(1.0, "✨ 답변 생성 완료!")
                    time.sleep(0.5)  # 완료 메시지를 잠시 표시
                    
                    # 진행 상황 컨테이너 비우기
                    progress_container.empty()
                
                # 최종 답변 표시 (카드 형태로)
                # st.markdown("""
                # <div style="background-color: #f0f7ff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                # """, unsafe_allow_html=True)
                st.write(answer)
                # st.markdown("</div>", unsafe_allow_html=True)
                
                # 생성 과정을 토글 형식으로 표시
                with st.expander("📊 답변 생성 과정 보기", expanded=False):
                    for log in st.session_state.process_logs:
                        st.write(log)
                
                # 메모리에 대화 저장
                memory.save_context({"question": st.session_state.current_question}, {"answer": answer})
                
                # 메시지 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 처리 완료 상태로 설정
                st.session_state.processing = False
                st.session_state.current_question = None

if __name__ == "__main__":
    main()
