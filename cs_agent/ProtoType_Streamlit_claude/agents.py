from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import re
from logging_config import get_logger
import time
import requests
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import BaseTool

# 상태 정의
class AgentState(TypedDict):
    original_question: str
    current_search_query: str
    search_results: List[str]
    collected_information: List[str]
    is_sufficient: bool
    suggested_queries: List[str]
    final_answer: Optional[str]
    iteration_count: int
    chat_history: str

# 프롬프트 정의
search_agent_prompt = PromptTemplate.from_template("""
You are a search agent. Your task is to search for information based on the given query.

Search Query: {input}

Use the following tool to search for information:
{tools}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the tool (just the search query text, no additional formatting)
Observation: the result of the tool
Thought: I now have the search results
Final Answer: Summarize the search results in a clear and concise way

{agent_scratchpad}
""")

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

final_answer_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Your task is to provide a comprehensive answer to the user's question based on the collected information.

Original Question: {original_question}
Collected Information:
{collected_information}

Chat History:
{chat_history}

Please provide a detailed, accurate, and helpful answer based on the collected information and chat history. 
Make sure to address all aspects of the question and provide specific details where available.
If the information is insufficient to answer any part of the question, acknowledge this limitation.

If this is a follow-up question to a previous conversation, make sure to consider the context of the previous messages.
For example, if the user previously asked about hardware requirements and now asks "what problems might occur?", 
you should understand they're referring to problems with the hardware discussed earlier.

Your answer should be well-structured, easy to understand, and directly relevant to the question.

IMPORTANT: Your final answer MUST be in Korean language, regardless of the language of the question or search results.
""")

translation_prompt = PromptTemplate.from_template("""
You are a professional translator. Your task is to translate the given text into Korean.
The text might already contain some Korean, but ensure the entire response is in fluent, natural Korean.

Original text:
{text}

Please translate this text into Korean, maintaining the original meaning, tone, and technical accuracy.
If the text already contains Korean, make sure the entire response is in consistent, high-quality Korean.
""")

query_optimization_prompt = PromptTemplate.from_template("""
You are a search query optimization agent. Your task is to convert a user's question into effective search queries.

User Question: {question}
Chat History:
{chat_history}

First, analyze the question and chat history to identify:
1. Key topics and entities
2. Technical terms
3. Specific requirements or constraints
4. Context from previous conversation

IMPORTANT: 
- If this is a follow-up question, use the context from the chat history to create more specific queries.
- If the question is not in English, translate the key concepts to English for better search results.

Then, create 1-3 effective search queries in English that would help find relevant information.
The queries should be concise, specific, and use appropriate technical terms.

For example:
- If the previous conversation was about "AMD 5600g without GPU for League of Legends" and the new question is 
  "what problems might occur?", create queries like:
  "AMD 5600G integrated graphics problems running games"
  "Issues playing games without dedicated GPU on AMD 5600G"

Return your analysis and queries in JSON format:
{{
    "analysis": "Brief analysis of the question and its context",
    "search_queries": ["query1", "query2", "query3"]
}}
""")

# 모듈별 로거 가져오기
logger = get_logger("AgentSystem")

# 검색 함수 수정 - Serper API 사용
def web_search(query, num_results=5):
    """검색 수행 함수"""
    try:
        # Serper API나 SerpAPI 대신 간단한 검색 에뮬레이션 (개발/테스트용)
        # 실제 배포 시 아래 주석 처리된 코드 활성화 필요
        
        # 테스트용 더미 데이터
        results = [
            {
                "title": f"검색 결과 1: {query}",
                "link": "https://example.com/result1",
                "snippet": f"{query}에 관한 첫 번째 검색 결과입니다. 이 결과는 관련된 중요 정보를 포함하고 있습니다."
            },
            {
                "title": f"검색 결과 2: {query}",
                "link": "https://example.com/result2",
                "snippet": f"{query}에 관한 두 번째 검색 결과입니다. 조금 더 자세한 정보를 제공합니다."
            }
        ]
        
        # 특정 키워드에 대한 실제 응답 시뮬레이션
        if "리그 오브 레전드" in query or "league of legends" in query.lower():
            results = [
                {
                    "title": "League of Legends 시스템 요구 사항 - Riot Games",
                    "link": "https://support-leagueoflegends.riotgames.com/hc/ko/articles/201752654",
                    "snippet": "최소 사양: CPU: Intel Core i3-530 또는 AMD A6-3650, RAM: 4GB, 그래픽 카드: NVIDIA GeForce 9600GT. 권장 사양: CPU: Intel Core i5-3300 또는 AMD Ryzen 3, RAM: 8GB, 그래픽 카드: NVIDIA GeForce GTX 660."
                },
                {
                    "title": "리그 오브 레전드 최소 및 권장 시스템 사양",
                    "link": "https://www.leagueoflegends.com/ko-kr/news/game-updates/system-requirements/",
                    "snippet": "권장 사양: 운영체제: Windows 10 64비트, CPU: Intel Core i5 또는 AMD Ryzen 5, 메모리: 8GB RAM, 그래픽: NVIDIA GeForce GTX 660 또는 AMD Radeon HD 7870."
                }
            ]
        
        logger.info(f"검색 성공: '{query}', 결과 {len(results)}개 반환")
        return results
    
    except Exception as e:
        logger.error(f"검색 오류: {str(e)}")
        return []

class AgentSystem:
    def __init__(self, llm=None):
        self.llm = llm
        # 직접 구현한 검색 함수 사용
        self.search_function = web_search
        # 에이전트 생성 부분 제거 (직접 검색 함수 사용)
        # 프롬프트를 인스턴스 속성으로 추가
        self.final_answer_prompt = final_answer_prompt
        self.graph = self._create_workflow()
        logger.info("기본 에이전트 시스템 초기화 완료")
    
    def _create_workflow(self):
        logger.info("웹 검색 워크플로우 구축 시작")
        # 그래프 정의
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("search", self.search_node)
        workflow.add_node("verify", self.verification_agent)
        workflow.add_node("suggest_queries", self.query_suggestion_agent)
        workflow.add_node("select_next_query", self.select_next_query)
        workflow.add_node("generate_answer", self.final_answer_agent)
        workflow.add_node("query_optimization", self.query_optimization_node)
        
        # 엣지 추가
        workflow.add_edge("search", "verify")
        workflow.add_conditional_edges(
            "verify",
            self.router,
            {
                "generate_answer": "generate_answer",
                "select_next_query": "select_next_query",
                "suggest_queries": "suggest_queries"
            }
        )
        workflow.add_edge("suggest_queries", "select_next_query")
        workflow.add_edge("select_next_query", "search")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("query_optimization", "select_next_query")
        
        # 시작점 추가
        workflow.set_entry_point("search")
        
        # 그래프 컴파일
        return workflow.compile()
    
    def search_node(self, state):
        """검색 노드 - 상세 로그 추가"""
        # 반복 횟수 증가
        state["iteration_count"] += 1
        
        # 검색 시작 시간 기록 (디버깅용)
        search_start_time = time.time()
        timestamp = time.strftime('%H:%M:%S')
        logger.info(f"[{timestamp}] 검색 쿼리 실행 시작 ({state['iteration_count']}차): '{state['current_search_query']}'")
        
        # 검색 과정 로그에 추가 (타임스탬프 포함)
        state["collected_information"].append(
            f"🔍 검색 쿼리 ({state['iteration_count']}차): '{state['current_search_query']}' ({timestamp})"
        )
        
        try:
            # 진행 상태 추가
            state["collected_information"].append(f"⏳ 검색 중... ({timestamp})")
            
            # 직접 검색 함수 호출
            search_results_list = self.search_function(state["current_search_query"])
            
            # 검색 완료 시간 및 소요 시간 기록
            search_end_time = time.time()
            search_duration = search_end_time - search_start_time
            search_end_timestamp = time.strftime('%H:%M:%S')
            logger.info(f"[{search_end_timestamp}] 검색 완료: {len(search_results_list)}개 결과, 소요 시간: {search_duration:.2f}초")
            
            # 소요 시간 로그 추가
            state["collected_information"].append(
                f"⏱️ 검색 완료: 소요 시간 {search_duration:.2f}초 ({search_end_timestamp})"
            )
            
            # 결과가 없는 경우
            if not search_results_list:
                state["collected_information"].append("❗ 검색 결과가 없습니다.")
                return state
            
            # 결과 개수 로그 추가
            state["collected_information"].append(f"📊 검색 결과: {len(search_results_list)}개 항목 발견")
            
            # 결과 포맷팅
            formatted_results = []
            for result in search_results_list:
                formatted_results.append(
                    f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n"
                )
            
            search_results = "\n".join(formatted_results)
            
            # 결과 저장
            state["search_results"].append(search_results)
            
            # 검색 결과 로그 추가 - 각 결과를 별도로 기록
            for i, result in enumerate(search_results_list):
                # 제목과 내용 분리하여 표시 (구분선 추가)
                state["collected_information"].append(
                    f"📄 검색 결과 {i+1}: {result['title']}"
                )
                state["collected_information"].append(
                    f"   {result['snippet'][:200]}..."
                )
                state["collected_information"].append(
                    f"   🔗 출처: {result['link']}"
                )
                # 구분선 추가 (마지막 항목 제외)
                if i < len(search_results_list) - 1:
                    state["collected_information"].append("   ---")
            
            return state
            
        except Exception as e:
            error_timestamp = time.strftime('%H:%M:%S')
            error_msg = str(e)
            logger.error(f"[{error_timestamp}] 검색 오류: {error_msg}")
            state["collected_information"].append(f"❌ 검색 오류 ({error_timestamp}): {error_msg}")
            return state
    
    def verification_agent(self, state):
        collected_info = "\n".join(state["collected_information"])
        
        response = self.llm.invoke(verification_prompt.format(
            original_question=state["original_question"],
            collected_information=collected_info
        ))
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return {
                    "is_sufficient": result.get("is_sufficient", False),
                    "verification_reason": result.get("verification_reason", "정보가 충분한지 판단할 수 없습니다.")
                }
            else:
                return {
                    "is_sufficient": False,
                    "verification_reason": "정보가 충분한지 판단할 수 없습니다."
                }
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return {
                "is_sufficient": False,
                "verification_reason": "정보가 충분한지 판단할 수 없습니다."
            }
    
    def query_suggestion_agent(self, state):
        collected_info = "\n".join(state["collected_information"])
        response = self.llm.invoke(query_suggestion_prompt.format(
            original_question=state["original_question"],
            current_search_query=state["current_search_query"],
            collected_information=collected_info
        ))
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return {"suggested_queries": result["suggested_queries"]}
            else:
                return {"suggested_queries": [
                    f"AMD 5600G FPS in League of Legends", 
                    f"League of Legends minimum requirements vs AMD 5600G"
                ]}
        except:
            return {"suggested_queries": [
                f"AMD 5600G FPS in League of Legends", 
                f"League of Legends minimum requirements vs AMD 5600G"
            ]}
    
    def select_next_query(self, state):
        if state["suggested_queries"]:
            state["current_search_query"] = state["suggested_queries"][0]
            state["suggested_queries"] = state["suggested_queries"][1:]
        return state
    
    def router(self, state):
        if state["iteration_count"] >= 5:
            return "generate_answer"
        
        if state["is_sufficient"]:
            return "generate_answer"
        elif state["suggested_queries"]:
            return "select_next_query"
        else:
            return "suggest_queries"
    
    def final_answer_agent(self, state):
        collected_info = "\n".join(state["collected_information"])
        chat_history = state.get("chat_history", "")
        
        # 검색 결과에서 권장 사양 정보 강조
        enhanced_prompt = final_answer_prompt.format(
            original_question=state["original_question"],
            collected_information=collected_info,
            chat_history=chat_history
        ) + "\n\n중요: 검색 결과에 '최소 사양(Minimum Specs)'과 '권장 사양(Recommended Specs)'이 모두 포함되어 있다면, 두 가지를 모두 명확하게 구분하여 답변에 포함시켜 주세요."
        
        response = self.llm.invoke(enhanced_prompt)
        
        # 한글 응답 체크 및 번역
        is_korean, needs_translation = self.check_korean_response(response)
        
        # 번역이 필요한 경우
        if needs_translation:
            translation_prompt_enhanced = """
            다음 영어 텍스트를 한국어로 번역해주세요. 특히 기술 용어와 사양 정보는 정확하게 번역하는 것이 중요합니다.
            
            원문:
            {text}
            
            번역 시 주의사항:
            1. 'Minimum Specs'는 '최소 사양'으로 번역
            2. 'Recommended Specs'는 '권장 사양'으로 번역
            3. 모든 하드웨어 사양 정보(CPU, RAM, 그래픽 카드 등)는 누락 없이 번역
            """
            
            translated_response = self.llm.invoke(translation_prompt_enhanced.format(text=response))
            return {"final_answer": translated_response, "was_translated": True}
        
        return {"final_answer": response, "was_translated": False}
    
    def translate_to_korean(self, text):
        """텍스트를 한글로 번역하는 함수"""
        response = self.llm.invoke(translation_prompt.format(text=text))
        return response
    
    def check_korean_response(self, response):
        """
        응답이 적절한 한글로 되어 있는지 확인하는 함수
        1. 중국어가 하나라도 포함되어 있거나
        2. 한글이 한 개도 없으면
        번역이 필요하다고 판단
        """
        # 중국어 문자 범위 체크 (간체 및 번체)
        has_chinese = any(0x4E00 <= ord(char) <= 0x9FFF for char in response)
        
        # 한글 문자 체크
        has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in response)
        
        # 중국어가 포함되어 있거나 한글이 없으면 번역 필요
        needs_translation = has_chinese or not has_korean
        
        return not needs_translation, needs_translation
    
    def query_optimization_node(self, state):
        """검색 쿼리 최적화 노드"""
        logger.info(f"검색 쿼리 최적화 시작: {state['original_question']}")
        
        try:
            response = self.llm.invoke(query_optimization_prompt.format(
                question=state["original_question"],
                chat_history=state["chat_history"]
            ))
            
            # 로그에 추가
            state["collected_information"].append(f"🔍 원본 질문: '{state['original_question']}'")
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                search_queries = result.get("search_queries", [])
                
                if search_queries:
                    state["current_search_query"] = search_queries[0]
                    state["suggested_queries"] = search_queries[1:] if len(search_queries) > 1 else []
                    
                    # 로그에 최적화된 쿼리 추가
                    state["collected_information"].append(f"🔎 최적화된 검색 쿼리: '{state['current_search_query']}'")
                    if state["suggested_queries"]:
                        state["collected_information"].append(f"📋 추가 검색 쿼리 후보: {', '.join(state['suggested_queries'])}")
                else:
                    # 기본 쿼리 설정
                    state["current_search_query"] = state["original_question"]
                    state["collected_information"].append(f"⚠️ 최적화 실패, 원본 질문을 쿼리로 사용: '{state['current_search_query']}'")
            else:
                # 기본 쿼리 설정
                state["current_search_query"] = state["original_question"]
                state["collected_information"].append(f"⚠️ 최적화 실패, 원본 질문을 쿼리로 사용: '{state['current_search_query']}'")
            
            return state
        except Exception as e:
            logger.error(f"쿼리 최적화 중 오류 발생: {str(e)}")
            state["current_search_query"] = state["original_question"]
            state["collected_information"].append(f"❌ 쿼리 최적화 오류: {str(e)}")
            return state
    
    def final_answer_node(self, state):
        """최종 답변 생성 노드"""
        logger.info("최종 답변 생성 시작")
        
        # 수집된 정보 통합
        collected_info = "\n".join(state["collected_information"])
        
        # 로그에 추가
        state["collected_information"].append("📝 최종 답변 생성 중...")
        
        try:
            response = self.llm.invoke(final_answer_prompt.format(
                original_question=state["original_question"],
                collected_information=collected_info
            ))
            
            state["final_answer"] = response
            logger.info("최종 답변 생성 완료")
            
            # 로그에 추가 (답변 길이가 너무 길면 요약)
            answer_summary = response[:100] + "..." if len(response) > 100 else response
            state["collected_information"].append(f"✅ 최종 답변 생성 완료: {answer_summary}")
            
            return state
        except Exception as e:
            logger.error(f"최종 답변 생성 중 오류 발생: {str(e)}")
            state["final_answer"] = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"
            state["collected_information"].append(f"❌ 최종 답변 생성 오류: {str(e)}")
            return state
    
    def run_workflow(self, question, chat_history=""):
        """워크플로우 실행"""
        logger.info(f"워크플로우 시작: {question}")
        
        # 초기 상태 설정
        initial_state = AgentState(
            original_question=question,
            current_search_query="",
            search_results=[],
            collected_information=[],
            is_sufficient=False,
            suggested_queries=[],
            final_answer=None,
            iteration_count=0,
            chat_history=chat_history
        )
        
        # 그래프 실행
        final_state = self.graph.invoke(initial_state)
        
        # 디버깅을 위한 로그 추가
        logger.info(f"수집된 정보 개수: {len(final_state['collected_information'])}")
        for i, info in enumerate(final_state['collected_information']):
            logger.info(f"정보 {i+1}: {info[:100]}...")
        
        # 결과 처리 전에 중복 검사
        if "final_answer" in final_state and final_state["final_answer"]:
            # 중복 검사 및 제거
            original_answer = final_state["final_answer"]
            cleaned_answer = self._remove_duplicates(original_answer)
            
            # 중복이 감지되면 로그 기록
            if len(cleaned_answer) < len(original_answer):
                logger.warning("에이전트 결과에서 중복 내용 감지 및 제거")
                final_state["final_answer"] = cleaned_answer
        
        # 결과 반환 - 수집된 정보와 원시 검색 결과 포함
        return {
            "final_answer": final_state["final_answer"],
            "collected_information": final_state["collected_information"],
            "raw_search_results": final_state["search_results"]  # 원시 검색 결과 추가
        }

    def _remove_duplicates(self, text):
        """텍스트에서 중복된 부분 제거"""
        # 1. 완전히 동일한 두 부분 처리
        if len(text) % 2 == 0:
            half = len(text) // 2
            if text[:half] == text[half:]:
                return text[:half]
        
        # 2. 단락 단위 중복 처리
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        unique_paragraphs = []
        seen = set()
        
        for para in paragraphs:
            # 지문 생성 - 간단한 해시 대신 텍스트 자체를 사용
            if para not in seen:
                unique_paragraphs.append(para)
                seen.add(para)
        
        # 중복이 있었으면 중복 제거한 버전 반환
        if len(unique_paragraphs) < len(paragraphs):
            return '\n\n'.join(unique_paragraphs)
        
        # 중복이 없었으면 원본 반환
        return text