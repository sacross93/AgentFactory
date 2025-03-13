from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import re
import logging
from datetime import datetime
import time
import traceback

# 기존 에이전트 임포트
from agents import AgentSystem
from pc_check_agent import process_pc_compatibility_query
import config
from utils import search_cache

# 로깅 설정
from logging_config import get_logger

# 모듈별 로거 가져오기
logger = get_logger("IntegratedAgent")

# 상태 정의
class IntegratedAgentState(TypedDict):
    question: str
    chat_history: str
    query_type: str  # "web_search", "pc_compatibility", "hybrid"
    web_search_results: Optional[Dict[str, Any]]
    pc_compatibility_results: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    errors: List[str]
    collected_information: List[str]  # 검색 과정 정보 추가
    search_keywords: List[str]
    web_search_queries: List[str]

# 질문 분류 프롬프트
query_classification_prompt = PromptTemplate.from_template("""
You are a query classification agent. Your task is to classify the user's question into one of the following categories:
1. "web_search" - Questions about software requirements, gaming performance, or general PC knowledge that require web search.
2. "pc_compatibility" - Questions about PC part compatibility, hardware specifications, or component recommendations.
3. "hybrid" - Questions that need both web search and PC compatibility analysis.

User question: {question}
Chat history: {chat_history}

Analyze the question carefully. If it mentions specific PC parts and their compatibility, it's likely "pc_compatibility".
If it's about software requirements or general knowledge, it's likely "web_search".
If it requires both hardware compatibility check and software performance information, it's "hybrid".

Examples:
- "롤을 할 수 있는 최소 사양이 뭐야?" -> "web_search" (This asks about minimum requirements for League of Legends)
- "AMD 5600G와 호환되는 메인보드 추천해줘" -> "pc_compatibility" (This asks about motherboard compatibility with AMD 5600G)
- "RTX 3070으로 배틀필드 2042를 풀옵션으로 할 수 있을까?" -> "hybrid" (This requires both hardware analysis and game requirements)

Return your classification as a JSON object with these fields:
- "query_type": one of "web_search", "pc_compatibility", or "hybrid"
- "reason": brief explanation for your classification

JSON:
""")

class IntegratedAgentGraph:
    def __init__(self, llm: OllamaLLM):
        """통합 에이전트 그래프 초기화 - 상태 관리 개선"""
        self.llm = llm
        self.agent_system = AgentSystem(llm)
        self.graph = self._create_graph()
        
        # 기본 상태 템플릿 - 모든 노드가 참조할 수 있는 초기 상태 정의
        self.default_state = {
            "search_keywords": [],
            "web_search_queries": [],
            "errors": [],
            "collected_information": []
        }
        logger.info("통합 에이전트 초기화 완료")
    
    def _create_graph(self):
        # 그래프 정의
        workflow = StateGraph(IntegratedAgentState)
        
        # 노드 추가
        workflow.add_node("classify_question", self.classify_question)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("pc_compatibility", self.pc_compatibility_node)
        workflow.add_node("hybrid_processing", self.hybrid_processing_node)
        workflow.add_node("generate_final_answer", self.generate_final_answer)
        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("hybrid_analysis", self.hybrid_analysis_node)
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "classify_question",
            self.route_by_query_type,
            {
                "web_search": "web_search",
                "pc_compatibility": "pc_compatibility",
                "hybrid": "hybrid_processing"
            }
        )
        
        # 결과 통합을 위한 엣지 추가
        workflow.add_edge("web_search", "generate_final_answer")
        workflow.add_edge("pc_compatibility", "generate_final_answer")
        workflow.add_edge("hybrid_processing", "generate_final_answer")
        workflow.add_edge("generate_queries", "hybrid_analysis")
        workflow.add_edge("hybrid_analysis", "generate_final_answer")
        workflow.add_edge("generate_final_answer", END)
        
        # 시작점 지정
        workflow.set_entry_point("classify_question")
        
        # 그래프 컴파일
        return workflow.compile()
    
    def classify_question(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """질문 유형 분류 노드"""
        logger.info(f"질문 분류 시작: {state['question']}")
        
        # 로그 추가
        state["collected_information"].append(f"🔍 질문 분석: '{state['question']}'")
        
        try:
            response = self.llm.invoke(query_classification_prompt.format(
                question=state["question"],
                chat_history=state["chat_history"]
            ))
            
            # JSON 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                query_type = result.get("query_type", "web_search")
                reason = result.get("reason", "No reason provided")
                
                # 로그 추가
                state["collected_information"].append(f"🧠 질문 유형: {query_type} - {reason}")
                
                state["query_type"] = query_type
                logger.info(f"질문 분류 완료: {query_type} - {reason}")
                return state
            else:
                # 기본값 설정
                state["query_type"] = "web_search"
                state["collected_information"].append("⚠️ 질문 유형 분류 실패, 기본값(web_search) 사용")
                logger.warning("JSON 추출 실패, 기본 유형(web_search) 사용")
                return state
        except Exception as e:
            # 오류 처리
            logger.error(f"질문 분류 중 오류 발생: {str(e)}")
            state["errors"].append(f"질문 분류 오류: {str(e)}")
            state["query_type"] = "web_search"
            state["collected_information"].append(f"❌ 질문 유형 분류 오류: {str(e)}")
            return state
    
    def route_by_query_type(self, state: IntegratedAgentState) -> str:
        """질문 유형에 따라 처리 경로 결정"""
        return state["query_type"]
    
    def web_search_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """웹 검색 기반 처리 - 캐싱 및 재시도 로직 추가"""
        start_time = time.time()
        logger.info(f"웹 검색 수행: {state['question']}")
        
        # 로그 추가 (타임스탬프 포함)
        state["collected_information"].append(
            f"🌐 웹 검색 시작: '{state['question']}' ({time.strftime('%H:%M:%S')})"
        )
        
        # 캐시에서 결과 확인
        cached_result = search_cache.get(state['question'])
        if cached_result:
            logger.info("캐시에서 검색 결과 사용")
            state["collected_information"].append("🔄 캐시된 검색 결과 사용")
            
            # 캐시된 결과에서 수집 정보 추출
            if "collected_information" in cached_result:
                for info in cached_result["collected_information"]:
                    state["collected_information"].append(info)
            
            state["web_search_results"] = cached_result
            return state
        
        # 최대 재시도 횟수
        max_retries = 3
        retry_delay = 2  # 초 단위
        
        for attempt in range(max_retries):
            try:
                # 검색 쿼리 생성 로그
                state["collected_information"].append(f"🔎 검색 쿼리 생성 중... (시도 {attempt+1}/{max_retries})")
                
                # 기존 에이전트 시스템 활용
                result = self.agent_system.run_workflow(
                    state["question"], 
                    state["chat_history"]
                )
                
                # 실행 시간 기록
                end_time = time.time()
                execution_time = end_time - start_time
                state["collected_information"].append(f"⏱️ 검색 실행 시간: {execution_time:.2f}초")
                
                # 결과 캐싱
                search_cache.set(state['question'], result)
                
                # 검색 과정 정보 추가
                if "collected_information" in result:
                    for info in result["collected_information"]:
                        state["collected_information"].append(info)
                
                state["web_search_results"] = result
                logger.info("웹 검색 완료")
                state["collected_information"].append("✅ 웹 검색 완료")
                return state
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"검색 시도 {attempt+1}/{max_retries} 실패: {error_msg}")
                state["collected_information"].append(f"⚠️ 검색 시도 {attempt+1}/{max_retries} 실패: {error_msg}")
                
                if attempt < max_retries - 1:
                    state["collected_information"].append(f"⏳ {retry_delay}초 후 재시도 중...")
                    time.sleep(retry_delay)
                else:
                    state["errors"].append(f"검색 실패: {error_msg}")
                    state["collected_information"].append("❌ 최대 재시도 횟수 초과, 검색 실패")
        
        # 모든 시도 실패 시
        state["web_search_results"] = {
            "final_answer": "검색 과정에서 오류가 발생했습니다. 다른 방식으로 질문을 시도해 보세요.",
            "collected_information": state["collected_information"]
        }
        return state
    
    def pc_compatibility_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """PC 부품 호환성 분석 - 디버깅 강화"""
        start_time = time.time()
        # 로그 추가 - 타임스탬프 포함
        timestamp = time.strftime('%H:%M:%S')
        state["collected_information"].append(f"🖥️ PC 호환성 분석 시작 ({timestamp})")
        logger.info(f"[{timestamp}] PC 호환성 분석: {state['question']}")
        
        try:
            # PC 호환성 분석 실행
            state["collected_information"].append(f"⚙️ 부품 정보 추출 중... ({timestamp})")
            
            # PC 호환성 모듈 직접 호출 시 모든 필요 정보 전달
            pc_input = {
                "question": state["question"],
                "search_keywords": state.get("search_keywords", []),
                "errors": []
            }
            
            # search_keywords 키가 없는 경우를 대비하여 기본값 설정
            try:
                from pc_check_agent import process_pc_compatibility_query
                
                # 모든 필수 필드가 있는 입력 전달
                pc_result = process_pc_compatibility_query(
                    state["question"], 
                    input_state=pc_input  # 필요한 모든 상태 변수 전달
                )
                
                # 결과 저장
                state["pc_compatibility_results"] = pc_result
                
                # 분석 결과 요약 로그 추가
                if pc_result and pc_result.get("explanation"):
                    state["collected_information"].append(f"✅ PC 호환성 분석 결과 확인: {len(pc_result.get('explanation', ''))}자")
                else:
                    state["collected_information"].append("⚠️ PC 호환성 분석 결과가 비어있습니다")
                
                return state
            except Exception as e:
                error_msg = f"PC 호환성 분석 오류: {str(e)}"
                state["errors"].append(error_msg)
                state["collected_information"].append(f"❌ {error_msg}")
                logger.error(error_msg)
                
                # 기본 결과 생성
                state["pc_compatibility_results"] = {
                    "explanation": "PC 호환성 분석 중 오류가 발생했습니다."
                }
                return state
        except Exception as e:
            error_msg = str(e)
            timestamp_error = time.strftime('%H:%M:%S')
            logger.error(f"[{timestamp_error}] PC 호환성 분석 오류: {error_msg}")
            state["collected_information"].append(f"❌ 호환성 분석 오류 ({timestamp_error}): {error_msg}")
            state["errors"].append(f"PC 호환성 분석 오류: {error_msg}")
            return state
    
    def hybrid_processing_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """웹 검색과 호환성 검사 모두 수행 - 완전히 독립적인 실행"""
        logger.info(f"하이브리드 처리 수행: {state['question']}")
        
        try:
            # 웹 검색 수행 - 독립적인 상태로 실행
            web_search_state = {
                "question": state["question"],
                "chat_history": state["chat_history"],
                "query_type": "web_search",
                "errors": [],
                "collected_information": []
            }
            
            state["collected_information"].append("🌐 웹 검색 수행 중...")
            
            try:
                # 웹 검색 에이전트 직접 실행
                web_result = self.agent_system.run_workflow(
                    state["question"], 
                    state["chat_history"]
                )
                state["web_search_results"] = web_result
                state["collected_information"].append("✅ 웹 검색 결과 확인")
            except Exception as web_error:
                error_msg = f"웹 검색 오류: {str(web_error)}"
                state["errors"].append(error_msg)
                state["collected_information"].append(f"❌ {error_msg}")
                logger.error(error_msg)
            
            # PC 호환성 검사 - 완전히 독립적으로 실행
            state["collected_information"].append("🖥️ PC 호환성 검사 모듈 호출 중...")
            
            try:
                # 모든 의존성을 제거하고 독립적으로 실행
                import pc_check_agent
                from importlib import reload
                
                # 모듈 리로드로 이전 상태에 영향을 받지 않게 함
                reload(pc_check_agent)
                
                # search_keywords 오류 방지를 위해 직접 질문만 전달
                pc_result = pc_check_agent.process_pc_compatibility_query(state["question"])
                
                # 결과 로깅
                if pc_result and pc_result.get("explanation"):
                    state["collected_information"].append(f"✅ PC 호환성 분석 결과 확인: {len(pc_result.get('explanation', ''))}자")
                else:
                    state["collected_information"].append("⚠️ PC 호환성 분석 결과가 비어있습니다")
                    
                state["pc_compatibility_results"] = pc_result
            except Exception as pc_error:
                # 오류 상세 정보 캡처
                import traceback
                error_msg = f"PC 호환성 검사 오류: {str(pc_error)}"
                trace_msg = traceback.format_exc()
                state["errors"].append(error_msg)
                state["collected_information"].append(f"❌ {error_msg}")
                logger.error(f"{error_msg}\n{trace_msg}")
                
                # 기본 결과 생성
                state["pc_compatibility_results"] = {
                    "explanation": "PC 호환성 분석 중 오류가 발생했습니다."
                }
            
            logger.info("하이브리드 처리 완료")
            return state
            
        except Exception as e:
            logger.error(f"하이브리드 처리 중 오류 발생: {str(e)}")
            state["errors"].append(f"하이브리드 처리 오류: {str(e)}")
            return state
    
    def generate_queries_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """검색 쿼리 생성 노드 - 에러 처리 강화"""
        try:
            logger.info(f"검색 쿼리 생성 시작: {state['question']}")
            
            # 초기 상태 확인 및 초기화 - 누락된 경우를 대비
            if 'search_keywords' not in state:
                # 검색 키워드 초기화
                state['search_keywords'] = []
                logger.warning("'search_keywords' 키가 상태에 없어서 초기화했습니다.")
            
            # 질문에서 키워드 추출을 위한 프롬프트
            keyword_extraction_prompt = PromptTemplate.from_template("""
            다음 질문에서 중요한 검색 키워드를 추출해주세요:
            질문: {question}
            
            최대 5개의 키워드만 추출하고, 쉼표로 구분된 목록으로 반환하세요.
            각 키워드는 1-3단어로 구성되어야 합니다.
            """)
            
            # 키워드 추출 체인
            keyword_extraction_chain = keyword_extraction_prompt | self.llm | StrOutputParser()
            
            # 키워드 추출 실행
            keywords_result = keyword_extraction_chain.invoke({"question": state["question"]})
            
            # 로그 추가
            state["collected_information"].append(f"🔑 추출된 키워드: {keywords_result}")
            
            # 키워드 정리 및 저장
            keywords = [kw.strip() for kw in keywords_result.split(",")]
            state["search_keywords"] = keywords
            
            # 검색 쿼리 생성 로직
            search_query_prompt = PromptTemplate.from_template("""
            다음 질문과 키워드를 기반으로 효과적인 웹 검색 쿼리를 2-3개 생성해주세요:
            질문: {question}
            키워드: {keywords}
            
            각 쿼리는 웹 검색 엔진에서 좋은 결과를 반환할 수 있도록 간결하고 명확해야 합니다.
            쉼표로 구분된 목록으로 반환하세요.
            """)
            
            # 쿼리 생성 체인
            search_query_chain = search_query_prompt | self.llm | StrOutputParser()
            
            # 쿼리 생성 실행
            query_result = search_query_chain.invoke({
                "question": state["question"],
                "keywords": ", ".join(state["search_keywords"])
            })
            
            # 로그 추가
            state["collected_information"].append(f"🔍 생성된 검색 쿼리: {query_result}")
            
            # 쿼리 정리 및 저장
            queries = [q.strip() for q in query_result.split(",")]
            
            # 하이브리드 분석용 쿼리 저장
            state["web_search_queries"] = queries[:3]  # 최대 3개 쿼리만 사용
            
            return state
        except Exception as e:
            error_msg = f"쿼리 생성 중 오류: {str(e)}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
            
            # 오류 정보 저장 및 기본값 설정
            state["errors"].append(error_msg)
            state["collected_information"].append(f"❌ {error_msg}")
            
            # 기본 검색 쿼리 설정 (복구 메커니즘)
            if "web_search_queries" not in state or not state["web_search_queries"]:
                state["web_search_queries"] = [state["question"]]  # 기본 쿼리로 원본 질문 사용
                state["collected_information"].append("⚠️ 기본 검색 쿼리를 사용합니다")
            
            return state

    def hybrid_analysis_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """하이브리드 분석 노드 - 오류 대응 강화"""
        start_time = time.time()
        try:
            timestamp = time.strftime('%H:%M:%S')
            state["collected_information"].append(f"🔄 하이브리드 분석 시작 ({timestamp})")
            logger.info(f"[{timestamp}] 하이브리드 분석: {state['question']}")
            
            # 웹 검색 부분
            try:
                # 검색 키워드 확인 및 복구
                if "search_keywords" not in state or not state["search_keywords"]:
                    # 누락된 경우 복구 시도
                    state = self.generate_queries_node(state)
                
                # 웹 검색 실행
                state["collected_information"].append(f"🌐 웹 검색 수행 중... ({time.strftime('%H:%M:%S')})")
                state = self.web_search_node(state)
            except Exception as e:
                error_msg = f"하이브리드 분석 중 웹 검색 오류: {str(e)}"
                state["errors"].append(error_msg)
                state["collected_information"].append(f"❌ {error_msg}")
                logger.error(error_msg)
            
            # PC 호환성 분석 부분
            try:
                state["collected_information"].append(f"🖥️ PC 호환성 분석 중... ({time.strftime('%H:%M:%S')})")
                state = self.pc_compatibility_node(state)
            except Exception as e:
                error_msg = f"하이브리드 분석 중 호환성 분석 오류: {str(e)}"
                state["errors"].append(error_msg)
                state["collected_information"].append(f"❌ {error_msg}")
                logger.error(error_msg)
            
            # 최종 답변 생성
            state["collected_information"].append(f"📝 통합 분석 결과 종합 중... ({time.strftime('%H:%M:%S')})")
            
            # 처리 시간 계산
            end_time = time.time()
            duration = end_time - start_time
            state["collected_information"].append(f"⏱️ 하이브리드 분석 소요 시간: {duration:.2f}초")
            
            return state
        except Exception as e:
            error_msg = f"하이브리드 분석 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["collected_information"].append(f"❌ {error_msg}")
            
            # 성능 관련 정보 추가
            end_time = time.time()
            duration = end_time - start_time
            state["collected_information"].append(f"⏱️ 하이브리드 분석 실패 시간: {duration:.2f}초")
            
            return state
    
    def generate_final_answer(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """최종 결과 생성 - 중복 방지 강화"""
        logger.info(f"최종 답변 생성 시작")
        
        query_type = state["query_type"]
        question = state["question"]
        
        try:
            # 검색 결과와 호환성 결과 추출
            web_results = state.get("web_search_results", None)
            pc_results = state.get("pc_compatibility_results", None)
            
            # 웹 검색 결과가 None인 경우 처리
            web_answer = "웹 검색 결과를 찾을 수 없습니다."
            if web_results is not None:
                web_answer = web_results.get("final_answer", web_answer)
            
            # PC 호환성 결과가 None인 경우 처리
            pc_explanation = "PC 부품 호환성 정보를 찾을 수 없습니다."
            if pc_results is not None:
                pc_explanation = pc_results.get("explanation", pc_explanation)
            
            # 쿼리 유형에 따라 적절한 응답 생성
            if query_type == "web_search":
                final_answer = web_answer
            elif query_type == "pc_compatibility":
                final_answer = pc_explanation
            elif query_type == "hybrid":
                # 두 결과 통합 (더 강력한 중복 방지)
                integration_prompt = PromptTemplate.from_template("""
                사용자 질문: {question}
                
                웹 검색 결과:
                {web_results}
                
                PC 부품 호환성 분석:
                {pc_results}
                
                위 두 가지 정보를 활용하여 사용자 질문에 종합적인 답변을 제공하세요.
                
                [중요]
                1. 답변은 반드시 중복 없이 한 번만 작성하세요. 
                2. 동일한 내용이나 단락을 두 번 작성하지 마세요.
                3. 답변은 4-5개 문단으로 구성하고, 총 250단어를 넘지 않도록 간결하게 작성하세요.
                4. 답변에 제목이나 헤더를 포함하지 마세요.
                """)
                
                try:
                    # 정확한 응답 포맷을 지정하여 중복 방지
                    final_answer = self.llm.invoke(integration_prompt.format(
                        question=question,
                        web_results=web_answer,
                        pc_results=pc_explanation
                    ))
                    
                    # 중복 검사 및 수정
                    final_answer = self._check_and_fix_duplicates(final_answer)
                except Exception as int_error:
                    logger.error(f"하이브리드 응답 생성 오류: {str(int_error)}")
                    # 단순히 두 답변을 연결하는 백업 방법
                    final_answer = f"웹 검색 결과: {web_answer}\n\nPC 호환성 분석: {pc_explanation}"
            else:
                final_answer = "질문 유형을 확인할 수 없습니다."
            
            state["final_answer"] = final_answer
            logger.info("최종 답변 생성 완료")
            return state
            
        except Exception as e:
            logger.error(f"최종 답변 생성 중 오류 발생: {str(e)}")
            state["errors"].append(f"최종 답변 생성 오류: {str(e)}")
            state["final_answer"] = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요."
            return state
        
    def _check_and_fix_duplicates(self, text):
        """답변 중복 검사 및 수정"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return text
        
        # 중복 단락 제거
        unique_paragraphs = []
        seen_content = set()
        
        for para in paragraphs:
            # 간단한 지문(fingerprint) 생성 - 문단의 처음 50자와 마지막 50자
            if len(para) > 100:
                fingerprint = para[:50] + para[-50:]
            else:
                fingerprint = para
            
            if fingerprint not in seen_content:
                unique_paragraphs.append(para)
                seen_content.add(fingerprint)
        
        # 결과 반환
        return '\n\n'.join(unique_paragraphs)
    
    def run_workflow(self, question: str, chat_history: str) -> Dict[str, Any]:
        """워크플로우 실행 - 상태 초기화 개선"""
        logger.info(f"워크플로우 실행: {question}")
        
        # 초기 상태 설정 - 모든 필수 키를 미리 정의하여 누락 방지
        inputs = {
            "question": question,
            "chat_history": chat_history,
            "query_type": "",
            "web_search_results": None,
            "pc_compatibility_results": None,
            "final_answer": None,
            "errors": [],
            "collected_information": [],
            "search_keywords": [],
            "web_search_queries": []
        }
        
        # 워크플로우 실행
        result = self.graph.invoke(inputs)
        
        # 결과 로깅
        if result["final_answer"]:
            logger.info(f"워크플로우 완료 - 답변 길이: {len(result['final_answer'])}")
        else:
            logger.warning("워크플로우 완료 - 답변 없음")
        
        # 최종 결과 반환
        return {
            "answer": result.get("final_answer", "답변을 생성할 수 없습니다."),
            "query_type": result.get("query_type", "unknown"),
            "collected_information": result.get("collected_information", []),
            "errors": result.get("errors", [])
        }

# 통합 에이전트 초기화 함수
def create_integrated_agent(llm=None):
    """통합 에이전트 생성"""
    if llm is None:
        # 기본 LLM 설정
        llm = OllamaLLM(
            model=config.DEFAULT_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
    
    return IntegratedAgentGraph(llm) 