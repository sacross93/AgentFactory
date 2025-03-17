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

# 세분화된 질문 분류 프롬프트
query_classification_prompt = PromptTemplate.from_template("""
당신은 PC 및 게임 관련 질문을 분류하는 AI입니다. 다음 카테고리로 질문을 분류하세요:

1. "user_pc_game_check" - 사용자의 PC 사양으로 특정 게임/프로그램을 실행할 수 있는지 묻는 질문
   예: "내 컴퓨터(CPU: 5600x, GPU: RTX 3080)로 배틀그라운드 돌아갈까?"

2. "game_pc_recommendation" - 특정 게임을 위한 PC 구성 추천을 요청하는 질문
   예: "배틀그라운드를 위한 PC 구성 추천해줘"

3. "program_requirements" - 프로그램이나 게임의 권장 사양만 묻는 질문
   예: "배틀그라운드 권장 사양이 뭐야?"

4. "part_compatibility" - 특정 부품과 호환되는 다른 부품을 찾는 질문
   예: "5600x와 호환되는 메인보드 알려줘"

5. "general_pc_info" - 일반적인 PC 관련 정보 요청
   예: "CPU 성능 순위가 어떻게 돼?"

사용자 질문: {question}
대화 기록: {chat_history}

분석 내용을 포함하여 어떤 카테고리에 속하는지 설명한 후, "category: [카테고리명]" 형식으로 마무리하세요.
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
        """개선된 에이전트 그래프 생성"""
        # 그래프 초기화
        workflow = StateGraph(IntegratedAgentState)
        
        # 노드 추가
        workflow.add_node("classify_query", self.classify_query_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("pc_compatibility", self.pc_compatibility_node)
        workflow.add_node("user_pc_game_check", self.user_pc_game_check_node)
        workflow.add_node("game_pc_recommendation", self.game_pc_recommendation_node)
        workflow.add_node("generate_final_answer", self.generate_final_answer)
        
        # 엣지 (흐름) 정의
        workflow.set_entry_point("classify_query")
        
        # 분류 결과에 따른 처리 경로 설정
        workflow.add_conditional_edges(
            "classify_query",
            lambda state: state["query_type"],
            {
                "program_requirements": "web_search",  # 프로그램 요구사항 -> 웹 검색
                "part_compatibility": "pc_compatibility",  # 부품 호환성 -> DB 검색
                "user_pc_game_check": "user_pc_game_check",  # 사용자 PC 게임 확인 -> 하이브리드
                "game_pc_recommendation": "game_pc_recommendation",  # 게임 PC 추천 -> 하이브리드
                "general_pc_info": "web_search",  # 일반 PC 정보 -> 웹 검색
                "web_search": "web_search"  # 직접 웹 검색 키에 대한 처리 추가
            }
        )
        
        # 나머지 노드들은 최종 답변 생성으로 이동
        workflow.add_edge("web_search", "generate_final_answer")
        workflow.add_edge("pc_compatibility", "generate_final_answer")
        workflow.add_edge("user_pc_game_check", "generate_final_answer")
        workflow.add_edge("game_pc_recommendation", "generate_final_answer")
        
        # 최종 답변 후 종료
        workflow.add_edge("generate_final_answer", END)
        
        return workflow.compile()
    
    def classify_query_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """질문 유형 분류 노드"""
        logger.info(f"질문 분류 시작: {state['question']}")
        
        # 로그 추가
        state["collected_information"].append(f"🔍 질문 분석: '{state['question']}'")
        
        try:
            response = self.llm.invoke(query_classification_prompt.format(
                question=state["question"],
                chat_history=state["chat_history"]
            ))
            
            # 카테고리 추출
            category_match = re.search(r'category:\s*(\w+)', response, re.IGNORECASE)
            
            if category_match:
                query_type = category_match.group(1).strip().lower()
                
                # 허용된 카테고리 목록 (그래프에 정의된 노드와 일치해야 함)
                allowed_categories = [
                    "program_requirements", 
                    "part_compatibility",
                    "user_pc_game_check", 
                    "game_pc_recommendation",
                    "general_pc_info"
                ]
                
                # 유효한 카테고리인지 확인
                if query_type not in allowed_categories:
                    # 기본값으로 'program_requirements' 사용 (이는 web_search로 매핑됨)
                    logger.warning(f"유효하지 않은 분류({query_type}), 'program_requirements'로 기본 설정")
                    query_type = "program_requirements"
            else:
                # 분류 실패 시 기본값 설정
                logger.warning("질문 분류 실패, 'program_requirements'로 기본 설정")
                query_type = "program_requirements"  # 'web_search'가 아닌 매핑된 카테고리 사용
            
            # JSON 출력 시도
            try:
                result_json = {
                    "query_type": query_type,
                    "analysis": response[:200] + "..."  # 분석 요약
                }
                state["collected_information"].append(f"🧠 질문 유형 분류: {query_type}")
                state["collected_information"].append(f"📋 분석: {response[:100]}...")
            except Exception as e:
                logger.warning(f"JSON 추출 실패, 기본 유형({query_type}) 사용")
                result_json = {"query_type": query_type}
            
            # 상태 업데이트
            state["query_type"] = query_type
            
            return state
        except Exception as e:
            error_msg = str(e)
            logger.error(f"질문 분류 오류: {error_msg}")
            state["errors"].append(f"질문 분류 오류: {error_msg}")
            
            # 오류 발생 시 기본값 설정 (안전하게 처리)
            state["query_type"] = "program_requirements"  # 이는 web_search 노드로 매핑됨
            return state
    
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
    
    def user_pc_game_check_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """사용자 PC로 게임 실행 가능 여부 확인 (시나리오 1)"""
        logger.info(f"사용자 PC 게임 호환성 확인: {state['question']}")
        
        try:
            # 1. 사용자 PC 사양 추출
            timestamp = time.strftime('%H:%M:%S')
            state["collected_information"].append(f"🖥️ 사용자 PC 사양 추출 중... ({timestamp})")
            
            user_pc_parts = self._extract_user_pc_parts(state["question"])
            state["user_pc_parts"] = user_pc_parts
            
            for part_type, part_name in user_pc_parts.items():
                state["collected_information"].append(f"🔍 감지된 부품: {part_type} - {part_name}")
            
            # 2. 게임/프로그램 이름 추출
            program_name = self._extract_program_name(state["question"])
            state["program_name"] = program_name
            state["collected_information"].append(f"🎮 분석 대상 프로그램: {program_name}")
            
            # 3. 웹 검색으로 프로그램 권장사양 확인
            search_query = f"{program_name} 권장사양 요구사항 시스템 사양"
            state["collected_information"].append(f"🌐 검색 쿼리: {search_query}")
            
            web_result = self.agent_system.run_workflow(search_query, state["chat_history"])
            state["web_search_results"] = web_result
            state["collected_information"].append("✅ 프로그램 권장사양 정보 수집 완료")
            
            # 4. PC 호환성 및 성능 확인 (사용자 PC 사양과 게임 요구사항 비교)
            pc_check_input = {
                "question": f"{program_name}에 {', '.join([f'{k}: {v}' for k, v in user_pc_parts.items()])} 사양이 적합한지 분석해줘",
                "search_keywords": user_pc_parts.values(),
                "program_requirements": web_result.get("final_answer", "")
            }
            
            pc_result = self._run_pc_compatibility_check(pc_check_input)
            state["pc_compatibility_results"] = pc_result
            state["collected_information"].append("✅ PC 호환성 분석 완료")
            
            return state
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"사용자 PC 게임 확인 오류: {error_msg}")
            state["errors"].append(f"사용자 PC 게임 확인 오류: {error_msg}")
            return state
    
    def game_pc_recommendation_node(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """게임을 위한 PC 구성 추천 (시나리오 2)"""
        logger.info(f"게임용 PC 구성 추천: {state['question']}")
        
        try:
            # 1. 게임/프로그램 이름 추출
            program_name = self._extract_program_name(state["question"])
            state["program_name"] = program_name
            state["collected_information"].append(f"🎮 분석 대상 프로그램: {program_name}")
            
            # 2. 웹 검색으로 프로그램 권장사양 확인
            search_query = f"{program_name} 권장사양 요구사항 시스템 사양"
            state["collected_information"].append(f"🌐 검색 쿼리: {search_query}")
            
            web_result = self.agent_system.run_workflow(search_query, state["chat_history"])
            state["web_search_results"] = web_result
            state["collected_information"].append("✅ 프로그램 권장사양 정보 수집 완료")
            
            # 중요: 검색 키워드 정의 (게임 사양 기반)
            # 기본 키워드 지정
            state["search_keywords"] = ["RTX 3060", "RTX 3070", "i5-12400F", "Ryzen 5 5600X"]
            state["part_types"] = ["cpu", "gpu", "motherboard", "memory"]
            
            # 3. 권장사양 기반으로 PC 구성 추천 요청 (수정된 부분)
            pc_recommendation_input = {
                "question": f"{program_name}를 위한 PC 구성을 추천해줘",
                "program_requirements": web_result.get("final_answer", ""),
                "search_keywords": state["search_keywords"],  # 중요: 키워드 전달
                "part_types": state["part_types"],  # 중요: 부품 유형 전달
                "query_type": "game_pc_recommendation"  # 쿼리 타입 명시
            }
            
            pc_result = self._run_pc_compatibility_check(pc_recommendation_input)
            state["pc_compatibility_results"] = pc_result
            state["collected_information"].append("✅ PC 구성 추천 완료")
            
            return state
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"게임용 PC 추천 오류: {error_msg}")
            state["errors"].append(f"게임용 PC 추천 오류: {error_msg}")
            return state
    
    def generate_final_answer(self, state: IntegratedAgentState) -> IntegratedAgentState:
        """각 시나리오에 맞는 최종 답변 생성"""
        logger.info(f"최종 답변 생성 시작: {state['query_type']}")
        query_type = state["query_type"]
        
        try:
            if query_type == "program_requirements":
                # 프로그램 권장 사양 정보만 제공
                web_results = state.get("web_search_results", {})
                answer = web_results.get("final_answer", "정보를 찾을 수 없습니다.")
                
            elif query_type == "part_compatibility":
                # 부품 호환성 정보 제공
                pc_results = state.get("pc_compatibility_results", {})
                answer = pc_results.get("explanation", "호환성 정보를 찾을 수 없습니다.")
                
            elif query_type == "user_pc_game_check":
                # 사용자 PC로 게임 실행 가능 여부
                template = PromptTemplate.from_template("""
                사용자의 PC 사양과 게임 요구 사항을 분석해 답변을 생성하세요.
                
                사용자 PC 사양:
                {user_pc_parts}
                
                게임 정보:
                프로그램: {program_name}
                
                게임 권장 사양:
                {program_requirements}
                
                PC 호환성 분석:
                {pc_compatibility}
                
                위 정보를 종합하여, 사용자의 PC로 해당 게임을 실행할 수 있는지, 어느 정도의 성능을 
                기대할 수 있는지 자세히 설명하세요. 가능하면 그래픽 설정 추천, 예상 FPS 등의 정보도 포함하세요.
                """)
                
                pc_parts_str = ""
                for k, v in state.get("user_pc_parts", {}).items():
                    pc_parts_str += f"- {k}: {v}\n"
                    
                web_results = state.get("web_search_results", {})
                pc_results = state.get("pc_compatibility_results", {})
                
                chain = template | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "user_pc_parts": pc_parts_str,
                    "program_name": state.get("program_name", "알 수 없는 프로그램"),
                    "program_requirements": web_results.get("final_answer", "요구 사항 정보 없음"),
                    "pc_compatibility": pc_results.get("explanation", "호환성 분석 정보 없음")
                })
                
            elif query_type == "game_pc_recommendation":
                # 게임을 위한 PC 구성 추천
                template = PromptTemplate.from_template("""
                게임의 요구 사항과 호환성 분석 결과를 바탕으로 최적의 PC 구성을 추천하세요.
                
                게임 정보:
                프로그램: {program_name}
                
                게임 권장 사양:
                {program_requirements}
                
                PC 구성 추천 분석:
                {pc_recommendation}
                
                다음 정보를 포함하여 답변하세요:
                1. 권장 CPU, GPU, RAM, 메인보드, 저장장치, 파워 서플라이
                2. 예산별 구성 (가능하면 저가, 중가, 고가 옵션)
                3. 구성 선택 이유 및 해당 게임에서의 예상 성능
                """)
                
                web_results = state.get("web_search_results", {})
                pc_results = state.get("pc_compatibility_results", {})
                
                chain = template | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "program_name": state.get("program_name", "알 수 없는 프로그램"),
                    "program_requirements": web_results.get("final_answer", "요구 사항 정보 없음"),
                    "pc_recommendation": pc_results.get("explanation", "PC 구성 추천 정보 없음")
                })
                
            elif query_type == "general_pc_info":
                # 일반 PC 정보 제공
                web_results = state.get("web_search_results", {})
                answer = web_results.get("final_answer", "정보를 찾을 수 없습니다.")
                
            else:
                # 기타 질문
                answer = "질문 유형을 확인할 수 없습니다. 다른 방식으로 질문해 주세요."
            
            # 중복 제거
            answer = self._remove_duplicates(answer)
            
            # 결과 저장
            state["final_answer"] = answer
            return state
            
        except Exception as e:
            logger.error(f"최종 답변 생성 오류: {str(e)}")
            state["errors"].append(f"최종 답변 생성 오류: {str(e)}")
            state["final_answer"] = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요."
            return state
        
    def _remove_duplicates(self, text):
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

    def _extract_user_pc_parts(self, question: str) -> Dict[str, str]:
        """사용자 질문에서 PC 부품 정보 추출"""
        # 템플릿 기반 추출
        template = PromptTemplate.from_template("""
        다음 질문에서 PC 부품 정보를 추출하고 JSON 형식으로 반환하세요:
        
        질문: {question}
        
        예시 출력:
        ```json
        {
          "cpu": "5600x",
          "gpu": "RTX 3080",
          "ram": "16GB",
          "storage": "1TB SSD"
        }
        ```
        
        질문에 언급되지 않은 부품은 포함하지 마세요.
        JSON만 출력하세요.
        """)
        
        # LLM 호출하여 부품 추출
        chain = template | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        # JSON 파싱
        try:
            # JSON 부분만 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result
            
            parts = json.loads(json_str)
            return parts
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 오류: {result}")
            return {}

    def _extract_program_name(self, question: str) -> str:
        """사용자 질문에서 프로그램/게임 이름 추출"""
        template = PromptTemplate.from_template("""
        다음 질문에서 언급된 게임이나 프로그램 이름을 추출하세요:
        
        질문: {question}
        
        게임/프로그램 이름만 정확히 답변하세요. 만약 명확한 이름이 언급되지 않았다면 '알 수 없음'이라고 답변하세요.
        """)
        
        chain = template | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        if result.lower() in ['알 수 없음', '모름', 'unknown']:
            return "알 수 없는 프로그램"
        
        return result.strip()

    def _run_pc_compatibility_check(self, input_data: dict) -> dict:
        """PC 호환성 모듈 실행"""
        from pc_check_agent import process_pc_compatibility_query
        
        try:
            # 로그 추가
            logger.info(f"PC 호환성 모듈 호출: {input_data['question']}")
            
            # 필수 필드 확인
            if "search_keywords" not in input_data or not input_data["search_keywords"]:
                logger.warning("검색 키워드가 없습니다. 기본값 사용")
                input_data["search_keywords"] = ["RTX 3060", "i5-12400F"] 
            
            # collected_information 필드 확인 및 초기화
            if "collected_information" not in input_data:
                input_data["collected_information"] = []
            
            # 안전하게 PC 호환성 모듈 호출
            result = process_pc_compatibility_query(
                input_data["question"],
                input_state=input_data
            )
            
            # 쿼리 로그가 있다면 수집된 정보에 추가 (고급 로깅)
            if "detailed_query_logs" in result:
                input_data["collected_information"].append("📊 SQL 실행 결과:")
                for log in result["detailed_query_logs"]:
                    input_data["collected_information"].append(f"  {log}")
            
            # 쿼리 요약 정보 추가
            if "query_summary" in result:
                input_data["collected_information"].append("📈 쿼리 요약:")
                for summary in result["query_summary"]:
                    input_data["collected_information"].append(f"  {summary}")
            
            return result
        except Exception as e:
            logger.error(f"PC 호환성 모듈 오류: {str(e)}")
            return {
                "explanation": f"PC 호환성 분석 중 오류가 발생했습니다: {str(e)}",
                "errors": [str(e)]
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