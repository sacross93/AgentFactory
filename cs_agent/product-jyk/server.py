import sys
import os
import socket

# 올바른 경로 설정
sys.path.append('/home/wlsdud022/AgentFactory')
# 상대 경로도 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import logging
import time
import asyncio
from datetime import datetime

# 직접적인 import 경로로 변경
from ProtoType_JYK.orchestrator_graph_jyk import orchestrator_graph

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api_server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("api_server")

# FastAPI 앱 초기화
app = FastAPI(
    title="제플몰 AI 상담 API",
    description="PC 부품 호환성 및 게임 요구사항을 확인하는 AI 상담 API",
    version="1.0.0"
)

# CORS 설정 - 프로덕션 환경용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.110.101", "https://192.168.110.101", "http://localhost:8000", "https://ai-chatbot.co.kr"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Origin", "Accept", "X-Requested-With"],
    max_age=3600  # preflight 요청 캐시 시간 (1시간)
)

# 진행중인 요청 저장소
active_requests = {}

# 요청 모델 정의
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    timeout: Optional[int] = 60  # 초 단위 타임아웃 (기본 60초)
    chat_history: Optional[List[Dict[str, str]]] = [] # 대화 기록 추가

# 상태 응답 모델
class StatusResponse(BaseModel):
    request_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    
# 결과 응답 모델
class QueryResponse(BaseModel):
    request_id: str
    answer: str
    processing_time: float
    status: str = "complete"
    additional_info: Optional[Dict[str, Any]] = None

# 요청 정보 저장 모델
class RequestInfo:
    def __init__(self, query: str, session_id: Optional[str] = None, chat_history: Optional[List] = None):
        self.start_time = time.time()
        self.query = query
        self.session_id = session_id
        self.chat_history = chat_history or []
        self.status = "pending"
        self.result = None
        self.progress = 0.0
        self.message = "요청이 처리 대기중입니다."
        self.request_id = f"{int(time.time())}-{session_id or 'anonymous'}"

# 비동기 처리 함수
async def process_query(request_info: RequestInfo, timeout: int):
    request_info.status = "processing"
    request_info.message = "요청 처리중..."
    
    try:
        # orchestrator_graph 함수 호출 (대화 기록 전달)
        loop = asyncio.get_event_loop()
        request_info.result = await loop.run_in_executor(
            None, 
            lambda: orchestrator_graph(
                request_info.query, 
                chat_history=request_info.chat_history
            )
        )
        
        request_info.status = "complete"
        processing_time = time.time() - request_info.start_time
        logger.info(f"요청 처리 완료: {request_info.request_id} (소요 시간: {processing_time:.2f}초)")
        
    except Exception as e:
        request_info.status = "error"
        request_info.message = f"처리 중 오류 발생: {str(e)}"
        request_info.result = f"죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다: {str(e)}"
        logger.error(f"요청 처리 실패: {request_info.request_id} - {str(e)}")
    
    # 타임아웃 후 요청 정보 삭제 스케줄링
    await asyncio.sleep(300)  # 5분 후 삭제
    if request_info.request_id in active_requests:
        del active_requests[request_info.request_id]

# 쿼리 처리 엔드포인트
@app.post("/query", response_model=StatusResponse)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    logger.info(f"쿼리 요청 받음: {request.query} (세션 ID: {request.session_id})")
    logger.debug(f"대화 기록: {request.chat_history}")
    
    # 요청 정보 생성 및 저장 (대화 기록 포함)
    request_info = RequestInfo(
        query=request.query, 
        session_id=request.session_id,
        chat_history=request.chat_history
    )
    active_requests[request_info.request_id] = request_info
    
    # 백그라운드에서 처리 시작
    background_tasks.add_task(process_query, request_info, request.timeout)
    
    return StatusResponse(
        request_id=request_info.request_id,
        status="accepted",
        message="요청이 접수되었습니다. 상태를 확인하려면 /status/{request_id} 엔드포인트를 사용하세요."
    )

# 상태 확인 엔드포인트
@app.get("/status/{request_id}", response_model=StatusResponse)
async def check_status(request_id: str):
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="요청 ID를 찾을 수 없습니다.")
    
    request_info = active_requests[request_id]
    
    return StatusResponse(
        request_id=request_id,
        status=request_info.status,
        progress=request_info.progress,
        message=request_info.message
    )

# 결과 가져오기 엔드포인트
@app.get("/result/{request_id}", response_model=QueryResponse)
async def get_result(request_id: str):
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="요청 ID를 찾을 수 없습니다.")
    
    request_info = active_requests[request_id]
    
    if request_info.status != "complete":
        if request_info.status == "error":
            raise HTTPException(status_code=500, detail=request_info.message)
        else:
            raise HTTPException(status_code=202, detail="처리가 아직 완료되지 않았습니다.")
    
    processing_time = time.time() - request_info.start_time
    
    return QueryResponse(
        request_id=request_id,
        answer=request_info.result,
        processing_time=processing_time,
        additional_info={
            "query": request_info.query,
            "timestamp": datetime.now().isoformat()
        }
    )

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    logger.info(f"헬스 체크 요청 받음 - {datetime.now().isoformat()}")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 서버 실행 함수
def start_server(host="0.0.0.0", port=8000, reload=False):
    # 서버 IP 주소 가져오기
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n서버 시작 중...")
    print(f"호스트 이름: {hostname}")
    print(f"로컬 IP: {local_ip}")
    print(f"접속 주소: http://{local_ip}:{port}")
    print(f"헬스 체크 URL: http://{local_ip}:{port}/health\n")
    
    uvicorn.run("server:app", host=host, port=port, reload=reload)

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000)