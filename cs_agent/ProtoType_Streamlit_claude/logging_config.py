import logging
import os
from datetime import datetime

def setup_logger():
    """
    전체 시스템에서 사용할 통합 로거를 설정합니다.
    """
    # 로그 디렉토리 생성
    log_dir = "./cs_agent/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 날짜/시간으로 로그 파일명 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/integrated_log_{current_time}.log"
    
    # 기본 로거 설정
    logger = logging.getLogger("IntegratedSystem")
    logger.setLevel(logging.INFO)  # 기본 레벨을 INFO로 설정하여 DEBUG 메시지 감소
    
    # 기존 핸들러 제거 (여러 번 호출 시 중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가 (중요 메시지만 콘솔에 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

# 앱 시작 시 로거 초기화
app_logger = setup_logger()

def get_logger(name=None):
    """
    모듈별 로거를 반환합니다. 모든 로그는 통합 로그 파일에 기록됩니다.
    """
    if name:
        return logging.getLogger(f"IntegratedSystem.{name}")
    return app_logger 