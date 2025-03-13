# utils.py
"""유틸리티 함수: 캐싱, 입력 검증, 로깅 등."""

import functools
import logging
from typing import List
import bleach

# 로깅 설정
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def cache_search_results(func):
    """검색 결과 캐싱 데코레이터."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(query, *args, **kwargs):
        if query in cache:
            logging.info(f"Cache hit for query: {query}")
            return cache[query]
        result = func(query, *args, **kwargs)
        cache[query] = result
        logging.info(f"Cache miss for query: {query}, stored result")
        return result
    return wrapper

def sanitize_input(user_input: str) -> str:
    """사용자 입력 검증 및 정화."""
    sanitized = bleach.clean(user_input, tags=[], strip=True)
    if not sanitized.strip():
        raise ValueError("입력값이 비어 있거나 유효하지 않습니다.")
    if len(sanitized) > 500:
        raise ValueError("입력값이 너무 깁니다. 500자 이내로 작성해주세요.")
    logging.info(f"Sanitized input: {sanitized}")
    return sanitized

def log_error(error: Exception, context: str):
    """에러 로깅."""
    logging.error(f"Error in {context}: {str(error)}")
    
def translate_to_korean(text):
    """텍스트를 한글로 번역"""
    response = llm.invoke(translation_prompt.format(text=text))
    return response

def check_korean_response(response):
    """응답이 한글인지 확인"""
    has_chinese = any(0x4E00 <= ord(char) <= 0x9FFF for char in response)
    has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in response)
    return not (has_chinese or not has_korean)