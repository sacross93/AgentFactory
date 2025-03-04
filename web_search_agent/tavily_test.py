from tavily import TavilyClient
import os
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY_JY")

def search_naver_financial_statements():
    # Tavily 클라이언트 초기화
    client = TavilyClient(api_key=tavily_api_key)
    
    # 여러 검색 쿼리 설정
    search_queries = [
        "네이버 (NAVER) ROE ROA 영업이익률 최신 실적",
        "네이버 (NAVER) 분기별 매출액 영업이익 당기순이익",
        "네이버 (NAVER) PER PBR 시가총액",
        "네이버 (NAVER) 재무상태표 손익계산서"
    ]
    
    try:
        all_results = []
        for query in search_queries:
            # 검색 실행
            search_result = client.search(
                query=query,
                search_depth="advanced",
                include_domains=[
                    "fnguide.com", 
                    "finance.naver.com", 
                    "dart.fss.or.kr",
                    "investing.com",
                    "marketscreener.com"
                ],
                max_results=3
            )
            
            if 'results' in search_result:
                all_results.extend(search_result['results'])
        
        # 결과 처리
        print("네이버 재무제표 상세 검색 결과:")
        print("\n=== 주요 재무 지표 및 실적 정보 ===")
        
        # 중복 URL 제거를 위한 집합
        seen_urls = set()
        
        for result in all_results:
            url = result.get('url')
            if url not in seen_urls:
                seen_urls.add(url)
                print("\n제목:", result.get('title'))
                print("URL:", url)
                print("내용 요약:", result.get('content'))
                
                # 스코어가 있는 경우 표시 (관련성 점수)
                if 'score' in result:
                    print("관련성 점수:", result.get('score'))
                print("-" * 70)
        
        return all_results
    
    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    results = search_naver_financial_statements()
