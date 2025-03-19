import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")

# Search Func 정의
## 웹 페이지 전체 내용 가져오기 함수
def get_full_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        article = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        if article:
            paragraphs = article.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        else:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        content = ' '.join(content.split())
        
        return content[:8000]
    except Exception as e:
        return f"Error fetching content: {str(e)}"

## 검색 및 내용 강화 함수
def enhanced_search(query):
    # 기본 검색 결과 가져오기
    search_results = search_tool.invoke(query)
    
    # 각 검색 결과에 대해 전체 내용 가져오기
    enhanced_results = []
    for result in search_results[:3]:  # 상위 3개 결과만 처리 (시간 절약)
        try:
            full_content = get_full_content(result['link'])
            # 원래 결과에 전체 내용 추가
            enhanced_result = result.copy()
            enhanced_result['full_content'] = full_content
            enhanced_results.append(enhanced_result)
        except Exception as e:
            print(f"Error processing {result['link']}: {str(e)}")
            enhanced_results.append(result)  # 오류 발생 시 원래 결과 사용
    
    # 나머지 결과는 그대로 추가
    for result in search_results[3:]:
        enhanced_results.append(result)
    
    return enhanced_results