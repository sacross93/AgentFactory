import requests
from bs4 import BeautifulSoup
import logging
import time
import json
from datetime import datetime

# 로깅 설정
log_filename = f'crawling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 콘솔에도 출력
    ]
)

# PC 관련 키워드 체크 함수
def is_pc_related(category_name):
    pc_keywords = ['PC', '컴퓨터', '그래픽카드', 'CPU', '메모리', 'SSD', 'HDD', '마우스', '키보드', '파워', '케이스']
    return any(keyword.lower() in category_name.lower() for keyword in pc_keywords)

def check_category_exists(session, cate1, cate2):
    api_url = "https://www.jchyunplace.co.kr/skin/shop/basic/product_list_include_plist.php"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    data = {
        "depth": "2",
        "cate1": str(cate1),
        "cate2": str(cate2),
        "search_cate": "0",
        "search": "",
        "search1": "",
        "sprice": "",
        "eprice": "",
        "page": "1",
        "list_sort_type": "",
        "view_type": "list"
    }
    
    try:
        print(f"카테고리 확인 중: cate1={cate1}, cate2={cate2}")  # 진행상황 출력
        response = session.post(api_url, headers=headers, data=data)
        if response.status_code == 200:
            # 응답에 제품이 있는지 확인
            soup = BeautifulSoup(response.text, 'html.parser')
            products = soup.select('.prd_view_type li.list')
            if products:
                # 첫 번째 제품의 이름을 가져와서 카테고리 이름으로 사용
                first_product = soup.select_one('.prd_view_type li.list a.name')
                category_name = first_product.get_text(strip=True) if first_product else f"Category {cate1}_{cate2}"
                print(f"✅ 유효한 카테고리 발견! cate1={cate1}, cate2={cate2}, 이름={category_name}, 제품 수={len(products)}")
                logging.info(f"유효한 카테고리 발견: cate1={cate1}, cate2={cate2}, 제품 수={len(products)}")
                return True, category_name
            else:
                print(f"❌ 제품 없음: cate1={cate1}, cate2={cate2}")
    except Exception as e:
        print(f"⚠️ 오류 발생: cate1={cate1}, cate2={cate2}, 오류={str(e)}")
        logging.error(f"카테고리 체크 중 오류 발생 (cate1={cate1}, cate2={cate2}): {str(e)}")
    
    return False, None

def load_saved_categories():
    try:
        with open('valid_categories.json', 'r', encoding='utf-8') as f:
            categories = json.load(f)
            print(f"저장된 카테고리 {len(categories)}개를 불러왔습니다.")
            return categories
    except FileNotFoundError:
        print("저장된 카테고리가 없습니다. 새로 검색을 시작합니다.")
        return None

def save_categories(categories):
    with open('valid_categories.json', 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"카테고리 정보를 valid_categories.json에 저장했습니다.")

def find_all_categories():
    # 저장된 카테고리가 있는지 먼저 확인
    saved_categories = load_saved_categories()
    if saved_categories:
        return saved_categories

    session = requests.Session()
    valid_categories = {}
    total_checked = 0
    
    print("카테고리 검색 시작...")
    # cate1은 2부터 시작, cate2는 1-100 범위로 제한
    for cate1 in range(2, 11):
        print(f"\n=== cate1={cate1} 검사 시작 ===")
        for cate2 in range(1, 101):
            exists, category_name = check_category_exists(session, cate1, cate2)
            total_checked += 1
            
            if exists:
                category_key = f"{cate1}_{cate2}"
                valid_categories[category_key] = {
                    'name': category_name,
                    'params': {
                        'depth': '2',
                        'cate1': str(cate1),
                        'cate2': str(cate2)
                    },
                    'products': []
                }
                
                # 진행 상황 요약 출력
                print(f"✅ 카테고리 발견! 총 {len(valid_categories)}개 (현재: {category_name})")
                # 발견할 때마다 저장
                save_categories(valid_categories)
            
            time.sleep(1)
    
    print(f"\n카테고리 검색 완료! 총 {len(valid_categories)}개의 유효한 카테고리 발견")
    save_categories(valid_categories)
    return valid_categories

def extract_product_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    product_links = []
    
    # 제품 리스트 찾기
    products = soup.select('.prd_view_type li.list')
    
    for product in products:
        # 제품 상세 링크 찾기 (name 클래스를 가진 a 태그)
        link_element = product.select_one('a.name')
        if link_element and 'href' in link_element.attrs:
            product_url = link_element['href']
            # URL에서 pd_no 파라미터만 추출
            if 'pd_no=' in product_url:
                product_links.append(product_url)
                logging.info(f"제품 링크 발견: {product_url}")
    
    return product_links

def crawl_category_products(category_info):
    base_url = "https://www.jchyunplace.co.kr"
    api_url = "https://www.jchyunplace.co.kr/skin/shop/basic/product_list_include_plist.php"
    session = requests.Session()
    all_product_links = []
    page = 1
    
    # POST 요청에 사용할 헤더
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{base_url}{category_info['url']}"
    }
    
    while True:
        # 기본 파라미터에 페이지 번호 추가
        data = {
            "search_cate": "0",
            "search": "",
            "search1": "",
            "sprice": "",
            "eprice": "",
            "page": str(page),
            "list_sort_type": "",
            "view_type": "list",
            **category_info['params']  # 카테고리별 파라미터 추가
        }
        
        logging.info(f"{category_info['name']} - Page {page} 크롤링중")
        
        try:
            response = session.post(api_url, headers=headers, data=data)
            if response.status_code != 200:
                logging.warning(f"{category_info['name']}의 {page}페이지 로드 실패 (status: {response.status_code})")
                break
            
            # 디버깅을 위한 응답 저장
            debug_filename = f"debug_{category_info['name'].replace('/', '_')}_page{page}.html"
            with open(debug_filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logging.info(f"HTML 내용이 {debug_filename}에 저장되었습니다.")
            
            # 페이지 렌더링을 위한 대기
            # time.sleep(2)
            
            # 제품 링크 추출
            product_links = extract_product_links(response.text)
            
            # 더 이상 제품이 없으면 종료
            if not product_links:
                break
                
            all_product_links.extend(product_links)
            page += 1
            
        except Exception as e:
            logging.error(f"크롤링 중 오류 발생: {str(e)}")
            break
    
    return all_product_links

if __name__ == "__main__":
    print("크롤링 시작...")
    print(f"로그 파일: {log_filename}")
    
    # 모든 유효한 카테고리 찾기
    categories = find_all_categories()
    print(f"\n총 {len(categories)}개의 유효한 카테고리를 찾았습니다.")
    
    # 각 카테고리의 제품 수집
    pc_categories = 0
    for category_key, category in categories.items():
        if is_pc_related(category['name']):
            pc_categories += 1
            print(f"\n[{pc_categories}] {category['name']} 크롤링 중...")
            product_links = crawl_category_products(category)
            category['products'] = product_links
            print(f"✅ {category['name']}에서 {len(product_links)}개의 제품 링크를 찾았습니다.")
    
    # 결과를 JSON 파일로 저장
    result_filename = f'crawling_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    
    print(f"\n크롤링 완료!")
    print(f"결과 파일: {result_filename}")
    
    # 최종 통계 출력
    total_products = sum(len(cat['products']) for cat in categories.values() if is_pc_related(cat['name']))
    print(f"총 {total_products}개의 PC 관련 제품을 찾았습니다.")
