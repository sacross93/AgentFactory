import json
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import time
import pandas as pd
import re
import os

# raw_xlsx 폴더 생성
os.makedirs('./cs_agent/raw_xlsx', exist_ok=True)

# 로그 설정 - 로그 파일을 raw_xlsx 폴더에 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./cs_agent/raw_xlsx/analysis_log_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info(f"로그 파일이 {log_filename}에 저장됩니다.")

def load_categories():
    try:
        with open('valid_categories.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("❌ valid_categories.json 파일을 찾을 수 없습니다.")
        return None

def get_last_page(soup):
    last_page_link = soup.select_one('a.next:contains("마지막")')
    if not last_page_link:
        # "마지막" 텍스트가 포함된 링크 찾기
        last_page_link = soup.find('a', class_='next', string='마지막')
    
    if last_page_link and 'href' in last_page_link.attrs:
        href = last_page_link['href']
        # 예: javascript:pageMove(30); 에서 30 추출
        if match := re.search(r'pageMove\((\d+)\)', href):
            return int(match.group(1))
    
    # 마지막 페이지 링크를 찾지 못한 경우 페이지네이션에서 가장 큰 숫자 찾기
    page_links = soup.select('div.paginate a')
    max_page = 1
    for link in page_links:
        if 'href' in link.attrs and 'pageMove' in link['href']:
            if match := re.search(r'pageMove\((\d+)\)', link['href']):
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)
    
    return max_page

def check_category_products(session, category_info):
    api_url = "https://www.jchyunplace.co.kr/skin/shop/basic/product_list_include_plist.php"
    page = 1
    product_details = []
    max_retries = 3  # 최대 재시도 횟수
    
    logging.info(f"=== {category_info['name']} 분석 시작 ===")
    
    # 첫 페이지 요청으로 마지막 페이지 확인
    data = {
        **category_info['params'],
        "search_cate": "0",
        "search": "",
        "search1": "",
        "sprice": "",
        "eprice": "",
        "page": "1",
        "list_sort_type": "",
        "view_type": "list"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    # 첫 페이지 요청 (재시도 로직 추가)
    for attempt in range(max_retries):
        try:
            response = session.post(api_url, headers=headers, data=data, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            last_page = get_last_page(soup)
            logging.info(f"총 페이지 수: {last_page}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                retry_wait = 5 * (attempt + 1)  # 점진적으로 대기 시간 증가
                logging.warning(f"첫 페이지 요청 실패 ({attempt+1}/{max_retries}): {str(e)}. {retry_wait}초 후 재시도...")
                time.sleep(retry_wait)
            else:
                logging.error(f"첫 페이지 요청 최종 실패: {str(e)}")
                return None
    
    while page <= last_page:
        data['page'] = str(page)
        logging.info(f"페이지 {page}/{last_page} 처리 중...")
        
        # 페이지 요청 (재시도 로직 추가)
        for attempt in range(max_retries):
            try:
                response = session.post(api_url, headers=headers, data=data, timeout=30)
                if response.status_code != 200:
                    logging.error(f"❌ 페이지 로드 실패 (status: {response.status_code})")
                    raise Exception(f"HTTP 오류: {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                products = soup.select('.prd_view_type li.list')
                
                for idx, product in enumerate(products, 1):
                    name_elem = product.select_one('a.name')
                    if name_elem and 'href' in name_elem.attrs:
                        product_url = name_elem['href']
                        
                        # 로그 출력 최소화 - 50개마다 한 번씩만 출력
                        if len(product_details) % 50 == 0 and len(product_details) > 0:
                            logging.info(f"현재까지 {len(product_details)}개 제품 처리 완료")
                        
                        # 제품 정보 추출 (재시도 로직은 extract_product_info 내부에 있음)
                        product_info = extract_product_info(session, product_url)
                        if product_info:
                            product_details.append(product_info)
                        
                        time.sleep(1)
                
                # 페이지 처리 성공
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    retry_wait = 5 * (attempt + 1)
                    logging.warning(f"페이지 {page} 처리 실패 ({attempt+1}/{max_retries}): {str(e)}. {retry_wait}초 후 재시도...")
                    time.sleep(retry_wait)
                else:
                    logging.error(f"페이지 {page} 처리 최종 실패: {str(e)}")
                    # 마지막 시도에서도 실패했지만, 다음 페이지로 계속 진행
        
        page += 1
        time.sleep(2)  # 페이지 간 대기 시간 증가
    
    # DataFrame 생성
    if product_details:
        df = pd.DataFrame(product_details)
        logging.info(f"✅ {category_info['name']} 카테고리의 {len(df)}개 제품 정보를 수집했습니다.")
        return df
    return None

def extract_product_info(session, product_url):
    base_url = "https://www.jchyunplace.co.kr"
    full_url = base_url + product_url if not product_url.startswith('http') else product_url
    max_retries = 3  # 최대 재시도 횟수
    
    for attempt in range(max_retries):
        try:
            response = session.get(full_url, timeout=30)
            
            if response.status_code != 200:
                logging.error(f"제품 페이지 로드 실패: {full_url}")
                raise Exception(f"HTTP 오류: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            info_dict = {}
            
            # 제품명 추출 (span.name에서 가져오기)
            product_name_elem = soup.select_one('span.name')
            if product_name_elem:
                info_dict['제품명(전체)'] = product_name_elem.get_text(strip=True)
            
            # 상세 정보 테이블 찾기
            table = soup.select_one('.more_info.info table')
            
            if table:
                for row in table.find_all('tr'):
                    th = row.find('th')
                    if th:
                        continue
                    
                    cells = row.find_all('td')
                    for i in range(0, len(cells), 2):
                        if i + 1 < len(cells):
                            key = cells[i].get_text(strip=True)
                            value = cells[i+1].get_text(strip=True)
                            if key and value:
                                info_dict[key] = value
            else:
                # 테이블을 찾지 못했지만 오류로 처리하지 않고 빈 정보 반환
                logging.warning("제품 상세 테이블을 찾을 수 없습니다.")
            
            return info_dict
        
        except Exception as e:
            if attempt < max_retries - 1:
                retry_wait = 5 * (attempt + 1)
                logging.warning(f"제품 정보 추출 실패 ({attempt+1}/{max_retries}): {str(e)}. {retry_wait}초 후 재시도...")
                time.sleep(retry_wait)
            else:
                logging.error(f"제품 정보 추출 최종 실패: {str(e)}")
                return {}  # 빈 딕셔너리 반환하여 프로세스 계속 진행
    
    return {}  # 모든 시도 실패 시 빈 딕셔너리 반환

def analyze_all_categories():
    categories = load_categories()
    if not categories:
        logging.error("유효한 카테고리가 없습니다.")
        print("유효한 카테고리가 없습니다. valid_categories.json 파일을 확인하세요.")
        return {}
    
    print(f"처리할 카테고리 목록: {list(categories.keys())}")
    logging.info(f"처리할 카테고리 목록: {list(categories.keys())}")
    
    session = requests.Session()
    category_dataframes = {}
    
    # raw_xlsx 폴더는 이미 파일 상단에서 생성됨
    
    try:
        for category_key, category in categories.items():
            logging.info("="*50)
            logging.info(f"카테고리 '{category['name']}' 처리 시작")
            print(f"\n카테고리 '{category['name']}' 처리 시작...")
            
            try:
                df = check_category_products(session, category)
                if df is not None and not df.empty:
                    category_dataframes[category['name']] = df
                    
                    # 각 카테고리별로 Excel 파일 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_filename = f"./cs_agent/raw_xlsx/{category['name']}_{timestamp}.xlsx"
                    df.to_excel(excel_filename, index=False)
                    logging.info(f"✅ '{category['name']}' 데이터를 {excel_filename}에 저장했습니다.")
                    print(f"✅ '{category['name']}' 데이터를 {excel_filename}에 저장했습니다.")
                    
                    # 각 카테고리 처리 후 현재 상태 출력
                    print(f"\n현재까지 처리된 카테고리: {list(category_dataframes.keys())}")
                    print(f"마지막으로 처리된 카테고리 '{category['name']}'의 데이터 샘플 (처음 5개 행):")
                    print(df.head())
                    
                    # 주요 정보 요약 출력
                    print(f"총 {len(df)}개 제품, 컬럼: {list(df.columns)[:5]}...")
                else:
                    print(f"카테고리 '{category['name']}'에서 데이터를 찾지 못했습니다.")
                    logging.warning(f"카테고리 '{category['name']}'에서 데이터를 찾지 못했습니다.")
            except Exception as e:
                logging.error(f"카테고리 {category['name']} 처리 중 오류 발생: {str(e)}")
                print(f"오류 발생: {str(e)}")
            
            logging.info(f"카테고리 '{category['name']}' 처리 완료")
            logging.info("="*50)
    except KeyboardInterrupt:
        logging.info("키보드 인터럽트 감지! 현재까지 수집된 데이터를 반환합니다.")
        print("\n키보드 인터럽트로 중단되었습니다.")
        print(f"현재까지 처리된 카테고리: {list(category_dataframes.keys())}")
    
    return category_dataframes

if __name__ == "__main__":
    try:
        category_dfs = analyze_all_categories()
        
        if category_dfs:
            logging.info("=== 최종 결과 ===")
            
            # 모든 카테고리 데이터를 하나의 Excel 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_data_filename = f"./cs_agent/raw_xlsx/all_categories_{timestamp}.xlsx"
            
            with pd.ExcelWriter(all_data_filename) as writer:
                for category_name, df in category_dfs.items():
                    logging.info(f"{category_name}: {len(df)}개 제품")
                    # 각 카테고리를 별도의 시트로 저장
                    df.to_excel(writer, sheet_name=category_name[:31], index=False)  # 시트 이름 최대 31자
                    
                    # 디버깅을 위해 데이터프레임 내용 출력
                    print(f"\n{category_name} 데이터프레임:")
                    print(df.head())
            
            logging.info(f"✅ 모든 카테고리 데이터를 {all_data_filename}에 저장했습니다.")
            print(f"✅ 모든 카테고리 데이터를 {all_data_filename}에 저장했습니다.")
    except KeyboardInterrupt:
        logging.info("프로그램이 사용자에 의해 중단되었습니다.")