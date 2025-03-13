import duckdb
import pandas as pd
import numpy as np
import logging
import datetime
import math
import re
import os

# 엑셀 파일 로드 (파일 존재 여부 확인)
raw_xlsx_dir = "./cs_agent/raw_xlsx/"

# 파일 패턴으로 가장 최신 파일 찾기 함수
def find_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.xlsx')]
    if not files:
        return None
    return max(files)  # 파일명 기준으로 가장 최신 파일 반환

# 로그 설정
log_filename = f"powersupply_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

power_file = find_latest_file(raw_xlsx_dir, "Power_")

# 콘솔에도 로그 출력
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 데이터베이스 연결
logging.info("데이터베이스 연결 중...")
conn = duckdb.connect('./cs_agent/db/pc_parts.db')
logging.info("데이터베이스 연결 성공")

# 테이블 스키마 확인
table_schema = conn.sql("DESCRIBE power_supply").fetchall()
column_types = {col[0]: col[1] for col in table_schema}
logging.info(f"파워서플라이 테이블 스키마: {column_types}")

# 테이블 구조 확인
table_info = conn.sql("PRAGMA table_info(power_supply)").fetchall()
primary_key = None
for col_info in table_info:
    if col_info[5]:  # 5번 인덱스는 primary key 여부
        primary_key = col_info[1]  # 1번 인덱스는 컬럼명
        break
logging.info(f"파워서플라이 테이블 기본 키: {primary_key}")

# 엑셀 파일 읽기
logging.info("PowerSupply.xlsx 파일 읽는 중...")
df = pd.read_excel(f"{raw_xlsx_dir}{power_file}")
logging.info(f"엑셀 파일 읽기 완료: {len(df)}개 행 발견")

# 기존 파워서플라이 데이터 삭제
logging.info("기존 파워서플라이 데이터 삭제 중...")
conn.execute("DELETE FROM power_supply")
logging.info("기존 파워서플라이 데이터 삭제 완료")

# 시퀀스 초기화 (DuckDB에서 지원하는 경우)
try:
    conn.execute("ALTER SEQUENCE power_supply_psu_id_seq RESTART WITH 1")
    logging.info("파워서플라이 ID 시퀀스 초기화 완료")
except:
    logging.warning("파워서플라이 ID 시퀀스 초기화 실패 (시퀀스가 없거나 지원되지 않음)")

# 데이터 전처리
logging.info("데이터 전처리 중...")

# NaN 값을 None으로 변환
df = df.replace({np.nan: None})
df = df.replace({float('nan'): None})
df = df.replace({math.nan: None})

# 숫자 추출 함수 (와트, 전류 등에 사용)
def extract_number(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        # 숫자 부분만 추출
        match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', value)
        if match:
            return int(float(match.group(1).replace(',', '')))
        return None
    except:
        logging.warning(f"숫자 변환 실패: {value}")
        return None

# 부동 소수점 추출 함수
def extract_float(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        # 숫자 부분만 추출
        match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', value)
        if match:
            return float(match.group(1).replace(',', ''))
        return None
    except:
        logging.warning(f"부동 소수점 변환 실패: {value}")
        return None

# 치수 추출 함수 (mm 단위 포함된 치수 처리)
def extract_dimension(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or np.isnan(value))):
        return value
    if not isinstance(value, str):
        return None
    try:
        # 숫자 부분만 추출 (소수점 포함)
        match = re.search(r'(\d+(?:\.\d+)?)', value)
        if match:
            # 소수점이 있는 경우 반올림하여 정수로 변환
            return round(float(match.group(1)))
        return None
    except:
        logging.warning(f"치수 변환 실패: {value}")
        return None

# 불리언 변환 함수
def convert_to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['y', 'yes', '예', '지원', '있음', 'o', 'true', 't', '1']:
            return True
        elif value in ['n', 'no', '아니오', '미지원', '없음', 'x', 'false', 'f', '0']:
            return False
    return None

# 컬럼 매핑 정의
column_mapping = {
    '수입/제조사': 'manufacturer',
    '제품분류': 'product_category',
    '정격출력': 'wattage',
    'PFC 방식': 'pfc_type',
    '+12V': 'plus12v',
    '+5V': 'plus5v',
    '+3.3V': 'plus3v3',
    '80PLUS': 'efficiency',
    '보증기간': 'warranty',
    '제품명(전체)': 'product_name',
    '모델명': 'model_name',
    '법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항': 'kc_certification',
    '제조국 또는 원산지': 'country_of_origin',
    '제조사/수입품의 경우 수입자를 함께 표기': 'manufacturer_importer',
    'A/S 책임자': 'as_manager',
    '크기': 'size',
    '무게': 'weight',
    '주요사항': 'key_features',
    '품질보증기준': 'certification',
    'A/S 책임자와 전화번호': 'as_contact',
    '모듈러 타입': 'modular_type',
    '6핀 PCI-E': 'pcie_6pin',
    '리폼 케이블': 'reform_cable',
    'LED라이트': 'led_light',
    'AURA SYNC': 'aura_sync',
    'MYSTIC LIGHT': 'mystic_light',
    'RGB FUSION': 'rgb_fusion',
    'POLYCHROME': 'polychrome',
    'RAZER CHROMA': 'razer_chroma',
    'TT RGB PLUS': 'tt_rgb_plus'
}

# 데이터 삽입
try:
    # 사용 가능한 컬럼 확인
    available_columns = set(column_types.keys())
    
    # 현재 최대 ID 값 확인
    max_id = conn.sql("SELECT COALESCE(MAX(psu_id), 0) FROM power_supply").fetchone()[0]
    logging.info(f"현재 최대 psu_id: {max_id}")
    
    # 모델명 중복 방지를 위한 사전
    used_model_names = set()
    
    # 데이터 삽입
    logging.info("파워서플라이 데이터 삽입 중...")
    inserted_count = 0
    
    for idx, row in df.iterrows():
        try:
            data = {}
            
            # 엑셀 데이터를 데이터베이스 컬럼에 매핑
            for excel_col, db_col in column_mapping.items():
                if excel_col in df.columns and db_col in available_columns:
                    value = row[excel_col]
                    
                    # Series 객체를 기본 Python 타입으로 변환
                    if isinstance(value, pd.Series):
                        value = value.iloc[0] if len(value) > 0 else None
                    
                    # NaN 값 처리
                    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
                        value = None
                    
                    # 컬럼 타입에 따른 변환
                    if db_col in column_types:
                        col_type = column_types[db_col].upper()
                        if 'BOOLEAN' in col_type and isinstance(value, str):
                            value = convert_to_bool(value)
                        elif 'FLOAT' in col_type and isinstance(value, str):
                            value = extract_float(value)
                        elif 'INTEGER' in col_type and isinstance(value, str):
                            value = extract_number(value)
                    
                    # 특정 컬럼에 대한 추가 처리
                    if db_col == 'wattage' and isinstance(value, str):
                        # 와트 정보 처리 (예: "750(W)" -> 750)
                        match = re.search(r'(\d+)', value)
                        if match:
                            value = int(match.group(1))
                    elif db_col in ['plus12v', 'plus5v', 'plus3v3'] and isinstance(value, str):
                        # 전류 정보 처리 (예: "62.5(A)" -> 62.5)
                        match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if match:
                            value = float(match.group(1))
                    elif db_col in ['size', 'weight'] and isinstance(value, str):
                        # 치수 정보 처리
                        value = extract_dimension(value)
                    
                    data[db_col] = value
            
            # 품명을 모델명으로 사용
            if 'product_name' in data and data['product_name'] is not None:
                if 'model_name' not in data or data['model_name'] is None or data['model_name'] == '상세정보참조':
                    data['model_name'] = data['product_name']
            
            # 필수 컬럼인 model_name이 없으면 건너뛰기
            if 'model_name' not in data or data['model_name'] is None:
                logging.warning(f"모델명이 없는 행 건너뛰기: 행 #{idx}")
                continue
            
            # 모델명 중복 확인 및 처리
            original_model_name = data['model_name']
            counter = 1
            while data['model_name'] in used_model_names:
                # 모델명에 일련번호 추가
                data['model_name'] = f"{original_model_name} ({counter})"
                counter += 1
            
            # 사용된 모델명 기록
            used_model_names.add(data['model_name'])
            
            # psu_id 값 설정 (자동 증가 필드가 작동하지 않는 경우)
            max_id += 1
            data['psu_id'] = max_id
                
            # SQL 쿼리 생성
            columns = ', '.join([f'"{col}"' for col in data.keys()])
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())
            
            query = f"INSERT INTO power_supply ({columns}) VALUES ({placeholders})"
            conn.execute(query, values)
            inserted_count += 1
            
            if inserted_count % 10 == 0:
                logging.info(f"{inserted_count}개 행 삽입 완료")
                
        except Exception as row_error:
            logging.error(f"행 #{idx} 삽입 중 오류: {str(row_error)}")
            continue
        
    conn.commit()
    logging.info(f"파워서플라이 데이터 삽입 완료: {inserted_count}개 행 삽입됨")
    
    # 삽입 후 데이터 확인
    final_count = conn.sql("SELECT COUNT(*) FROM power_supply").fetchone()[0]
    logging.info(f"최종 파워서플라이 데이터 수: {final_count}")
    
except Exception as e:
    try:
        conn.rollback()
    except:
        logging.error("롤백 실패: 활성화된 트랜잭션이 없습니다.")
    logging.error(f"데이터 삽입 중 오류 발생: {str(e)}")
    raise

# 데이터베이스 연결 종료
conn.close()
logging.info("데이터베이스 연결 종료")