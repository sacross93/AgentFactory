import duckdb
import pandas as pd
import numpy as np
import logging
import datetime
import math
import re

# 로그 설정
log_filename = f"memory_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 콘솔에도 로그 출력
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 데이터베이스 연결
logging.info("데이터베이스 연결 중...")
conn = duckdb.connect('pc_parts.db')
logging.info("데이터베이스 연결 성공")

# 테이블 스키마 확인
table_schema = conn.sql("DESCRIBE memory").fetchall()
column_types = {col[0]: col[1] for col in table_schema}
logging.info(f"메모리 테이블 스키마: {column_types}")

# 테이블 구조 확인
table_info = conn.sql("PRAGMA table_info(memory)").fetchall()
primary_key = None
for col_info in table_info:
    if col_info[5]:  # 5번 인덱스는 primary key 여부
        primary_key = col_info[1]  # 1번 인덱스는 컬럼명
        break
logging.info(f"메모리 테이블 기본 키: {primary_key}")

# 엑셀 파일 읽기
logging.info("Memory.xlsx 파일 읽는 중...")
df = pd.read_excel("Memory.xlsx")
logging.info(f"엑셀 파일 읽기 완료: {len(df)}개 행 발견")

# 기존 메모리 데이터 삭제
logging.info("기존 메모리 데이터 삭제 중...")
conn.execute("DELETE FROM memory")
logging.info("기존 메모리 데이터 삭제 완료")

# 시퀀스 초기화 (DuckDB에서 지원하는 경우)
try:
    conn.execute("ALTER SEQUENCE memory_memory_id_seq RESTART WITH 1")
    logging.info("메모리 ID 시퀀스 초기화 완료")
except:
    logging.warning("메모리 ID 시퀀스 초기화 실패 (시퀀스가 없거나 지원되지 않음)")

# 데이터 전처리
logging.info("데이터 전처리 중...")

# NaN 값을 None으로 변환
df = df.replace({np.nan: None})
df = df.replace({float('nan'): None})
df = df.replace({math.nan: None})

# 숫자 추출 함수 (용량, 클럭 등에 사용)
def extract_number(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        # 숫자 부분만 추출
        if '(' in value:
            return int(value.split('(')[0])
        if ',' in value:
            return int(value.split(',')[0])
        return int(value)
    except:
        logging.warning(f"숫자 변환 실패: {value}")
        return None

# 부동 소수점 추출 함수 (전압 등에 사용)
def extract_float(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        # 숫자 부분만 추출
        if '(' in value:
            return float(value.split('(')[0])
        if 'V' in value:
            return float(re.sub(r'[^\d.]', '', value))
        return float(value)
    except:
        logging.warning(f"부동 소수점 변환 실패: {value}")
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
        if value.lower() in ['true', 'yes', '1', 'y', '예', '지원', 'o']:
            return True
        if value.lower() in ['false', 'no', '0', 'n', '아니오', '미지원', 'x']:
            return False
    return None

# 용량 컬럼 전처리
if '용량' in df.columns:
    df['용량'] = df['용량'].apply(extract_number)

if '메모리 용량' in df.columns:
    df['메모리 용량'] = df['메모리 용량'].apply(extract_number)

# 클럭 컬럼 전처리
if '클럭' in df.columns:
    df['클럭'] = df['클럭'].apply(extract_number)

if '동작 클럭' in df.columns:
    df['동작 클럭'] = df['동작 클럭'].apply(extract_number)

# 전압 컬럼 전처리
if '전압' in df.columns:
    df['전압'] = df['전압'].apply(extract_float)

if '정격전압' in df.columns:
    df['정격전압'] = df['정격전압'].apply(extract_float)

# 불리언 컬럼 전처리
bool_columns = ['ECC', '온다이ECC', 'REG', 'XMP', 'EXPO', '클럭드라이버', '방열판', 'LED', 'RGB제어']
for col in bool_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_bool)

# 컬럼명 매핑 (한글 -> 영어)
column_mapping = {
    '수입/제조사': 'manufacturer',
    '용도': 'purpose',
    '분류': 'classification',
    '제품 분류': 'product_classification',
    '사용 장치': 'device_usage',
    '규격': 'memory_standard',
    '메모리 규격': 'memory_type',
    '용량': 'capacity',
    '메모리 용량': 'memory_capacity',
    '클럭': 'clock',
    '동작 클럭': 'operating_clock',
    '타이밍': 'timing',
    '메모리 타이밍': 'memory_timing',
    '전압': 'voltage',
    '정격전압': 'rated_voltage',
    '패키지': 'package',
    '패키지 구성': 'package_composition',
    'ECC': 'ecc',
    '온다이ECC': 'on_die_ecc',
    'REG': 'reg',
    'XMP': 'xmp',
    'EXPO': 'expo',
    '클럭드라이버': 'clock_driver',
    '방열판': 'heatsink',
    'LED': 'led',
    'LED색': 'led_color',
    'RGB제어': 'rgb_control',
    'AURA SYNC': 'aura_sync',
    'MYSTIC LIGHT': 'mystic_light',
    'POLYCHROME-SYNC': 'polychrome_sync',
    'RGB FUSION': 'rgb_fusion',
    'T-FORCE BLITZ': 'tt_rgb_plus',
    'XPG RGB': 'razer_chroma',
    '품명': 'product_name',
    '모델명': 'model_name',
    'KC 인증정보': 'kc_certification',
    '소비전력': 'power_consumption',
    '에너지소비효율등급': 'energy_efficiency',
    '법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항': 'certification',
    '동일모델의 출시년월': 'release_date',
    '제조자,수입품의 경우 수입자를 함께 표기': 'manufacturer_importer',
    '제조국 또는 원산지': 'origin',
    '제조사/수입품의 경우 수입자를 함께 표기': 'manufacturer_info',
    '제조국': 'country_of_origin',
    '제조회사': 'manufacturer',
    'A/S 책임자': 'as_manager',
    '소비자상담 관련 전화번호': 'customer_service',
    'A/S 책임자와 전화번호': 'as_contact',
    '크기': 'size',
    '무게': 'weight',
    '주요사항': 'key_features',
    '품질보증기준': 'warranty',
    '-': None
}

# 데이터프레임 컬럼 이름 변경
renamed_columns = {}
for col in df.columns:
    if col in column_mapping and column_mapping[col] is not None:
        renamed_columns[col] = column_mapping[col]

df = df.rename(columns=renamed_columns)

# 필요한 컬럼만 선택 (memory 테이블의 컬럼과 일치하는 것들)
memory_columns = conn.sql("select * from memory limit 0").columns
available_columns = [col for col in memory_columns if col in df.columns]

# memory_id 컬럼 제외 (자동 증가 필드)
if 'memory_id' in available_columns:
    available_columns.remove('memory_id')
    
inserted_count = 0

# 데이터 삽입
logging.info("메모리 데이터 삽입 중...")
try:
    # 사용 가능한 컬럼 목록 가져오기
    available_columns = [col[0] for col in table_schema]
    
    # memory_id는 자동 생성되므로 제외
    if 'memory_id' in available_columns:
        available_columns.remove('memory_id')
    
    # 기존 데이터 확인
    existing_count = conn.sql("SELECT COUNT(*) FROM memory").fetchone()[0]
    logging.info(f"기존 메모리 데이터 수: {existing_count}")
    
    # 테이블 생성 쿼리 확인
    create_table_query = conn.sql("SELECT sql FROM sqlite_master WHERE type='table' AND name='memory'").fetchone()
    if create_table_query:
        logging.info(f"메모리 테이블 생성 쿼리: {create_table_query[0]}")
    
    # 현재 최대 ID 값 확인
    max_id = conn.sql("SELECT COALESCE(MAX(memory_id), 0) FROM memory").fetchone()[0]
    logging.info(f"현재 최대 memory_id: {max_id}")
    
    # 모델명 중복 방지를 위한 사전
    used_model_names = set()
    
    for idx, row in df.iterrows():
        try:
            # 필요한 컬럼만 포함하는 딕셔너리 생성
            data = {}
            for col in available_columns:
                if col in df.columns:
                    # Series 객체를 기본 Python 타입으로 변환
                    value = row[col]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0] if len(value) > 0 else None
                    
                    # NaN 값 처리
                    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
                        value = None
                    
                    # 컬럼 타입에 따른 변환
                    if col in column_types:
                        col_type = column_types[col].upper()
                        if 'BOOLEAN' in col_type and isinstance(value, str):
                            value = convert_to_bool(value)
                        elif 'FLOAT' in col_type and isinstance(value, str):
                            value = extract_float(value)
                        elif 'INTEGER' in col_type and isinstance(value, str):
                            value = extract_number(value)
                    
                    data[col] = value
            
            # model_name이 없는 경우 product_name 사용
            if 'model_name' in data and data['model_name'] is None and 'product_name' in data:
                data['model_name'] = data['product_name']
            
            # model_name이 '상세정보참조'인 경우 product_name으로 대체
            if 'model_name' in data and data['model_name'] == '상세정보참조' and 'product_name' in data and data['product_name'] is not None:
                data['model_name'] = data['product_name']
                logging.info(f"행 #{idx}: 모델명 '상세정보참조'를 품명 '{data['product_name']}'으로 대체")
            
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
            
            # memory_id 값 설정 (자동 증가 필드가 작동하지 않는 경우)
            max_id += 1
            data['memory_id'] = max_id
                
            # SQL 쿼리 생성
            columns = ', '.join([f'"{col}"' for col in data.keys()])
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())
            
            query = f"INSERT INTO memory ({columns}) VALUES ({placeholders})"
            conn.execute(query, values)
            inserted_count += 1
            
            if inserted_count % 10 == 0:
                logging.info(f"{inserted_count}개 행 삽입 완료")
                
        except Exception as row_error:
            logging.error(f"행 #{idx} 삽입 중 오류: {str(row_error)}")
            continue
        
    conn.commit()
    logging.info(f"메모리 데이터 삽입 완료: {inserted_count}개 행 삽입됨")
    
    # 삽입 후 데이터 확인
    final_count = conn.sql("SELECT COUNT(*) FROM memory").fetchone()[0]
    logging.info(f"최종 메모리 데이터 수: {final_count}")
    
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