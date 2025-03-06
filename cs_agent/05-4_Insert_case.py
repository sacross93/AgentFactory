import duckdb
import pandas as pd
import numpy as np
import logging
import datetime
import math
import re

# 로그 설정
log_filename = f"case_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
table_schema = conn.sql("DESCRIBE case_chassis").fetchall()
column_types = {col[0]: col[1] for col in table_schema}
logging.info(f"케이스 테이블 스키마: {column_types}")

# 테이블 구조 확인
table_info = conn.sql("PRAGMA table_info(case_chassis)").fetchall()
primary_key = None
for col_info in table_info:
    if col_info[5]:  # 5번 인덱스는 primary key 여부
        primary_key = col_info[1]  # 1번 인덱스는 컬럼명
        break
logging.info(f"케이스 테이블 기본 키: {primary_key}")

# 엑셀 파일 읽기
logging.info("Case.xlsx 파일 읽는 중...")
df = pd.read_excel("Case.xlsx")
logging.info(f"엑셀 파일 읽기 완료: {len(df)}개 행 발견")

# 기존 케이스 데이터 삭제
logging.info("기존 케이스 데이터 삭제 중...")
conn.execute("DELETE FROM case_chassis")
logging.info("기존 케이스 데이터 삭제 완료")

# 시퀀스 초기화 (DuckDB에서 지원하는 경우)
try:
    conn.execute("ALTER SEQUENCE case_chassis_case_id_seq RESTART WITH 1")
    logging.info("케이스 ID 시퀀스 초기화 완료")
except:
    logging.warning("케이스 ID 시퀀스 초기화 실패 (시퀀스가 없거나 지원되지 않음)")

# 데이터 전처리
logging.info("데이터 전처리 중...")

# NaN 값을 None으로 변환
df = df.replace({np.nan: None})
df = df.replace({float('nan'): None})
df = df.replace({math.nan: None})

# 컬럼 매핑 정의
column_mapping = {
    '수입/제조사': 'manufacturer',
    '제품 분류': 'case_type',
    '케이스 타입': 'case_type',
    '지원 파워': 'power_supply_type',
    'ATX': 'atx_support',
    'mATX': 'matx_support',
    'MiniITX': 'itx_support',
    'CPU쿨러장착높이': 'cpu_cooler_height',
    'VGA장착길이': 'vga_length',
    '2.5베이': 'ssd_bays',
    '3.5베이': 'hdd_bays',
    'PCI슬롯': 'expansion_slots',
    'USB': 'usb_ports',
    'USB 3.0': 'usb_31_gen1',
    '너비': 'width',
    '높이': 'height',
    '깊이': 'depth',
    '장착 팬 개수': 'fans_included',
    '측면': 'side_panel',
    '품명': 'product_name',
    '파워 포함 여부': 'power_included',
    '브랜드별 지원파워규격': 'supported_mb_types',
    'EATX': 'eatx_support',
    '수랭쿨러 지원': 'radiator_support',
    '전면 팬': 'front_fan',
    '상단 팬': 'top_fan',
    '후면 팬': 'rear_fan',
    '하단 팬': 'bottom_fan',
    '측면 팬': 'side_fan',
    '전면 라디에이터': 'front_radiator',
    '상단 라디에이터': 'top_radiator',
    '후면 라디에이터': 'rear_radiator',
    '측면 라디에이터': 'side_radiator',
    'USB 3.1 Gen1': 'usb_31_gen1',
    'USB 3.1 Gen2': 'usb_31_gen2',
    'USB 3.1 Type-C': 'usb_31_type_c',
    'USB 2.0': 'usb_20',
    '오디오 포트': 'audio_ports',
    'RGB 지원': 'rgb_support',
    'RGB 컨트롤러': 'rgb_controller',
    'AURA SYNC': 'aura_sync',
    'Mystic Light': 'mystic_light',
    'RGB Fusion': 'rgb_fusion',
    'Polychrome Sync': 'polychrome_sync',
    'TT RGB Plus': 'tt_rgb_plus',
    'Razer Chroma': 'razer_chroma',
    '무게': 'weight',
    '재질': 'material',
    '먼지 필터': 'dust_filter',
    'PSU 슈라우드': 'psu_shroud',
    'KC인증': 'kc_certification',
    '정격전압': 'rated_voltage',
    '소비전력': 'power_consumption',
    '에너지효율': 'energy_efficiency',
    '인증': 'certification',
    '출시일': 'release_date',
    '제조/수입자': 'manufacturer_importer',
    '원산지': 'origin',
    '제조자정보': 'manufacturer_info',
    '제조국': 'country_of_origin',
    'A/S책임자': 'as_manager',
    '고객상담': 'customer_service',
    'A/S연락처': 'as_contact',
    '크기': 'size',
    '주요특징': 'key_features',
    '보증기간': 'warranty'
}

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
            return int(re.search(r'(\d+)', value.split('(')[0]).group(1))
        if ',' in value:
            return int(re.search(r'(\d+)', value.split(',')[0]).group(1))
        match = re.search(r'(\d+)', value)
        if match:
            return int(match.group(1))
        return None
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
            return float(re.search(r'([\d.]+)', value.split('(')[0]).group(1))
        match = re.search(r'([\d.]+)', value)
        if match:
            return float(match.group(1))
        return None
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
        if value.lower() in ['true', 'yes', '1', 'y', '예', '지원', 'o', '있음']:
            return True
        if value.lower() in ['false', 'no', '0', 'n', '아니오', '미지원', 'x', '없음']:
            return False
    return None

# 치수 추출 함수 (너비, 높이, 깊이 등에 사용)
def extract_dimension(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value
    try:
        # 숫자 부분만 추출
        match = re.search(r'([\d.]+)', value)
        if match:
            return float(match.group(1))
        return None
    except:
        logging.warning(f"치수 변환 실패: {value}")
        return None

try:
    # 테이블 컬럼 가져오기
    table_columns = [col[0] for col in table_schema]
    
    # 필수 컬럼 확인
    required_columns = ['model_name', 'manufacturer']
    
    # case_id는 자동 증가 필드이므로 제외
    available_columns = [col for col in table_columns if col != 'case_id']
    
    # 현재 최대 ID 값 확인
    max_id = conn.sql("SELECT COALESCE(MAX(case_id), 0) FROM case_chassis").fetchone()[0]
    logging.info(f"현재 최대 case_id: {max_id}")
    
    # 데이터 삽입
    logging.info("케이스 데이터 삽입 중...")
    inserted_count = 0
    
    # 모델명 중복 방지를 위한 사전
    used_model_names = set()
    
    for idx, row in df.iterrows():
        try:
            # 필요한 컬럼만 포함하는 딕셔너리 생성
            data = {}
            
            # 엑셀 컬럼을 DB 컬럼으로 매핑
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
                    if db_col in ['width', 'height', 'depth'] and isinstance(value, str):
                        value = extract_dimension(value)
                    
                    data[db_col] = value
            
            # 품명을 모델명으로 사용
            if 'product_name' in data and 'model_name' not in data:
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
            
            # case_id 값 설정 (자동 증가 필드가 작동하지 않는 경우)
            max_id += 1
            data['case_id'] = max_id
                
            # SQL 쿼리 생성
            columns = ', '.join([f'"{col}"' for col in data.keys()])
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())
            
            query = f"INSERT INTO case_chassis ({columns}) VALUES ({placeholders})"
            conn.execute(query, values)
            inserted_count += 1
            
            if inserted_count % 10 == 0:
                logging.info(f"{inserted_count}개 행 삽입 완료")
                
        except Exception as row_error:
            logging.error(f"행 #{idx} 삽입 중 오류: {str(row_error)}")
            continue
        
    conn.commit()
    logging.info(f"케이스 데이터 삽입 완료: {inserted_count}개 행 삽입됨")
    
    # 삽입 후 데이터 확인
    final_count = conn.sql("SELECT COUNT(*) FROM case_chassis").fetchone()[0]
    logging.info(f"최종 케이스 데이터 수: {final_count}")
    
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