import duckdb
import pandas as pd
import numpy as np
import logging
import datetime
import math
import re

# 로그 설정
log_filename = f"gpu_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
table_schema = conn.sql("DESCRIBE gpu").fetchall()
column_types = {col[0]: col[1] for col in table_schema}
logging.info(f"GPU 테이블 스키마: {column_types}")

# 테이블 구조 확인
table_info = conn.sql("PRAGMA table_info(gpu)").fetchall()
primary_key = None
for col_info in table_info:
    if col_info[5]:  # 5번 인덱스는 primary key 여부
        primary_key = col_info[1]  # 1번 인덱스는 컬럼명
        break
logging.info(f"GPU 테이블 기본 키: {primary_key}")

# 엑셀 파일 읽기
logging.info("GPU.xlsx 파일 읽는 중...")
df = pd.read_excel("GPU.xlsx")
logging.info(f"엑셀 파일 읽기 완료: {len(df)}개 행 발견")

# 기존 GPU 데이터 삭제
logging.info("기존 GPU 데이터 삭제 중...")
conn.execute("DELETE FROM gpu")
logging.info("기존 GPU 데이터 삭제 완료")

# 시퀀스 초기화 (DuckDB에서 지원하는 경우)
try:
    conn.execute("ALTER SEQUENCE gpu_gpu_id_seq RESTART WITH 1")
    logging.info("GPU ID 시퀀스 초기화 완료")
except:
    logging.warning("GPU ID 시퀀스 초기화 실패 (시퀀스가 없거나 지원되지 않음)")

# 데이터 전처리
logging.info("데이터 전처리 중...")

# NaN 값을 None으로 변환
df = df.replace({np.nan: None})
df = df.replace({float('nan'): None})
df = df.replace({math.nan: None})

# 숫자 추출 함수 (클럭, 용량 등에 사용)
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
        if value.lower() in ['true', 'yes', '1', 'y', '예', '지원', 'o']:
            return True
        if value.lower() in ['false', 'no', '0', 'n', '아니오', '미지원', 'x', '-']:
            return False
    return None

# 컬럼 매핑 정의
column_mapping = {
    '수입/제조사': 'manufacturer',
    '칩셋': 'chipset_manufacturer',
    '분류': 'gpu_type',
    '칩셋모델': 'chipset',
    '기본 클럭': 'core_clock',
    '부스트 클럭': 'memory_clock',
    '메모리용량': 'memory_capacity',
    '종류': 'memory_type',
    '버스': 'memory_bus',
    '쿠다프로세서': 'cuda_cores',
    '스트림프로세서(AMD)': 'stream_processors',
    'RT 코어': 'rt_cores',
    '텐서 코어': 'tensor_cores',
    '장착 인터페이스': 'interface',
    'HDMI': 'hdmi',
    'DP': 'display_port',
    'DVI': 'dvi',
    'D-SUB': 'vga',
    '전원 커넥터': 'power_pin',
    '소비전력': 'power_consumption',
    '권장 파워': 'recommended_psu',
    '냉각 방식': 'cooling_type',
    '팬 개수': 'cooling_fan',
    '길이': 'length',
    '너비': 'width',
    '높이': 'height',
    '백플레이트': 'backplate',
    'LED': 'led',
    'RGB': 'rgb',
    'AURA SYNC': 'aura_sync',
    'MYSTIC LIGHT': 'mystic_light',
    'RGB FUSION': 'rgb_fusion',
    'POLYCHROME SYNC': 'polychrome_sync',
    'TT RGB PLUS': 'tt_rgb_plus',
    'RAZER CHROMA': 'razer_chroma',
    'DirectX': 'directx',
    'OpenGL': 'opengl',
    'OpenCL': 'opencl',
    'Vulkan': 'vulkan',
    'CUDA': 'cuda',
    'PhysX': 'physx',
    'SLI/CrossFire': 'sli_crossfire',
    'VR Ready': 'vr_ready',
    'DLSS': 'dlss',
    'Ray Tracing': 'ray_tracing',
    'HDCP': 'hdcp',
    '멀티 모니터': 'multi_monitor',
    '품명': 'product_name',
    'KC 인증정보': 'kc_certification',
    '정격전압': 'rated_voltage',
    '에너지소비효율등급': 'energy_efficiency',
    '인증': 'certification',
    '동일모델의 출시년월': 'release_date',
    '제조자,수입품의 경우 수입자를 함께 표기': 'manufacturer_importer',
    '원산지': 'origin',
    '제조자': 'manufacturer_info',
    '제조국': 'country_of_origin',
    'A/S 책임자': 'as_manager',
    '고객상담': 'customer_service',
    'A/S 책임자와 전화번호': 'as_contact',
    '크기': 'size',
    '무게': 'weight',
    '주요사항': 'key_features',
    '품질보증기준': 'warranty'
}

try:
    # 테이블의 모든 컬럼 가져오기
    all_columns = [col[0] for col in table_schema]
    
    # 사용 가능한 컬럼 필터링 (primary key 제외)
    available_columns = all_columns.copy()
    if primary_key in available_columns:
        available_columns.remove(primary_key)
    
    # 현재 최대 ID 값 확인
    max_id = conn.sql(f"SELECT COALESCE(MAX({primary_key}), 0) FROM gpu").fetchone()[0]
    logging.info(f"현재 최대 GPU ID: {max_id}")
    
    # 데이터 삽입
    logging.info("GPU 데이터 삽입 중...")
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
                    if db_col in ['core_clock', 'memory_clock'] and isinstance(value, str):
                        value = extract_float(value)
                    elif db_col in ['memory_capacity'] and isinstance(value, str):
                        # 메모리 용량에서 숫자만 추출 (예: "8(GB)" -> 8)
                        match = re.search(r'(\d+)', value)
                        if match:
                            value = int(match.group(1))
                    elif db_col in ['memory_bus'] and isinstance(value, str):
                        # 메모리 버스에서 숫자만 추출 (예: "128(bit)" -> 128)
                        match = re.search(r'(\d+)', value)
                        if match:
                            value = int(match.group(1))
                    elif db_col in ['length', 'width', 'height'] and isinstance(value, str):
                        # 치수 정보 처리 (예: "249.9(mm)" -> 250)
                        value = extract_dimension(value)
                    
                    data[db_col] = value
            
            # 품명을 모델명으로 사용
            if 'product_name' in data and data['product_name'] is not None:
                data['model_name'] = data['product_name']
            # 칩셋 정보로 모델명 생성
            elif 'chipset' in data and data['chipset'] is not None:
                if 'manufacturer' in data and data['manufacturer'] is not None:
                    data['model_name'] = f"{data['manufacturer']} {data['chipset']}"
                else:
                    data['model_name'] = data['chipset']
            
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
            
            # gpu_id 값 설정 (자동 증가 필드가 작동하지 않는 경우)
            max_id += 1
            data['gpu_id'] = max_id
                
            # SQL 쿼리 생성
            columns = ', '.join([f'"{col}"' for col in data.keys()])
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())
            
            query = f"INSERT INTO gpu ({columns}) VALUES ({placeholders})"
            conn.execute(query, values)
            inserted_count += 1
            
            if inserted_count % 10 == 0:
                logging.info(f"{inserted_count}개 행 삽입 완료")
                
        except Exception as row_error:
            logging.error(f"행 #{idx} 삽입 중 오류: {str(row_error)}")
            continue
        
    conn.commit()
    logging.info(f"GPU 데이터 삽입 완료: {inserted_count}개 행 삽입됨")
    
    # 삽입 후 데이터 확인
    final_count = conn.sql("SELECT COUNT(*) FROM gpu").fetchone()[0]
    logging.info(f"최종 GPU 데이터 수: {final_count}")
    
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