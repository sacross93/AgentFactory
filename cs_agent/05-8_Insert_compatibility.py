import duckdb
import pandas as pd
import logging
from datetime import datetime
import os
import numpy as np
import re

# 로깅 설정
log_filename = f'compatibility_update_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 데이터베이스 연결
conn = duckdb.connect('./cs_agent/db/pc_parts.db')

# 마지막 업데이트 시간 저장 파일
LAST_UPDATE_FILE = 'last_compatibility_update.txt'

def get_last_update_time():
    """마지막 업데이트 시간 가져오기"""
    if os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, 'r') as f:
            try:
                return datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S')
            except:
                return None
    return None

def save_last_update_time():
    """현재 시간을 마지막 업데이트 시간으로 저장"""
    with open(LAST_UPDATE_FILE, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def get_table_data(table_name):
    """테이블 데이터를 가져오는 함수"""
    try:
        # 테이블 이름이 SQL 예약어인 경우 따옴표로 감싸기
        query = f'SELECT * FROM "{table_name}"'
        conn.execute(query)
        return conn.fetch_df()
    except Exception as e:
        logging.error(f"{table_name} 테이블 데이터 가져오기 실패: {e}")
        return pd.DataFrame()

def check_if_update_needed(compatibility_table, source_tables):
    """호환성 테이블 업데이트가 필요한지 확인"""
    last_update = get_last_update_time()
    
    # 호환성 테이블에 데이터가 없는지 확인
    try:
        conn.execute(f'SELECT COUNT(*) FROM "{compatibility_table}"')
        count = conn.fetchone()[0]
        if count == 0:
            logging.info(f"{compatibility_table} 테이블이 비어 있습니다.")
            return True
    except Exception as e:
        logging.error(f"{compatibility_table} 테이블 확인 중 오류: {e}")
        return True
    
    if not last_update:
        return False  # 이미 데이터가 있고 마지막 업데이트 시간이 없으면 업데이트 불필요
    
    # 소스 테이블 중 하나라도 마지막 업데이트 이후 변경되었는지 확인
    for table in source_tables:
        try:
            conn.execute(f'SELECT MAX(updated_at) FROM "{table}"')
            result = conn.fetchone()
            if result and result[0] and result[0] > last_update:
                logging.info(f"{table} 테이블이 마지막 호환성 업데이트 이후 변경되었습니다.")
                return True
        except:
            pass  # updated_at 컬럼이 없을 수 있음
    
    logging.info(f"{compatibility_table} 테이블은 이미 최신 상태입니다.")
    return False

# 필요한 테이블 데이터 가져오기
cases = get_table_data("case_chassis")
cpus = get_table_data("cpu")
coolers = get_table_data("cpu_cooler")
gpus = get_table_data("gpu")
motherboards = get_table_data("motherboard")
memories = get_table_data("memory")
psus = get_table_data("power_supply")
storages = get_table_data("storage")

# 테이블 스키마 확인 함수
def get_column_names(df, expected_columns):
    """데이터프레임에서 예상 컬럼과 일치하는 실제 컬럼 이름 찾기"""
    actual_columns = {}
    for expected in expected_columns:
        # 정확히 일치하는 컬럼 찾기
        if expected in df.columns:
            actual_columns[expected] = expected
        else:
            # 유사한 컬럼 찾기 (대소문자 무시, 언더스코어 무시 등)
            for col in df.columns:
                if expected.lower().replace('_', '') == col.lower().replace('_', ''):
                    actual_columns[expected] = col
                    break
            
            # 찾지 못한 경우
            if expected not in actual_columns:
                actual_columns[expected] = None
    
    return actual_columns

# 데이터 전처리 및 정규화 함수들
def normalize_socket_type(socket_str):
    """소켓 타입 문자열을 정규화하여 일관성 있게 만듭니다."""
    if socket_str is None or pd.isna(socket_str):
        return ""
    
    socket_str = str(socket_str).strip().upper()
    
    # 공백 제거 및 일관성 있는 형식으로 변환
    socket_str = socket_str.replace(" ", "")
    
    # LGA 소켓 정규화
    if "LGA" in socket_str:
        # LGA1700, LGA 1700 등을 LGA1700으로 통일
        socket_str = re.sub(r'LGA\s*(\d+)', r'LGA\1', socket_str)
    
    # AM4, AM 4 등을 AM4로 통일
    if "AM" in socket_str:
        socket_str = re.sub(r'AM\s*(\d+)', r'AM\1', socket_str)
    
    return socket_str

def normalize_form_factor(form_str):
    """폼팩터 문자열을 정규화합니다."""
    if form_str is None or pd.isna(form_str):
        return ""
    
    form_str = str(form_str).strip().upper()
    
    # 공백 및 하이픈 처리
    normalized = form_str.replace(" ", "").replace("-", "")
    
    # 일반적인 폼팩터 정규화
    mapping = {
        "MATX": "MICROATX",
        "MICRO-ATX": "MICROATX",
        "MICRO ATX": "MICROATX",
        "M-ATX": "MICROATX",
        "ITX": "MINIITX",
        "MINI-ITX": "MINIITX",
        "MINI ITX": "MINIITX",
        "EATX": "EXTENDEDATX",
        "E-ATX": "EXTENDEDATX",
        "EXTENDED ATX": "EXTENDEDATX",
        "EXTENDED-ATX": "EXTENDEDATX"
    }
    
    for key, value in mapping.items():
        if key in normalized:
            return value
    
    # ATX는 그대로 유지
    if "ATX" in normalized and not any(k in normalized for k in mapping.keys()):
        return "ATX"
    
    return normalized

def convert_dimension_to_mm(dimension_str):
    """차원 문자열(높이, 길이 등)을 밀리미터(mm) 단위로 변환합니다."""
    if dimension_str is None or pd.isna(dimension_str):
        return None
    
    try:
        # 숫자만 추출
        dimension_str = str(dimension_str).strip().lower()
        num_match = re.search(r'(\d+(\.\d+)?)', dimension_str)
        
        if not num_match:
            return None
        
        value = float(num_match.group(1))
        
        # 단위 확인 및 변환
        if 'cm' in dimension_str:
            return value * 10  # cm -> mm
        elif 'in' in dimension_str or 'inch' in dimension_str:
            return value * 25.4  # inch -> mm
        elif 'mm' in dimension_str:
            return value  # 이미 mm
        else:
            # 단위가 명시되지 않은 경우 기본적으로 mm로 가정
            return value
    except (ValueError, TypeError):
        logging.warning(f"차원 변환 실패: {dimension_str}")
        return None

def normalize_power_rating(power_str):
    """전력 문자열을 와트(W) 단위로 변환합니다."""
    if power_str is None or pd.isna(power_str):
        return None
    
    try:
        # 숫자만 추출
        power_str = str(power_str).strip().lower()
        num_match = re.search(r'(\d+(\.\d+)?)', power_str)
        
        if not num_match:
            return None
        
        value = float(num_match.group(1))
        
        # 단위 확인
        if 'kw' in power_str:
            return value * 1000  # kW -> W
        else:
            # 단위가 없거나 W인 경우
            return value
    except (ValueError, TypeError):
        logging.warning(f"전력 변환 실패: {power_str}")
        return None

# 호환성 판단 함수 개선
def is_compatible_socket(cpu_socket, mb_socket):
    """CPU 소켓과 메인보드 소켓의 호환성을 판단합니다."""
    if not cpu_socket or not mb_socket:
        return False, "소켓 정보 부족"
    
    cpu_socket = normalize_socket_type(cpu_socket)
    mb_socket = normalize_socket_type(mb_socket)
    
    if cpu_socket == mb_socket:
        return True, "소켓 일치"
    
    # 소켓 패밀리 호환성 (예: LGA1200과 LGA1155 등)
    socket_compatibility = {
        "LGA1700": ["LGA1700"],
        "LGA1200": ["LGA1200", "LGA1151", "LGA1150"],
        "LGA1151": ["LGA1151", "LGA1150", "LGA1155"],
        "LGA1150": ["LGA1150", "LGA1155", "LGA1156"],
        "LGA1155": ["LGA1155", "LGA1156"],
        "LGA2066": ["LGA2066"],
        "AM4": ["AM4"],
        "AM5": ["AM5"]
    }
    
    if cpu_socket in socket_compatibility and mb_socket in socket_compatibility.get(cpu_socket, []):
        return True, "소켓 패밀리 호환"
    
    return False, "소켓 불일치"

# 1. CPU와 메인보드 호환성 (소켓 타입 기준)
def update_cpu_mb_compatibility():
    if not check_if_update_needed("cpu_mb_compatibility", ["cpu", "motherboard"]):
        return
        
    if cpus.empty or motherboards.empty:
        logging.warning("CPU 또는 메인보드 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    cpu_cols = get_column_names(cpus, ['cpu_id', 'socket_type'])
    mb_cols = get_column_names(motherboards, ['mb_id', 'socket_type'])
    
    # 메인보드 ID 컬럼 이름이 다를 수 있음
    if mb_cols['mb_id'] is None:
        for possible_id in ['motherboard_id', 'id', 'mainboard_id']:
            if possible_id in motherboards.columns:
                mb_cols['mb_id'] = possible_id
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in cpu_cols.values() or None in mb_cols.values():
        logging.warning(f"CPU 또는 메인보드 테이블에 필요한 컬럼이 없습니다. CPU 컬럼: {cpu_cols}, MB 컬럼: {mb_cols}")
        return
    
    compatibility_data = []
    
    for cpu_id, cpu_socket in zip(cpus[cpu_cols['cpu_id']], cpus[cpu_cols['socket_type']]):
        for mb_id, mb_socket in zip(motherboards[mb_cols['mb_id']], motherboards[mb_cols['socket_type']]):
            is_compatible, reason = is_compatible_socket(cpu_socket, mb_socket)
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'cpu_id': cpu_id,
                'mb_id': mb_id,
                'compatible': 1 if is_compatible else 0,
                'reason': reason  # 호환성 이유 저장
            })
    
    if compatibility_data:
        # 테이블 스키마 확인 및 필요시 ALTER TABLE
        columns = conn.execute("PRAGMA table_info(cpu_mb_compatibility)").fetchall()
        column_names = [col[1] for col in columns]
        
        # reason 컬럼이 없으면 추가
        if 'reason' not in column_names:
            conn.execute("ALTER TABLE cpu_mb_compatibility ADD COLUMN reason TEXT")
            logging.info("cpu_mb_compatibility 테이블에 reason 컬럼 추가 완료")
        
        df = pd.DataFrame(compatibility_data)
        
        # 기존 데이터 삭제 대신 증분 업데이트 (예시)
        conn.execute("BEGIN TRANSACTION")
        try:
            # 기존 데이터 삭제
            conn.execute("DELETE FROM cpu_mb_compatibility")
            # 새 데이터 삽입
            conn.execute("INSERT INTO cpu_mb_compatibility SELECT * FROM df")
            conn.execute("COMMIT")
            logging.info(f"CPU-메인보드 호환성 테이블 업데이트 완료: {len(df)}개 항목")
        except Exception as e:
            conn.execute("ROLLBACK")
            logging.error(f"CPU-메인보드 호환성 업데이트 오류: {e}")
    else:
        logging.warning("CPU-메인보드 호환성 데이터가 없습니다.")

# 2. CPU와 쿨러 호환성 (소켓 지원 기준)
def update_cpu_cooler_compatibility():
    if not check_if_update_needed("cpu_cooler_compatibility", ["cpu", "cpu_cooler"]):
        return
        
    if cpus.empty or coolers.empty:
        logging.warning("CPU 또는 쿨러 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    cpu_cols = get_column_names(cpus, ['cpu_id', 'socket_type'])
    cooler_cols = get_column_names(coolers, ['cooler_id', 'socket_support'])
    
    # 쿨러 ID 컬럼 이름이 다를 수 있음
    if cooler_cols['cooler_id'] is None:
        for possible_id in ['cpu_cooler_id', 'id']:
            if possible_id in coolers.columns:
                cooler_cols['cooler_id'] = possible_id
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in cpu_cols.values() or None in cooler_cols.values():
        logging.warning(f"CPU 또는 쿨러 테이블에 필요한 컬럼이 없습니다. CPU 컬럼: {cpu_cols}, 쿨러 컬럼: {cooler_cols}")
        return
    
    compatibility_data = []
    
    for cpu_id, cpu_socket in zip(cpus[cpu_cols['cpu_id']], cpus[cpu_cols['socket_type']]):
        for cooler_id, socket_support in zip(coolers[cooler_cols['cooler_id']], coolers[cooler_cols['socket_support']]):
            # 쿨러의 socket_support에 CPU 소켓이 포함되어 있으면 호환됨
            compatible = 0
            if socket_support is not None and isinstance(socket_support, str):
                if cpu_socket in socket_support:
                    compatible = 1
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'cpu_id': cpu_id,
                'cooler_id': cooler_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM cpu_cooler_compatibility")
        conn.execute("INSERT INTO cpu_cooler_compatibility SELECT * FROM df")
        logging.info(f"CPU-쿨러 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("CPU-쿨러 호환성 데이터가 없습니다.")

# 3. 쿨러와 케이스 호환성 (쿨러 높이와 케이스 지원 높이 기준)
def update_cooler_case_compatibility():
    if not check_if_update_needed("cooler_case_compatibility", ["cpu_cooler", "case_chassis"]):
        return
        
    if coolers.empty or cases.empty:
        logging.warning("쿨러 또는 케이스 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    cooler_cols = get_column_names(coolers, ['cooler_id', 'height'])
    case_cols = get_column_names(cases, ['case_id', 'cpu_cooler_height'])
    
    # ID 컬럼 이름이 다를 수 있음
    if cooler_cols['cooler_id'] is None:
        for possible_id in ['cpu_cooler_id', 'id']:
            if possible_id in coolers.columns:
                cooler_cols['cooler_id'] = possible_id
                break
                
    if case_cols['case_id'] is None:
        for possible_id in ['case_chassis_id', 'id', 'chassis_id']:
            if possible_id in cases.columns:
                case_cols['case_id'] = possible_id
                break
    
    # CPU 쿨러 높이 컬럼이 다를 수 있음
    if case_cols['cpu_cooler_height'] is None:
        for possible_col in ['cooler_height', 'max_cooler_height', 'max_cpu_cooler_height']:
            if possible_col in cases.columns:
                case_cols['cpu_cooler_height'] = possible_col
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in cooler_cols.values() or None in case_cols.values():
        logging.warning(f"쿨러 또는 케이스 테이블에 필요한 컬럼이 없습니다. 쿨러 컬럼: {cooler_cols}, 케이스 컬럼: {case_cols}")
        return
    
    compatibility_data = []
    
    for cooler_id, cooler_height in zip(coolers[cooler_cols['cooler_id']], coolers[cooler_cols['height']]):
        for case_id, case_cooler_height in zip(cases[case_cols['case_id']], cases[case_cols['cpu_cooler_height']]):
            compatible = 0
            
            # 둘 다 숫자 값이 있는 경우에만 비교
            if cooler_height is not None and case_cooler_height is not None:
                try:
                    if float(cooler_height) <= float(case_cooler_height):
                        compatible = 1
                except (ValueError, TypeError):
                    pass
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'cooler_id': cooler_id,
                'case_id': case_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM cooler_case_compatibility")
        conn.execute("INSERT INTO cooler_case_compatibility SELECT * FROM df")
        logging.info(f"쿨러-케이스 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("쿨러-케이스 호환성 데이터가 없습니다.")

# 4. 메인보드와 케이스 호환성 (폼팩터 기준)
def update_mb_case_compatibility():
    if not check_if_update_needed("mb_case_compatibility", ["motherboard", "case_chassis"]):
        return
        
    if motherboards.empty or cases.empty:
        logging.warning("메인보드 또는 케이스 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    mb_cols = get_column_names(motherboards, ['mb_id', 'form_factor'])
    case_cols = get_column_names(cases, ['case_id', 'atx_support', 'matx_support', 'itx_support', 'eatx_support'])
    
    # ID 컬럼 이름이 다를 수 있음
    if mb_cols['mb_id'] is None:
        for possible_id in ['motherboard_id', 'id', 'mainboard_id']:
            if possible_id in motherboards.columns:
                mb_cols['mb_id'] = possible_id
                break
                
    if case_cols['case_id'] is None:
        for possible_id in ['case_chassis_id', 'id', 'chassis_id']:
            if possible_id in cases.columns:
                case_cols['case_id'] = possible_id
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    missing_cols = [col for col, name in mb_cols.items() if name is None] + [col for col, name in case_cols.items() if name is None]
    if missing_cols:
        logging.warning(f"메인보드 또는 케이스 테이블에 필요한 컬럼이 없습니다: {missing_cols}")
        return
    
    compatibility_data = []
    
    for mb_id, mb_form_factor in zip(motherboards[mb_cols['mb_id']], motherboards[mb_cols['form_factor']]):
        for case_id, atx_support, matx_support, itx_support, eatx_support in zip(
            cases[case_cols['case_id']], 
            cases[case_cols['atx_support']], 
            cases[case_cols['matx_support']], 
            cases[case_cols['itx_support']], 
            cases[case_cols['eatx_support']]):
            
            compatible = 0
            
            # NA 값을 False로 처리
            atx_support = False if pd.isna(atx_support) else bool(atx_support)
            matx_support = False if pd.isna(matx_support) else bool(matx_support)
            itx_support = False if pd.isna(itx_support) else bool(itx_support)
            eatx_support = False if pd.isna(eatx_support) else bool(eatx_support)
            
            # 폼팩터에 따른 호환성 확인
            if mb_form_factor == 'ATX' and atx_support:
                compatible = 1
            elif mb_form_factor == 'm-ATX' and matx_support:
                compatible = 1
            elif mb_form_factor == 'ITX' and itx_support:
                compatible = 1
            elif mb_form_factor == 'E-ATX' and eatx_support:
                compatible = 1
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'mb_id': mb_id,
                'case_id': case_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM mb_case_compatibility")
        conn.execute("INSERT INTO mb_case_compatibility SELECT * FROM df")
        logging.info(f"메인보드-케이스 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("메인보드-케이스 호환성 데이터가 없습니다.")

# 5. 메인보드와 메모리 호환성 (메모리 타입 기준)
def update_mb_memory_compatibility():
    if not check_if_update_needed("mb_memory_compatibility", ["motherboard", "memory"]):
        return
        
    if motherboards.empty or memories.empty:
        logging.warning("메인보드 또는 메모리 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    mb_cols = get_column_names(motherboards, ['mb_id', 'memory_support'])
    memory_cols = get_column_names(memories, ['memory_id', 'memory_type'])
    
    # ID 컬럼 이름이 다를 수 있음
    if mb_cols['mb_id'] is None:
        for possible_id in ['motherboard_id', 'id', 'mainboard_id']:
            if possible_id in motherboards.columns:
                mb_cols['mb_id'] = possible_id
                break
                
    if memory_cols['memory_id'] is None:
        for possible_id in ['id', 'ram_id']:
            if possible_id in memories.columns:
                memory_cols['memory_id'] = possible_id
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in mb_cols.values() or None in memory_cols.values():
        logging.warning(f"메인보드 또는 메모리 테이블에 필요한 컬럼이 없습니다. MB 컬럼: {mb_cols}, 메모리 컬럼: {memory_cols}")
        return
    
    compatibility_data = []
    
    for mb_id, mb_memory_support in zip(motherboards[mb_cols['mb_id']], motherboards[mb_cols['memory_support']]):
        for memory_id, memory_type in zip(memories[memory_cols['memory_id']], memories[memory_cols['memory_type']]):
            compatible = 0
            
            # 메모리 타입 호환성 확인
            if mb_memory_support == 'DDR4' and memory_type == 'DDR4':
                compatible = 1
            elif mb_memory_support == 'DDR5' and memory_type == 'DDR5':
                compatible = 1
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'mb_id': mb_id,
                'memory_id': memory_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM mb_memory_compatibility")
        conn.execute("INSERT INTO mb_memory_compatibility SELECT * FROM df")
        logging.info(f"메인보드-메모리 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("메인보드-메모리 호환성 데이터가 없습니다.")

# 6. GPU와 케이스 호환성 (GPU 길이와 케이스 지원 길이 기준)
def update_gpu_case_compatibility():
    if not check_if_update_needed("gpu_case_compatibility", ["gpu", "case_chassis"]):
        return
        
    if gpus.empty or cases.empty:
        logging.warning("GPU 또는 케이스 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    gpu_cols = get_column_names(gpus, ['gpu_id', 'length'])
    case_cols = get_column_names(cases, ['case_id', 'vga_length'])
    
    # ID 컬럼 이름이 다를 수 있음
    if gpu_cols['gpu_id'] is None:
        for possible_id in ['id', 'vga_id']:
            if possible_id in gpus.columns:
                gpu_cols['gpu_id'] = possible_id
                break
                
    if case_cols['case_id'] is None:
        for possible_id in ['case_chassis_id', 'id', 'chassis_id']:
            if possible_id in cases.columns:
                case_cols['case_id'] = possible_id
                break
    
    # GPU 길이 컬럼이 다를 수 있음
    if case_cols['vga_length'] is None:
        for possible_col in ['gpu_length', 'max_gpu_length', 'max_vga_length', 'graphics_card_length']:
            if possible_col in cases.columns:
                case_cols['vga_length'] = possible_col
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in gpu_cols.values() or None in case_cols.values():
        logging.warning(f"GPU 또는 케이스 테이블에 필요한 컬럼이 없습니다. GPU 컬럼: {gpu_cols}, 케이스 컬럼: {case_cols}")
        return
    
    compatibility_data = []
    
    for gpu_id, gpu_length in zip(gpus[gpu_cols['gpu_id']], gpus[gpu_cols['length']]):
        for case_id, case_vga_length in zip(cases[case_cols['case_id']], cases[case_cols['vga_length']]):
            compatible = 0
            
            # 둘 다 숫자 값이 있는 경우에만 비교
            if gpu_length is not None and case_vga_length is not None:
                try:
                    if float(gpu_length) <= float(case_vga_length):
                        compatible = 1
                except (ValueError, TypeError):
                    pass
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'gpu_id': gpu_id,
                'case_id': case_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        # GPU-케이스 호환성 테이블이 없는 경우 생성
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpu_case_compatibility (
                    id INTEGER PRIMARY KEY,
                    gpu_id INTEGER,
                    case_id INTEGER,
                    compatible BOOLEAN,
                    FOREIGN KEY (gpu_id) REFERENCES gpu (gpu_id),
                    FOREIGN KEY (case_id) REFERENCES case_chassis (case_id)
                )
            """)
            logging.info("GPU-케이스 호환성 테이블 생성 완료")
        except Exception as e:
            logging.warning(f"GPU-케이스 호환성 테이블 생성 중 오류: {e}")
        
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM gpu_case_compatibility")
        conn.execute("INSERT INTO gpu_case_compatibility SELECT * FROM df")
        logging.info(f"GPU-케이스 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("GPU-케이스 호환성 데이터가 없습니다.")

# 7. PSU와 케이스 호환성 (PSU 폼팩터와 케이스 지원 폼팩터 기준)
def update_psu_case_compatibility():
    if not check_if_update_needed("psu_case_compatibility", ["power_supply", "case_chassis"]):
        return
        
    if psus.empty or cases.empty:
        logging.warning("PSU 또는 케이스 데이터가 없어 호환성 업데이트를 건너뜁니다.")
        return
    
    # 컬럼 이름 확인
    psu_cols = get_column_names(psus, ['psu_id', 'form_factor'])
    case_cols = get_column_names(cases, ['case_id', 'power_supply_type'])
    
    # ID 컬럼 이름이 다를 수 있음
    if psu_cols['psu_id'] is None:
        for possible_id in ['id', 'power_id', 'power_supply_id']:
            if possible_id in psus.columns:
                psu_cols['psu_id'] = possible_id
                break
                
    if case_cols['case_id'] is None:
        for possible_id in ['case_chassis_id', 'id', 'chassis_id']:
            if possible_id in cases.columns:
                case_cols['case_id'] = possible_id
                break
    
    # 필요한 컬럼이 없으면 건너뛰기
    if None in psu_cols.values() or None in case_cols.values():
        logging.warning(f"PSU 또는 케이스 테이블에 필요한 컬럼이 없습니다. PSU 컬럼: {psu_cols}, 케이스 컬럼: {case_cols}")
        return
    
    compatibility_data = []
    
    for psu_id, psu_form_factor in zip(psus[psu_cols['psu_id']], psus[psu_cols['form_factor']]):
        for case_id, case_psu_support in zip(cases[case_cols['case_id']], cases[case_cols['power_supply_type']]):
            compatible = 0
            
            # PSU 폼팩터가 케이스 지원 타입에 포함되어 있으면 호환됨
            if psu_form_factor is not None and case_psu_support is not None:
                try:
                    psu_form_lower = str(psu_form_factor).lower()
                    case_support_lower = str(case_psu_support).lower()
                    
                    # ATX PSU는 대부분의 케이스와 호환됨
                    if 'atx' in psu_form_lower and 'atx' in case_support_lower:
                        compatible = 1
                    # SFX PSU는 SFX 지원 케이스와만 호환됨
                    elif 'sfx' in psu_form_lower and 'sfx' in case_support_lower:
                        compatible = 1
                except (ValueError, TypeError):
                    pass
            
            compatibility_data.append({
                'id': len(compatibility_data) + 1,
                'psu_id': psu_id,
                'case_id': case_id,
                'compatible': compatible
            })
    
    if compatibility_data:
        # PSU-케이스 호환성 테이블이 없는 경우 생성
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS psu_case_compatibility (
                    id INTEGER PRIMARY KEY,
                    psu_id INTEGER,
                    case_id INTEGER,
                    compatible BOOLEAN,
                    FOREIGN KEY (psu_id) REFERENCES power_supply (psu_id),
                    FOREIGN KEY (case_id) REFERENCES case_chassis (case_id)
                )
            """)
            logging.info("PSU-케이스 호환성 테이블 생성 완료")
        except Exception as e:
            logging.warning(f"PSU-케이스 호환성 테이블 생성 중 오류: {e}")
        
        df = pd.DataFrame(compatibility_data)
        conn.execute("DELETE FROM psu_case_compatibility")
        conn.execute("INSERT INTO psu_case_compatibility SELECT * FROM df")
        logging.info(f"PSU-케이스 호환성 테이블 업데이트 완료: {len(df)}개 항목")
    else:
        logging.warning("PSU-케이스 호환성 데이터가 없습니다.")

# 10. 시스템 호환성 (전체 시스템 호환성 종합)
def update_system_compatibility():
    # 시스템 호환성은 다른 호환성 테이블을 기반으로 계산하므로
    # 다른 호환성 테이블이 모두 업데이트된 후에 실행해야 함
    logging.info("시스템 호환성 테이블은 필요할 때 동적으로 계산됩니다.")
    pass

# 모든 호환성 테이블 업데이트 함수 수정 - 오류 처리 개선
def update_all_compatibility_tables():
    logging.info("호환성 테이블 업데이트 시작")
    
    # 각 호환성 테이블 업데이트 함수 호출
    try:
        update_cpu_mb_compatibility()
    except Exception as e:
        logging.error(f"CPU-메인보드 호환성 업데이트 중 오류: {e}")
    
    try:
        update_cpu_cooler_compatibility()
    except Exception as e:
        logging.error(f"CPU-쿨러 호환성 업데이트 중 오류: {e}")
    
    try:
        update_cooler_case_compatibility()
    except Exception as e:
        logging.error(f"쿨러-케이스 호환성 업데이트 중 오류: {e}")
    
    try:
        update_mb_case_compatibility()
    except Exception as e:
        logging.error(f"메인보드-케이스 호환성 업데이트 중 오류: {e}")
    
    try:
        update_mb_memory_compatibility()
    except Exception as e:
        logging.error(f"메인보드-메모리 호환성 업데이트 중 오류: {e}")
    
    try:
        update_gpu_case_compatibility()
    except Exception as e:
        logging.error(f"GPU-케이스 호환성 업데이트 중 오류: {e}")
    
    try:
        update_psu_case_compatibility()
    except Exception as e:
        logging.error(f"PSU-케이스 호환성 업데이트 중 오류: {e}")
    
    # 마지막 업데이트 시간 저장
    save_last_update_time()
    
    logging.info("호환성 테이블 업데이트 완료")

if __name__ == "__main__":
    try:
        update_all_compatibility_tables()
        conn.close()
    except Exception as e:
        logging.error(f"호환성 테이블 업데이트 중 오류 발생: {e}")
        conn.close()