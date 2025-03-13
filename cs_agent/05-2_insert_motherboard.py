import duckdb
import pandas as pd
import numpy as np
import logging
import datetime
import os

raw_xlsx_dir = "./cs_agent/raw_xlsx/"

def find_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.xlsx')]
    if not files:
        return None
    return max(files)

mb_file = find_latest_file(raw_xlsx_dir, "Mainboard_")

# 로그 설정
log_filename = f"motherboard_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
conn = duckdb.connect('./cs_agent/db/pc_parts.db')
logging.info("데이터베이스 연결 성공")

# 테이블 컬럼 확인
test = conn.sql("SELECT * FROM motherboard")
columns = test.columns
logging.info(f"마더보드 테이블 컬럼: {columns}")
logging.info(f"마더보드 테이블 컬럼 수: {len(columns)}")

# 기존 마더보드 데이터 삭제
logging.info("기존 마더보드 데이터 삭제 중...")
conn.execute("DELETE FROM motherboard")
logging.info("기존 마더보드 데이터 삭제 완료")

# 마더보드 데이터 로드
logging.info("마더보드 엑셀 파일 로드 중...")
motherboard = pd.read_excel(f"{raw_xlsx_dir}{mb_file}")
logging.info(f"마더보드 엑셀 파일 로드 완료 - {len(motherboard)} 행 발견")

# NaN 값을 None으로 변환하는 함수
def replace_nan(value):
    if isinstance(value, (float, np.float64)) and np.isnan(value):
        return None
    return value

# 불리언 값으로 변환하는 함수
def to_boolean(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value.lower() in ['있음', '예', 'yes', 'true', '1', 'o', '지원']
    return bool(value)

# 숫자 추출 함수 (예: "4(EA)" -> 4)
def extract_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        import re
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
    return None

# 메모리 용량 추출 함수 (예: "최대 128(GB)" -> 128)
def extract_memory_capacity(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        import re
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
    return None

# 엑셀 컬럼과 DB 컬럼 매핑
column_mapping = {
    'manufacturer': '수입/제조사',
    'cpu_support': '사용 CPU',
    'socket_type': '소켓',
    'chipset': '칩셋',
    'form_factor': '보드 규격',
    'cpu_count': 'CPU 장착 개수',
    'memory_support': '지원 메모리',
    'memory_speed': '속도',
    'memory_slots': '슬롯',
    'max_memory': '지원 용량',
    'memory_channel': '지원 채널',
    'm2_slots': 'M.2',
    'sata3': 'SATA3',
    'vga': 'D-SUB',
    'hdmi': 'HDMI',
    'display_port': 'DP',
    'pcie_x16': 'PCI-Ex. x16',
    'pcie_x1': 'PCI-Ex. x1',
    'usb_31_gen1': 'USB 3.1',
    'wifi_support': '무선랜',
    'bluetooth_support': '블루투스',
    'lan_speed': '유선랜 속도',
    'audio_chipset': '오디오 칩셋',
    'rgb_header': 'RGB 헤더',
    'argb_header': 'ARGB 헤더',
    'fan_headers': '시스템팬 헤더',
    'cpu_fan_headers': 'CPU팬 헤더',
    'pump_headers': '수냉 펌프',
    'debug_led': '디버그 LED',
    'post_display': 'POST 디스플레이',
    'clear_cmos': 'CMOS 클리어',
    'bios_flashback': 'BIOS 플래시백',
    'dual_bios': '듀얼 BIOS',
    'ez_mode': 'EZ 모드',
    'product_name': '제품명',
    'kc_certification': 'KC 인증',
    'rated_voltage': '정격전압',
    'power_consumption': '소비전력',
    'energy_efficiency': '에너지효율',
    'certification': '인증',
    'release_date': '출시일',
    'manufacturer_importer': '제조자/수입자',
    'origin': '원산지',
    'manufacturer_info': '제조사 정보',
    'country_of_origin': '제조국',
    'as_manager': 'A/S 책임자',
    'customer_service': '소비자상담 관련 전화번호',
    'as_contact': 'A/S 안내',
    'size': '크기',
    'weight': '무게',
    'key_features': '주요 특징',
    'warranty': '품질보증기준'
}

# 마더보드 데이터 삽입
logging.info("마더보드 데이터 삽입 시작...")

# 모든 컬럼 이름 가져오기 (mb_id 포함)
db_columns = columns

# INSERT 쿼리 생성 (mb_id 포함)
placeholders = ', '.join(['?'] * len(db_columns))
columns_str = ', '.join(db_columns)
insert_query = f"INSERT INTO motherboard ({columns_str}) VALUES ({placeholders})"

logging.info(f"사용할 INSERT 쿼리: {insert_query}")
logging.info(f"파라미터 개수: {len(db_columns)}")

success_count = 0
failure_count = 0
failure_reasons = {}

for idx, row in motherboard.iterrows():
    mb_id = idx + 1  # 1부터 시작하는 고유 ID 생성
    
    # 제품명 컬럼에서 직접 가져오기
    model_name = row.get('제품명(전체)', '')
    if pd.isna(model_name) or model_name == '':
        # 제품명이 없는 경우 대체 방법 사용
        manufacturer = row.get('수입/제조사', '')
        chipset = row.get('칩셋', '')
        socket = row.get('소켓', '')
        model_name = f"{manufacturer} {chipset} {socket} #{mb_id}"
    
    try:
        # 값 준비
        values = [mb_id]  # mb_id를 첫 번째 값으로 추가
        values.append(model_name)  # model_name
        
        # 나머지 컬럼 값 추가
        for col in db_columns[2:]:  # mb_id와 model_name 이후의 컬럼들
            excel_col = column_mapping.get(col)
            
            if excel_col is None:
                values.append(None)
                continue
                
            value = row.get(excel_col, None)
            
            # 데이터 타입에 따른 변환
            if col in ['memory_slots', 'max_memory', 'pcie_x16', 'pcie_x8', 'pcie_x4', 'pcie_x1', 'm2_slots', 'sata3', 'lan_ports', 'usb_31_gen2', 'usb_31_gen1', 'usb_20', 'usb_type_c', 'rgb_header', 'argb_header', 'fan_headers', 'cpu_fan_headers', 'pump_headers', 'temperature_sensors']:
                values.append(extract_number(value))
            elif col in ['memory_channel', 'cpu_count']:
                values.append(extract_number(value))
            elif col == 'max_memory':
                values.append(extract_memory_capacity(value))
            elif col in ['thunderbolt_support', 'wifi_support', 'bluetooth_support', 'display_port', 'hdmi', 'dvi', 'vga', 'corsair_header', 'aura_sync', 'mystic_light', 'rgb_fusion', 'polychrome_sync', 'razer_chroma', 'tt_rgb_plus', 'debug_led', 'post_display', 'clear_cmos', 'bios_flashback', 'dual_bios', 'ez_mode', 'sata_raid', 'nvme_raid']:
                values.append(to_boolean(value))
            else:
                values.append(replace_nan(value))
        
        # 데이터 삽입
        conn.execute(insert_query, values)
        success_count += 1
        logging.info(f"마더보드 {mb_id}: {model_name} 삽입 성공")
    except Exception as e:
        failure_count += 1
        error_msg = str(e)
        logging.error(f"마더보드 {mb_id} 삽입 실패: {error_msg}")
        logging.error(f"시도한 값: {values}")
        
        # 에러 유형별 카운팅
        if error_msg in failure_reasons:
            failure_reasons[error_msg] += 1
        else:
            failure_reasons[error_msg] = 1
        continue

# 결과 요약 로깅
logging.info("=" * 50)
logging.info(f"마더보드 데이터 삽입 완료: 총 {len(motherboard)}개 중 {success_count}개 성공, {failure_count}개 실패")
if failure_count > 0:
    logging.info("실패 원인 분석:")
    for reason, count in failure_reasons.items():
        logging.info(f"- {reason}: {count}개")
logging.info("=" * 50)

# 데이터 확인
result = conn.sql("SELECT COUNT(*) AS count FROM motherboard").fetchone()[0]
logging.info(f"마더보드 테이블 레코드 수: {result}")

# 샘플 데이터 확인
logging.info("마더보드 테이블 샘플 데이터:")
conn.sql("SELECT * FROM motherboard LIMIT 5").show()

# 데이터베이스 연결 닫기
conn.close()
logging.info("데이터베이스 연결 종료")

print(f"로그 파일이 {log_filename}에 저장되었습니다.") 
