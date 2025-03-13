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

cpu_file = find_latest_file(raw_xlsx_dir, "CPU_")

# 로그 설정
log_filename = f"cpu_insert_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# 기존 CPU 데이터 삭제
logging.info("기존 CPU 데이터 삭제 중...")
conn.execute("DELETE FROM cpu")
logging.info("기존 CPU 데이터 삭제 완료")

# CPU 데이터 로드
logging.info("CPU 엑셀 파일 로드 중...")
cpu = pd.read_excel(f"{raw_xlsx_dir}{cpu_file}")
logging.info(f"CPU 엑셀 파일 로드 완료 - {len(cpu)} 행 발견")

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
        return value.lower() in ['있음', '예', 'yes', 'true', '1', 'o']
    return bool(value)

# 숫자 추출 함수 (예: "4.2(GHz)" -> 4.2)
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

# 성공 및 실패 카운터 초기화
success_count = 0
failure_count = 0
failure_reasons = {}

# CPU 데이터 삽입
logging.info("CPU 데이터 삽입 시작...")
for index, row in cpu.iterrows():
    # CPU ID 생성 (1부터 시작)
    cpu_id = index + 1
    
    # 모델명 생성 (인텔 또는 AMD 모델명 사용)
    # 두 모델명 중 하나만 존재하는 경우가 대부분이므로 그에 맞게 처리
    intel_model = row['(인텔) 모델명'] if pd.notna(row['(인텔) 모델명']) and row['(인텔) 모델명'] != '-' else None
    amd_model = row['(AMD) 모델명'] if pd.notna(row['(AMD) 모델명']) and row['(AMD) 모델명'] != '-' else None
    
    # 제품명(전체) 컬럼 데이터 가져오기
    full_product_name = row['제품명(전체)'] if pd.notna(row['제품명(전체)']) else None
    
    if intel_model:
        model_name = intel_model
    elif amd_model:
        model_name = amd_model
    else:
        # 둘 다 없는 경우 제조사와 ID를 조합하여 고유한 모델명 생성
        manufacturer = row['수입/제조사'] if pd.notna(row['수입/제조사']) else "Unknown"
        model_name = f"{manufacturer}_CPU_{cpu_id}"
    
    # 제품명(전체)가 있으면 model_name과 product_name에 사용
    if full_product_name:
        model_name = full_product_name
        product_name = full_product_name
    else:
        product_name = model_name
    
    try:
        # 데이터 삽입 쿼리 - 컬럼명을 명시적으로 지정
        conn.execute("""
        INSERT INTO cpu (
            cpu_id, model_name, manufacturer, generation, intel_model, amd_model,
            cores, threads, socket_type, base_clock, turbo_clock, l3_cache,
            integrated_graphics, graphics_model, graphics_clock, pbp_mtp,
            tdp, process, optane, hyperthreading, sensemi, storemi,
            vr_ready, ryzen_master, v_cache, memory_support, memory_bus,
            memory_channels, package, product_name, kc_certification,
            rated_voltage, power_consumption, energy_efficiency, release_date,
            manufacturer_importer, country_of_origin, size, weight,
            key_features, warranty, as_contact, certification, origin,
            manufacturer_info, as_manager, customer_service, ryzen_ai, ppt,
            intel_xtu, intel_dlboost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cpu_id,                                                                  # cpu_id
            model_name,                                                              # model_name
            replace_nan(row['수입/제조사']),                                          # manufacturer
            replace_nan(row['세대명']),                                               # generation
            replace_nan(row['(인텔) 모델명']),                                        # intel_model
            replace_nan(row['(AMD) 모델명']),                                         # amd_model
            extract_number(row['코어 갯수']),                                         # cores
            extract_number(row['쓰레드']),                                            # threads
            replace_nan(row['소켓 형태']),                                            # socket_type
            extract_number(row['동작 클럭']),                                         # base_clock
            extract_number(row['터보 클럭']),                                         # turbo_clock
            replace_nan(row['L3 캐시메모리']),                                        # l3_cache
            to_boolean(row['내장그래픽']),                                            # integrated_graphics
            replace_nan(row['그래픽 코어 모델']),                                     # graphics_model
            extract_number(row['그래픽 코어 클럭']),                                  # graphics_clock
            replace_nan(row['PBP/MTP']),                                             # pbp_mtp
            extract_number(row['열 설계 전력(TDP)']),                                 # tdp
            replace_nan(row.get('제조공정', None)),                                   # process
            to_boolean(row.get('옵테인', None)),                                      # optane
            to_boolean(row.get('하이퍼스레드', None)),                                # hyperthreading
            to_boolean(row.get('SENSEMI', None)),                                    # sensemi
            to_boolean(row.get('StoreMI', None)),                                    # storemi
            to_boolean(row.get('VR Ready 프리미엄', None)),                           # vr_ready
            to_boolean(row.get('Ryzen Master', None)),                               # ryzen_master
            to_boolean(row.get('3D V캐시', None)),                                    # v_cache
            replace_nan(row.get('지원 메모리 규격', None)),                            # memory_support
            replace_nan(row.get('메모리 버스', None)),                                 # memory_bus
            extract_number(row.get('메모리 채널', None)),                              # memory_channels
            replace_nan(row.get('패키지', None)),                                      # package
            product_name,                                                             # product_name
            replace_nan(row.get('KC 인증정보', None)),                                 # kc_certification
            replace_nan(row.get('정격전압', None)),                                    # rated_voltage
            replace_nan(row.get('소비전력', None)),                                    # power_consumption
            replace_nan(row.get('에너지소비효율등급', None)),                           # energy_efficiency
            replace_nan(row.get('동일모델의 출시년월', None)),                          # release_date
            replace_nan(row.get('제조자,수입품의 경우 수입자를 함께 표기', None)),      # manufacturer_importer
            replace_nan(row.get('제조국', None)),                                      # country_of_origin
            replace_nan(row.get('크기', None)),                                        # size
            replace_nan(row.get('무게', None)),                                        # weight
            replace_nan(row.get('주요사항', None)),                                    # key_features
            replace_nan(row.get('품질보증기준', None)),                                # warranty
            replace_nan(row.get('A/S 책임자와 전화번호', None)),                       # as_contact
            replace_nan(row.get('법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항', None)),  # certification
            replace_nan(row.get('제조국 또는 원산지', None)),                          # origin
            replace_nan(row.get('제조사/수입품의 경우 수입자를 함께 표기', None)),      # manufacturer_info
            replace_nan(row.get('A/S 책임자', None)),                                 # as_manager
            replace_nan(row.get('소비자상담 관련 전화번호', None)),                    # customer_service
            to_boolean(row.get('AMD Ryzen AI', None)),                               # ryzen_ai
            replace_nan(row.get('PPT', None)),                                       # ppt
            to_boolean(row.get('인텔 XTU', None)),                                    # intel_xtu
            to_boolean(row.get('인텔 딥러닝부스트', None))                             # intel_dlboost
        ))
        success_count += 1
        logging.info(f"CPU {cpu_id}: {model_name} 삽입 성공")
    except Exception as e:
        failure_count += 1
        error_msg = str(e)
        logging.error(f"CPU {cpu_id} 삽입 실패: {error_msg}")
        
        # 에러 유형별 카운팅
        if error_msg in failure_reasons:
            failure_reasons[error_msg] += 1
        else:
            failure_reasons[error_msg] = 1
        continue

# 결과 요약 로깅
logging.info("=" * 50)
logging.info(f"CPU 데이터 삽입 완료: 총 {len(cpu)}개 중 {success_count}개 성공, {failure_count}개 실패")
if failure_count > 0:
    logging.info("실패 원인 분석:")
    for reason, count in failure_reasons.items():
        logging.info(f"- {reason}: {count}개")
logging.info("=" * 50)

# 데이터 확인
result = conn.sql("SELECT COUNT(*) AS count FROM cpu").fetchone()[0]
logging.info(f"CPU 테이블 레코드 수: {result}")

# 샘플 데이터 확인
logging.info("CPU 테이블 샘플 데이터:")
conn.sql("SELECT * FROM cpu LIMIT 5").show()

# 데이터베이스 연결 닫기
conn.close()
logging.info("데이터베이스 연결 종료")

print(f"로그 파일이 {log_filename}에 저장되었습니다.")