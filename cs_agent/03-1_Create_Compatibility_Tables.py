import duckdb
import logging
import datetime

# 로그 설정
log_filename = f"compatibility_tables_create_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

try:
    # 테이블 컬럼 확인 함수
    def get_table_columns(table_name):
        try:
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            return [col[1] for col in columns]
        except Exception as e:
            logging.error(f"{table_name} 테이블 컬럼 확인 중 오류: {str(e)}")
            return []

    # 누락된 컬럼 확인 및 추가 함수
    def add_column_if_not_exists(table_name, column_name, column_type):
        try:
            columns = get_table_columns(table_name)
            
            if column_name not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                logging.info(f"{table_name} 테이블에 {column_name} 컬럼 추가 완료")
            else:
                logging.info(f"{table_name} 테이블에 {column_name} 컬럼이 이미 존재합니다")
        except Exception as e:
            logging.error(f"{table_name} 테이블에 {column_name} 컬럼 추가 중 오류: {str(e)}")

    # PSU 테이블에 form_factor 컬럼 추가 (없는 경우)
    add_column_if_not_exists("power_supply", "form_factor", "VARCHAR")

    # 1. CPU와 메인보드 호환성 테이블 생성
    logging.info("CPU-메인보드 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cpu_mb_compatibility (
        id INTEGER PRIMARY KEY,
        cpu_id INTEGER,
        mb_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (cpu_id) REFERENCES cpu (cpu_id),
        FOREIGN KEY (mb_id) REFERENCES motherboard (mb_id)
    )
    """)
    logging.info("CPU-메인보드 호환성 테이블 생성 완료")

    # 2. CPU와 쿨러 호환성 테이블 생성
    logging.info("CPU-쿨러 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cpu_cooler_compatibility (
        id INTEGER PRIMARY KEY,
        cpu_id INTEGER,
        cooler_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (cpu_id) REFERENCES cpu (cpu_id),
        FOREIGN KEY (cooler_id) REFERENCES cpu_cooler (cooler_id)
    )
    """)
    logging.info("CPU-쿨러 호환성 테이블 생성 완료")

    # 3. 쿨러와 케이스 호환성 테이블 생성
    logging.info("쿨러-케이스 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cooler_case_compatibility (
        id INTEGER PRIMARY KEY,
        cooler_id INTEGER,
        case_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (cooler_id) REFERENCES cpu_cooler (cooler_id),
        FOREIGN KEY (case_id) REFERENCES case_chassis (case_id)
    )
    """)
    logging.info("쿨러-케이스 호환성 테이블 생성 완료")

    # 4. 메인보드와 케이스 호환성 테이블 생성
    logging.info("메인보드-케이스 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mb_case_compatibility (
        id INTEGER PRIMARY KEY,
        mb_id INTEGER,
        case_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (mb_id) REFERENCES motherboard (mb_id),
        FOREIGN KEY (case_id) REFERENCES case_chassis (case_id)
    )
    """)
    logging.info("메인보드-케이스 호환성 테이블 생성 완료")

    # 5. 메인보드와 메모리 호환성 테이블 생성
    logging.info("메인보드-메모리 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mb_memory_compatibility (
        id INTEGER PRIMARY KEY,
        mb_id INTEGER,
        memory_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (mb_id) REFERENCES motherboard (mb_id),
        FOREIGN KEY (memory_id) REFERENCES memory (memory_id)
    )
    """)
    logging.info("메인보드-메모리 호환성 테이블 생성 완료")

    # 6. GPU와 케이스 호환성 테이블 생성
    logging.info("GPU-케이스 호환성 테이블 생성 중...")
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

    # 7. PSU와 케이스 호환성 테이블 생성
    logging.info("PSU-케이스 호환성 테이블 생성 중...")
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

    # 8. GPU와 PSU 호환성 테이블 생성
    logging.info("GPU-PSU 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS gpu_psu_compatibility (
        id INTEGER PRIMARY KEY,
        gpu_id INTEGER,
        psu_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (gpu_id) REFERENCES gpu (gpu_id),
        FOREIGN KEY (psu_id) REFERENCES power_supply (psu_id)
    )
    """)
    logging.info("GPU-PSU 호환성 테이블 생성 완료")

    # 9. 메인보드와 스토리지 호환성 테이블 생성
    logging.info("메인보드-스토리지 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS mb_storage_compatibility (
        id INTEGER PRIMARY KEY,
        mb_id INTEGER,
        storage_id INTEGER,
        compatible BOOLEAN,
        FOREIGN KEY (mb_id) REFERENCES motherboard (mb_id),
        FOREIGN KEY (storage_id) REFERENCES storage (storage_id)
    )
    """)
    logging.info("메인보드-스토리지 호환성 테이블 생성 완료")

    # 10. 시스템 호환성 테이블 생성
    logging.info("시스템 호환성 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS system_compatibility (
        id INTEGER PRIMARY KEY,
        cpu_id INTEGER,
        mb_id INTEGER,
        cooler_id INTEGER,
        memory_id INTEGER,
        gpu_id INTEGER,
        storage_id INTEGER,
        case_id INTEGER,
        psu_id INTEGER,
        compatible BOOLEAN,
        compatibility_issues TEXT,
        FOREIGN KEY (cpu_id) REFERENCES cpu (cpu_id),
        FOREIGN KEY (mb_id) REFERENCES motherboard (mb_id),
        FOREIGN KEY (cooler_id) REFERENCES cpu_cooler (cooler_id),
        FOREIGN KEY (memory_id) REFERENCES memory (memory_id),
        FOREIGN KEY (gpu_id) REFERENCES gpu (gpu_id),
        FOREIGN KEY (storage_id) REFERENCES storage (storage_id),
        FOREIGN KEY (case_id) REFERENCES case_chassis (case_id),
        FOREIGN KEY (psu_id) REFERENCES power_supply (psu_id)
    )
    """)
    logging.info("시스템 호환성 테이블 생성 완료")

    # 11. 호환성 규칙 테이블 생성
    logging.info("호환성 규칙 테이블 생성 중...")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS compatibility_rules (
        rule_id INTEGER PRIMARY KEY,
        component_type1 VARCHAR,
        component_type2 VARCHAR,
        rule_description TEXT,
        rule_logic TEXT,
        priority INTEGER
    )
    """)
    logging.info("호환성 규칙 테이블 생성 완료")

    # CPU 테이블 컬럼 확인
    cpu_columns = get_table_columns("cpu")
    logging.info(f"CPU 테이블 컬럼: {cpu_columns}")
    
    # CPU 쿨러 테이블 컬럼 확인
    cooler_columns = get_table_columns("cpu_cooler")
    logging.info(f"CPU 쿨러 테이블 컬럼: {cooler_columns}")

    # CPU-쿨러 호환성에서 발생하는 NoneType 오류 수정을 위한 데이터 업데이트
    logging.info("CPU 테이블의 소켓 정보 NULL 값 확인 및 수정...")
    if "socket_type" in cpu_columns:
        conn.execute("UPDATE cpu SET socket_type = '' WHERE socket_type IS NULL")
        logging.info("CPU 테이블의 socket_type 컬럼 NULL 값 수정 완료")
    
    logging.info("CPU 쿨러 테이블의 소켓 지원 정보 NULL 값 확인 및 수정...")
    if "socket_support" in cooler_columns:
        conn.execute("UPDATE cpu_cooler SET socket_support = '' WHERE socket_support IS NULL")
        logging.info("CPU 쿨러 테이블의 socket_support 컬럼 NULL 값 수정 완료")

    # 데이터베이스 변경사항 커밋
    conn.commit()
    logging.info("모든 호환성 테이블 생성 및 수정 완료")

except Exception as e:
    logging.error(f"호환성 테이블 생성 중 오류 발생: {str(e)}")
    try:
        conn.rollback()
    except Exception as rollback_error:
        logging.error(f"롤백 중 오류 발생: {str(rollback_error)}")

finally:
    # 데이터베이스 연결 종료
    conn.close()
    logging.info("데이터베이스 연결 종료") 