import duckdb
import pandas as pd

# 데이터베이스 연결
conn = duckdb.connect('pc_parts.db')

# AMD 7800 시리즈 CPU 정보 확인
print("AMD 7800 시리즈 CPU 정보:")
cpu_info = conn.sql("""
    SELECT cpu_id, model_name, socket_type 
    FROM cpu 
    WHERE model_name LIKE '%7800%' AND manufacturer = 'AMD'
""")
cpu_info.show()

# 소켓 타입 확인
socket_types = cpu_info.fetchall()
if not socket_types:
    print("AMD 7800 시리즈 CPU를 찾을 수 없습니다.")
else:
    # 소켓 타입 기반으로 호환되는 마더보드 조회
    print("\n소켓 타입 기반 호환 마더보드:")
    motherboards = conn.sql("""
        SELECT mb_id, model_name, manufacturer, socket_type, chipset, form_factor
        FROM motherboard
        WHERE socket_type IN (
            SELECT socket_type 
            FROM cpu 
            WHERE model_name LIKE '%7800%' AND manufacturer = 'AMD'
        )
        ORDER BY manufacturer, chipset
    """)
    motherboards.show()
    
    # 호환성 테이블이 있는 경우 더 정확한 조회
    print("\n호환성 테이블 기반 호환 마더보드:")
    try:
        compatible_motherboards = conn.sql("""
            SELECT m.mb_id, m.model_name, m.manufacturer, m.socket_type, m.chipset, m.form_factor
            FROM motherboard m
            JOIN cpu_mb_compatibility c ON m.socket_type = c.socket_type
            JOIN cpu cp ON c.cpu_id = cp.cpu_id
            WHERE cp.model_name LIKE '%7800%' AND cp.manufacturer = 'AMD'
            ORDER BY m.manufacturer, m.chipset
        """)
        compatible_motherboards.show()
    except Exception as e:
        print(f"호환성 테이블 조회 중 오류 발생: {e}")
        print("호환성 테이블이 없거나 구조가 다를 수 있습니다.")

# 결과를 데이터프레임으로 변환하여 추가 분석 가능
try:
    df_motherboards = motherboards.df()
    print(f"\n총 {len(df_motherboards)}개의 호환 마더보드를 찾았습니다.")
    
    # 제조사별 통계
    print("\n제조사별 호환 마더보드 수:")
    manufacturer_counts = df_motherboards['manufacturer'].value_counts()
    for manufacturer, count in manufacturer_counts.items():
        print(f"- {manufacturer}: {count}개")
    
    # 폼팩터별 통계
    print("\n폼팩터별 호환 마더보드 수:")
    form_factor_counts = df_motherboards['form_factor'].value_counts()
    for form_factor, count in form_factor_counts.items():
        print(f"- {form_factor}: {count}개")
except Exception as e:
    print(f"데이터프레임 변환 중 오류 발생: {e}")

# 데이터베이스 연결 종료
# conn.close()