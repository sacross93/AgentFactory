import duckdb

conn = duckdb.connect('./cs_agent/db/pc_parts.db')

# 모든 테이블의 컬럼 정보 출력 함수
def print_all_table_columns():
    tables = conn.execute("SHOW TABLES").fetchall()
    
    for table in tables:
        table_name = table[0]
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        
        print(f"\n테이블: {table_name}")
        print("컬럼:")
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            print(f"  - {col_name} ({col_type})")
            
print_all_table_columns()