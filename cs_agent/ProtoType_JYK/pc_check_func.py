import json
import duckdb
import re

DB_PATH = '/home/wlsdud022/AgentFactory/cs_agent/db/pc_parts.db'

# DB 스키마 정보 가져오기
def get_db_table():
    conn = duckdb.connect(DB_PATH)
    tables = conn.execute("SHOW TABLES").fetchall()
    conn.close()
    return tables

def get_db_samples(tables):
    conn = duckdb.connect(DB_PATH)
    table_columns = {}
    for table in tables:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
        table_columns[table] = [col[1] for col in columns]
    conn.close()
    return table_columns

def sql(query):
    conn = duckdb.connect(DB_PATH)
    fixed_query = fix_union_query(query)
    try:
        result = conn.sql(fixed_query).fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        # print(f"Query: {query}")
        conn.close()
        return [e]

def multiple_sql(queries):
    conn = duckdb.connect(DB_PATH)
    all_results = []
    for query_item in queries:
        query_type = query_item['type']
        query_sql = query_item['sql']
        try:
            query_results = conn.sql(query_sql).fetchall()
            all_results.extend(query_results)
            print(f"Successfully executed {query_type} query, found {len(query_results)} results")
        except Exception as e:
            print(f"Error executing {query_type} query: {e}")
            # print(f"Query: {query_sql}")
            all_results.extend([e])
    conn.close()
    return all_results

def load_db_description():
    db_desc_path = '/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK/db_desc.json'
    try:
        with open(db_desc_path, 'r', encoding='utf-8') as f:
            db_description = json.load(f)
        return db_description
    except FileNotFoundError:
        print(f"경고: {db_desc_path} 파일을 찾을 수 없습니다.")
        return {}
    except json.JSONDecodeError:
        print(f"경고: {db_desc_path} 파일의 JSON 형식이 올바르지 않습니다.")
        return {}
    
def fix_union_query(query):
    pattern = r'(.*)\s+LIMIT\s+\d+\s+UNION ALL\s+(.*)'
    replacement = r'\1 UNION ALL \2'
    fixed_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return fixed_query
