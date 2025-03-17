import duckdb
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import re

# 1. DuckDB 데이터베이스 연결
db = SQLDatabase.from_uri("duckdb:///./cs_agent/db/pc_parts.db")
engine = create_engine("duckdb:///./cs_agent/db/pc_parts.db")

# 2. Ollama 모델 초기화
llm = OllamaLLM(
    model="qwen2.5-coder:32b",
    base_url="http://192.168.110.102:11434",
    temperature=0.1
)

# 3. SQL 쿼리 생성 체인 설정
sql_chain = create_sql_query_chain(llm, db)

# 4. 자연어 질문으로 SQL 쿼리 생성
question = "cpu 5600x랑 호환되는 PC 구성 출력해줘"
sql_query_raw = sql_chain.invoke({"question": question})

sql_query = re.search(r'```sql\n(.*?)```', sql_query_raw, re.DOTALL)
if sql_query:
    sql_query = sql_query.group(1).strip()
else:
    raise ValueError("SQL 쿼리를 응답에서 추출할 수 없습니다.")

# 5. 생성된 쿼리 출력 및 실행
print("Generated SQL Query:", sql_query)

# SQLAlchemy 엔진으로 Pandas DataFrame 가져오기
df = pd.read_sql_query(sql_query, engine)
print("Query Result (Pandas DataFrame):\n", df)