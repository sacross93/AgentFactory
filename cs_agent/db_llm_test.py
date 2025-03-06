from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import json
import duckdb

# Ollama 모델 초기화
llm = OllamaLLM(
    model="exaone3.5:32b",
    base_url="http://192.168.110.102:11434"
)

user_question = "CPU AMD 7800X3D를 사용하고 있는데 메인보드 호환되는 거 리스트를 알려줘"

prompt = f"""
You are an AI assistant tasked with generating SQL queries to check compatibility between computer components based on the provided database schema. The database includes tables for individual components (CPU, motherboard, memory, GPU, power supply, case, storage, CPU cooler) and compatibility tables (e.g., `cpu_mb_compatibility`, `mb_memory_compatibility`) that define relationships between them. It also includes a `system_compatibility` table for overall system compatibility and a `compatibility_rules` table for specific compatibility rules. The user may or may not provide specific component models. Your goal is to write clear and efficient SQL queries to determine if the selected components are compatible with each other.

### Provided Database Information:
- **Table List**: `case_chassis`, `compatibility_rules`, `cooler_case_compatibility`, `cpu`, `cpu_cooler`, `cpu_cooler_compatibility`, `cpu_mb_compatibility`, `gpu`, `gpu_case_compatibility`, `gpu_psu_compatibility`, `mb_case_compatibility`, `mb_memory_compatibility`, `mb_storage_compatibility`, `memory`, `motherboard`, `power_supply`, `storage`, `system_compatibility`
- **Schema**: Each table has unique attributes (e.g., `cpu.socket_type`, `motherboard.form_factor`) and IDs (e.g., `cpu_id`, `mb_id`). Compatibility tables use a `compatible` BOOLEAN field to indicate compatibility. `system_compatibility` tracks total power consumption and issues. `compatibility_rules` defines attribute comparison rules (e.g., `exact_match`, `greater_than`).

### Instructions:
1. **Handle User Input**:
   - If the user provides specific component models (e.g., 'Intel Core i9-13900K', 'ASUS ROG Strix Z790-E'), tailor the SQL query to those models.
   - If no models are provided, create a generic compatibility query with comments for users to modify the WHERE clause.

2. **Compatibility Checks**:
   - **Component Pairs**: For each pair (e.g., CPU-motherboard, GPU-power supply), use the relevant compatibility table (e.g., `cpu_mb_compatibility`, `gpu_psu_compatibility`) to check the `compatible` field.
   - **No Direct Table**: If no compatibility table exists, refer to `compatibility_rules` or compare component attributes (e.g., `cpu.socket_type` = `motherboard.socket_type`).
   - **System-Wide Check**: For a full build, use `system_compatibility` to verify overall compatibility and power consumption.

3. **SQL Query Creation**:
   - JOIN necessary tables and apply compatibility conditions in WHERE or ON clauses.
   - Add comments to explain each query section (e.g., "-- Check CPU and motherboard compatibility").
   - Return component model names and compatibility status (`compatible`).

4. **Exception Handling**:
   - If a component is missing or lacks compatibility data, design the query to return "Compatibility cannot be determined."
   - Use COALESCE or CASE to handle NULL values.

5. **Power Consumption Check**:
   - Compare `system_compatibility.total_power_consumption` with `power_supply.wattage` to ensure sufficiency.

6. **Example Queries**:
   - CPU and Motherboard Compatibility:
     ```sql
     SELECT 
         c.model_name AS cpu_model, 
         m.model_name AS mb_model, 
         COALESCE(cmc.compatible, FALSE) AS compatible
     FROM cpu c
     LEFT JOIN cpu_mb_compatibility cmc ON c.cpu_id = cmc.cpu_id
     LEFT JOIN motherboard m ON cmc.mb_id = m.mb_id
     WHERE c.model_name = 'Intel Core i9-13900K' 
       AND m.model_name = 'ASUS ROG Strix Z790-E';
       
User Question: {user_question}
"""

response = llm.invoke(prompt)
print(response)

conn = duckdb.connect("pc_parts.db")

conn.sql("""
SELECT 
    m.model_name AS motherboard_model,
    COALESCE(cmc.compatible, FALSE) AS is_compatible
FROM 
    cpu c
JOIN 
    cpu_mb_compatibility cmc ON c.cpu_id = cmc.cpu_id
JOIN 
    motherboard m ON cmc.mb_id = m.mb_id
WHERE 
    c.model_name like '%7800X3D%'  -- Ensure correct CPU model name
    AND cmc.compatible = TRUE;             -- Filter only compatible motherboards
""")


conn.sql("""
    select *
    from cpu_mb_compatibility;
""")