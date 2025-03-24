import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from cs_agent.ProtoType_JYK.pc_check_graph import run_pc_check
from web_search_langraph import run_web_search
from typing import Dict, Any, List, Literal
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="gemma3:27b", base_url="http://192.168.110.102:11434")

# 어떤 Agent graph를 실행시킬지 판단하는 중앙 Agent
## 중앙 Agent Json 구조(pydantic)
class AgentSelection(BaseModel):
    reason: str = Field(description="Explain why you think this Agent is suitable to resolve the user's query.")
    agent_type: Literal["pc_check", "web_search"] = Field(description="The type of agent to execute.")

class OrchestratorAgent(BaseModel):
    reasoning: str = Field(description="Overall reasoning for the agent selection strategy.")
    agents: List[AgentSelection] = Field(description="List of agents to execute for resolving the user's query.")

# JSON Output Parser 설정
orchestrator_parser = JsonOutputParser(pydantic_object=OrchestratorAgent)

## 중앙 Agent 정의
def orchestrator_agent(user_query: str) -> Dict[str, Any]:
    oa_prompt = f"""
    You are an expert analyst who determines which specialized agent(s) should be deployed to resolve the user's query.
    
    Analyze the user's question carefully and decide which agent(s) would be most appropriate to address it.
    
    Available agents:
    1. pc_check: This agent has access to our company's computer parts database. Use this agent ONLY when the query requires:
       - Retrieving specific model names and numbers from our database
       - Checking hardware compatibility between components
       - Accessing detailed technical specifications for computer products in our catalog
       - Finding compatible computer configurations from our product lineup
       - DO NOT use this agent if the user is merely mentioning their hardware specs without needing database verification
       - DO NOT use this agent for general performance questions that can be answered through web search
    
    2. web_search: This agent can search the internet for information. Use this agent when the query requires:
       - Finding general information not specific to our computer database
       - Researching solutions, reviews, or comparisons
       - Getting up-to-date information about products, technologies, or trends
       - Finding game requirements and performance benchmarks
       - Answering questions that require broader knowledge
    
    IMPORTANT SELECTION CRITERIA:
    - Choose ONLY the agent(s) that are strictly necessary to answer the query
    - If the user mentions their hardware but is asking about external information (like game requirements), use ONLY web_search
    - Use pc_check ONLY when information from our company's specific database is required
    - For questions about whether certain hardware can run a game well, web_search alone is usually sufficient
    
    Your response must be a JSON object with two keys:
        reasoning:
            - Provide an explanation of your overall agent selection strategy.
        agents:
            - Provide a list of selected agents.
            - Each agent should include:
                - reason: Why this agent is suitable for resolving the user's query
                - agent_type: The type of agent to execute ("pc_check" or "web_search")
    
    Here is the output schema:
    ```
    {{"properties":{{"reasoning": {{"title": "Reasoning", "description": "Overall reasoning for the agent selection strategy.", "type": "string"}}, "agents": {{"title": "Agents", "description": "List of agents to execute for resolving the user's query.", "type": "array", "items": {{"properties": {{"reason": {{"title": "Reason", "description": "Explain why you think this Agent is suitable to resolve the user's query.", "type": "string"}}, "agent_type": {{"title": "Agent Type", "description": "The type of agent to execute.", "enum": ["pc_check", "web_search"], "type": "string"}}}}, "required": ["reason", "agent_type"]}}}}}}, "required": ["reasoning", "agents"]}}
    ```
    
    User question: {user_query}
    """
    
    response = llm.invoke(oa_prompt)
    result = orchestrator_parser.parse(response)
    return result

# web agent 선택 질의
# test = orchestrator_agent("배틀그라운드 권장사양을 알고싶어. 난 CPU는 5600x 사용하고 있고 GPU는 RTX 3080사용하고 있는데 잘 돌아갈지 궁금하거든")
# pc compatibility 선택 질의
# orchestrator_agent("cpu 5600g랑 gpu는 3060 사용하고 있는데 이거랑 호환되는 메인보드랑 케이스 5개씩 알려줘")
# 둘 다 선택 질의
# orchestrator_agent("배틀그라운드 권장사양을 알고싶어. 해당 권장사양을 기반으로 호환되는 실제 부품들을 알고 싶어")

# from web_search_langraph import run_web_search
# test = run_web_search("배틀그라운드 권장사양을 알고싶어. 권장사양에 맞게 PC 구성을 하려고 하거든")

# web agent 기반 pc check 해주는 AGent
## 웹 검색 결과를 받아와서 pc check 에이전트에 전달해주는 Json 구조(pydantic)
class WebSearchResult(BaseModel):
    discription: str = Field(description="The description of the web search result.")
    cpu: str = Field(description="The cpu of the web search result.")
    gpu: str = Field(description="The gpu of the web search result.")
    ram: str = Field(description="The ram of the web search result.")
    motherboard: str = Field(description="The motherboard of the web search result.")
    psu: str = Field(description="The psu of the web search result.")
    case: str = Field(description="The case of the web search result.")
    storage: str = Field(description="The storage of the web search result.")

# web agent 기반 pc check 해주는 Agent
def web_search_based_pc_check(web_search_result: str) -> Dict[str, Any]:
    classification_prompt = PromptTemplate.from_template("""
    You are an expert at extracting computer component information from text.
    
    Analyze the provided web search result and extract any mentioned computer components.
    The output should be formatted as a JSON instance that conforms to the JSON schema below.
    
    Your response must be a JSON object with eight keys:
        description:
            - A brief summary of the web search result.
        cpu:
            - The CPU model mentioned in the web search result. If not found, use "Not specified".
        gpu:
            - The GPU model mentioned in the web search result. If not found, use "Not specified".
        ram:
            - The RAM specifications mentioned in the web search result. If not found, use "Not specified".
        motherboard:
            - The motherboard model mentioned in the web search result. If not found, use "Not specified".
        psu:
            - The power supply unit specifications mentioned in the web search result. If not found, use "Not specified".
        case:
            - The computer case mentioned in the web search result. If not found, use "Not specified".
        storage:
            - The storage specifications mentioned in the web search result. If not found, use "Not specified".
            
    Here is the output schema:
    ```
    {{"properties":{{"description": {{"title": "Description", "description": "A brief summary of the web search result.", "type": "string"}}, "cpu": {{"title": "Cpu", "description": "The CPU model mentioned in the web search result.", "type": "string"}}, "gpu": {{"title": "Gpu", "description": "The GPU model mentioned in the web search result.", "type": "string"}}, "ram": {{"title": "Ram", "description": "The RAM specifications mentioned in the web search result.", "type": "string"}}, "motherboard": {{"title": "Motherboard", "description": "The motherboard model mentioned in the web search result.", "type": "string"}}, "psu": {{"title": "Psu", "description": "The power supply unit specifications mentioned in the web search result.", "type": "string"}}, "case": {{"title": "Case", "description": "The computer case mentioned in the web search result.", "type": "string"}}, "storage": {{"title": "Storage", "description": "The storage specifications mentioned in the web search result.", "type": "string"}}}}}}
    ```
    
    Extract all computer components mentioned in the web search result and provide your response in the specified JSON format.
    
    Web search result: {web_search_result}
    """)
    
    chain = classification_prompt | llm | orchestrator_parser
    
    result = chain.invoke({"web_search_result": web_search_result})
    return result
    
# 테스트
# test2 = web_search_based_pc_check(test)
# 테스트
# test3 = run_pc_check(test2)

# 종합 결과를 기반으로 답변을 주는 Agent
def sumary_answer_agent(web_search_result: str, pc_check_result: str, user_question: str) -> Dict[str, Any]:
    final_answer_prompt = PromptTemplate.from_template("""
    당신은 결과들을 모아서 사용자이 질의를 해결하는 답변을 잘하는 전문가입니다.
    
    해당 답변들을 보고 사용자의 질의를 해결하기 위해 최선을 다해서 답변해주세요.
    1. 절대 결과에 있는 정보 외의 다른 정보를 사용하면 안됩니다.
    2. 각종 오류에 관한 말은 최대한 일반적이고 돌려서 작성해주세요.
    3. 컴퓨터에 대해 잘 모르는 사람도 이해할 수 있도록 작성해주세요.
    
    각 결과들은 아래와 같습니다.
    web search result: ```{web_search_result}```
    pc check result: ```{pc_check_result}```
    
    사용자 질의는 아래와 같습니다.
    user question: ```{user_question}```
    """)
    
    chain = final_answer_prompt | llm
    
    result = chain.invoke({"web_search_result": web_search_result, "pc_check_result": pc_check_result, "user_question": user_question})
    
    return result

# test4 = sumary_answer_agent(test, test3, "배틀그라운드 권장사양을 알고싶어. 권장사양에 맞게 PC 구성을 하려고 하거든")