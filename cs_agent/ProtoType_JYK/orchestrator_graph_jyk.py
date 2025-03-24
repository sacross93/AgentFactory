import sys
sys.path.append('/home/wlsdud022/AgentFactory/cs_agent/ProtoType_JYK')
from cs_agent.ProtoType_JYK.pc_check_graph import run_pc_check
from web_search_langraph import run_web_search
from orchestrator_agent import orchestrator_agent, web_search_based_pc_check, sumary_answer_agent
from langgraph.graph import StateGraph, END

user_query = "배틀그라운드 권장사양을 알고싶어. 권장사양에 맞게 추천 제품도 알려줘"

select_agents = orchestrator_agent(user_query)

if len(select_agents['agents']) == 1:
    using_agent = select_agents['agents'][0]['agent_type']
    if using_agent == "web_search":
        result = run_web_search(user_query)
    elif using_agent == "pc_check":
        result = run_pc_check(user_query)

elif len(select_agents['agents']) > 1:
    using_agent = [agent['agent_type'] for agent in select_agents['agents']]
    if "web_search" in using_agent:
        web_result = run_web_search(user_query)
    if "pc_check" in using_agent:
        web_pc_extracted = web_search_based_pc_check(web_result)
        pc_result = run_pc_check(web_pc_extracted)
        final_answer = sumary_answer_agent(user_question=user_query, pc_check_result=pc_result, web_search_result=web_result)
else:
    result = sumary_answer_agent(user_question=user_query, pc_check_result="찾지 못함", web_search_result="찾지 못함")





