from google import genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any
from tavily import TavilyClient

load_dotenv('./.env')
genai.configure(api_key=os.getenv('GEMINI_API_KEY_JY'))
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY_JY')
model = genai.GenerativeModel('gemini-1.5-flash')

# 사용자 질문
user_query = "2025년에 게재된 LLM 관련 탑티어 논문에 대해 알려줘"

# 상태 타입 정의
StateType = Dict[str, Any]

# Tavily 클라이언트 초기화
tavily_client = TavilyClient()

class Summary(BaseModel):
    summary: str
    
class JudgeSummary(BaseModel):
    is_adequate: bool
    basis_for_judgment: str

Search_result = """Abstract
Test-time scaling is a promising new approach to language modeling that uses extra test-time compute to improve performance. Recently, OpenAI's o1 model showed this capability but did not publicly share its methodology, leading to many replication efforts. We seek the simplest approach to achieve test-time scaling and strong reasoning performance. First, we curate a small dataset s1K of 1,000 questions paired with reasoning traces relying on three criteria we validate through ablations: difficulty, diversity, and quality. Second, we develop budget forcing to control test-time compute by forcefully terminating the model's thinking process or lengthening it by appending "Wait" multiple times to the model's generation when it tries to end. This can lead the model to double-check its answer, often fixing incorrect reasoning steps. After supervised finetuning the Qwen2.5-32B-Instruct language model on s1K and equipping it with budget forcing, our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24). Further, scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24. Our model, data, and code are open-source at https://github.com/simplescaling/s1.

Machine Learning, ICML, Large language models, Test-time scaling, Test-time compute
1Introduction
Refer to caption
Figure 1:Test-time scaling with s1-32B. We benchmark s1-32B on reasoning-intensive tasks and vary test-time compute.
Performance improvements of language models (LMs) over the past years have largely relied on scaling up train-time compute using large-scale self-supervised pretraining (Kaplan et al., 2020; Hoffmann et al., 2022). The creation of these powerful models has set the stage for a new scaling paradigm built on top of them: test-time scaling. The aim of this approach is to increase the compute at test time to get better results. There has been much work exploring this idea (Snell et al., 2024; Welleck et al., 2024), and the viability of this paradigm was recently validated by OpenAI o1 (OpenAI, 2024). o1 has demonstrated strong reasoning performance with consistent gains from scaling test-time compute. OpenAI describes their approach as using large-scale reinforcement learning (RL) implying the use of sizable amounts of data (OpenAI, 2024). This has led to various attempts to replicate their models relying on techniques like Monte Carlo Tree Search (Gao et al., 2024b; Zhang et al., 2024a), multi-agent approaches (Qin et al., 2024), and others (Wang et al., 2024a; Huang et al., 2024b, 2025). Among these approaches, DeepSeek R1 (DeepSeek-AI et al., 2025) has successfully replicated o1-level performance, also employing reinforcement learning via millions of samples and multiple training stages. However, despite the large number of o1 replication attempts, none have openly replicated a clear test-time scaling behavior. Thus, we ask: what is the simplest approach to achieve both test-time scaling and strong reasoning performance?

We show that training on only 1,000 samples with next-token prediction and controlling thinking duration via a simple test-time technique we refer to as budget forcing leads to a strong reasoning model that scales in performance with more test-time compute. Specifically, we construct s1K, which consists of 1,000 carefully curated questions paired with reasoning traces and answers distilled from Gemini Thinking Experimental (Google, 2024). We perform supervised fine-tuning (SFT) of an off-the-shelf pretrained model on our small dataset requiring just 26 minutes of training on 16 H100 GPUs. After training, we control the amount of test-time compute our model spends using budget forcing: (I) If the model generates more thinking tokens than a desired limit, we forcefully end the thinking process by appending an end-of-thinking token delimiter. Ending the thinking this way makes the model transition to generating its answer. (II) If we want the model to spend more test-time compute on a problem, we suppress the generation of the end-of-thinking token delimiter and instead append "Wait" to the model's current reasoning trace to encourage more exploration. Equipped with this simple recipe – SFT on 1,000 samples and test-time budget forcing – our model s1-32B exhibits test-time scaling (Figure 1). Further, s1-32B is the most sample-efficient reasoning model and outperforms closed-source models like OpenAI's o1-preview (Figure 2).

We conduct extensive ablation experiments targeting (a) our selection of 1,000 (1K) reasoning samples and (b) our test-time scaling. For (a), we find that jointly incorporating difficulty, diversity, and quality measures into our selection algorithm is important. Random selection, selecting samples with the longest reasoning traces, or only selecting maximally diverse samples all lead to significantly worse performance (around 
−
30% on AIME24 on average). Training on our full data pool of 59K examples, a superset of s1K, does not offer substantial gains over our 1K selection. This highlights the importance of careful data selection and echoes prior findings for instruction tuning (Zhou et al., 2023). For (b), we define desiderata for test-time scaling methods to compare different approaches. Budget forcing leads to the best scaling as it has perfect controllability with a clear positive slope leading to strong performance.

In summary, our contributions are: We develop simple methods for creating a sample-efficient reasoning dataset (§2) and test-time scaling (§3); Based on these we build s1-32B which is competitive with o1-preview (§4); We ablate subtleties of data (§5.1) and test-time scaling (§5.2). We end with a discussion to motivate future work on simple reasoning (§6). Our code, model, and data are open-source at https://github.com/simplescaling/s1."""

prompt = f"""Please summarize the following search results using the Chain-of-Density method.

Search Results: {Search_result}

Use this JSON schema:

result = {{
    'summary': 'Provide the summary IN KOREAN following this structure:
1. First, write a short initial summary (1-2 sentences)
2. Then, gradually expand the summary by adding more specific details in each iteration
3. Create at least 3 increasingly detailed versions
4. Finally, combine the key information into a comprehensive summary'
}}

"""

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY_JY'))
summary_response = client.models.generate_content(
    model='gemini-1.5-flash',
    contents=prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Summary,
    },
)

# print(summary_response.text)

judge_summary_prompt = f"""You are an expert in evaluating the adequacy of summaries.

Please analyze if this summary properly captures the full content and determine if it is adequate.

Summary: {summary_response.text}

Full Content: {Search_result}

Use this JSON schema:

result = {{
    'basis_for_judgment': 'Please provide the basis for your judgment in korean',
    'is_adequate': 'Write true if the summary includes the full content, false if it does not'
}}
"""

judge_summary_response = client.models.generate_content(
    model='gemini-1.5-flash',
    contents=judge_summary_prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': JudgeSummary,
    },
)

# print(judge_summary_response.text)

ss_response = client.models.generate_content(
    model='gemini-1.5-flash',
    contents=judge_summary_prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': JudgeSummary,
    },
)