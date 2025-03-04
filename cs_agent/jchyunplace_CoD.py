from glob import glob
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import requests
from bs4 import BeautifulSoup

load_dotenv('./.env')
genai.configure(api_key=os.getenv('GEMINI_API_KEY_JY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def crawl_product_info(url):
    # (crawl_product_info 함수는 이전과 동일)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        
        info_div = soup.find("div", {"class": "more_info info"})
        data = {}
        
        if info_div:
            table = info_div.find("table")
            if table:
                rows = table.find_all("tr")
                current_section = ""
                
                for row in rows:
                    th = row.find("th", {"colspan": "2"})
                    if th:
                        current_section = th.text.strip()
                        continue
                        
                    cells = row.find_all(["td"])
                    if len(cells) == 4:
                        key1 = cells[0].text.strip()
                        value1 = cells[1].text.strip()
                        key2 = cells[2].text.strip()
                        value2 = cells[3].text.strip()
                        
                        if value1:
                            data[key1] = value1
                        if value2:
                            data[key2] = value2
                            
        return pd.DataFrame(list(data.items()), columns=['항목', '값'])
    return None

urls = [
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=37715",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=72838",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=104584",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=40163",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=44346",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=84638",
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=35314"
]

df_list = []
for url in urls:
    df = crawl_product_info(url)
    if df is not None:
        df_list.append(df)
    time.sleep(1)


Recommended_specifications = {
    "구분": ["운영 체제 (OS)", "그래픽 카드 (GPU)", "메모리 (RAM)", "저장 공간 (Storage)", "프로세서 (CPU)"],
    "최소 사양": [
        "Windows 10/11, Linux, macOS",
        "NVIDIA GTX 1060 6GB, AMD Radeon RX 580 8GB (4GB VRAM 이상)",
        "8GB",
        "10GB 이상 (SSD 권장)",
        "Intel Core i5-8400, AMD Ryzen 5 2600"
    ],
    "권장 사양": [
        "Windows 11, Linux, macOS Monterey",
        "NVIDIA RTX 3060 이상 (6GB VRAM 이상)",
        "16GB ~ 32GB",
        "50GB ~ 100GB (SSD)",
        "Intel Core i7-12700K, AMD Ryzen 9 5900X"
    ]
}
Recommended_specifications_df = pd.DataFrame(Recommended_specifications)

user_question = "Stable diffusion을 내 PC에서 실행하고 싶은데 견적을 작성했어. 적합한 사양인지 확인해줘. 장비간에 호환이 되는지도 궁금해"



# --- CoD 프롬프팅 시작 ---
prompt = f"""You are a computer hardware expert. A user wants to run Stable Diffusion and is asking for a compatibility check of their PC parts.

User Question: {user_question}
Product Data: {df_list}
Recommended Specifications: {Recommended_specifications_df}

Analyze the components and provide a concise response (maximum 10 sentences) that covers:
1. Overall suitability for Stable Diffusion
2. Any critical compatibility issues between components
3. Key recommendations or upgrades needed
4. Please do not omit the product information and be sure to enter it all again.

Important:
- Focus only on the most important points
- Respond in Korean
- Keep the response brief and actionable
"""

response = model.generate_content(prompt)
print("\n--- Analysis Results ---\n", response.text)


Recommended_specifications = {
    "구분": ["운영 체제 (OS)", "그래픽 카드 (GPU)", "메모리 (RAM)", "저장 공간 (Storage)", "프로세서 (CPU)"],
    "최소 사양": [
        "Windows 10/11, Linux, macOS",
        "NVIDIA GTX 1060 6GB, AMD Radeon RX 580 8GB (4GB VRAM 이상)",
        "8GB",
        "10GB 이상 (SSD 권장)",
        "Intel Core i5-8400, AMD Ryzen 5 2600"
    ],
    "권장 사양": [
        "Windows 11, Linux, macOS Monterey",
        "NVIDIA RTX 3060 이상 (6GB VRAM 이상)",
        "16GB ~ 32GB",
        "50GB ~ 100GB (SSD)",
        "Intel Core i7-12700K, AMD Ryzen 9 5900X"
    ]
}
Recommended_specifications_df = pd.DataFrame(Recommended_specifications)

user_question = "Stable diffusion을 내 PC에서 실행하고 싶은데 견적을 작성했어. 적합한 사양인지 확인해줘. 장비간에 호환이 되는지도 궁금해"



# --- CoD 프롬프팅 시작 ---
prompt = f"""You are a computer hardware expert.  A user wants to run Stable Diffusion and is asking for a compatibility check and spec review of their chosen PC parts.

User Question: {user_question}

Product Data: {df_list}

Recommended Specifications for Stable Diffusion: {Recommended_specifications_df}

**Step 1: Initial Analysis (Low Density)**
Provide a brief overview of the user's goal, the provided PC parts, and potential areas of concern based on the recommended Stable Diffusion specifications. Don't go into deep detail yet.

Please do not omit the product information and be sure to enter it all again.
"""

response1 = model.generate_content(prompt)
print("--- Step 1: Initial Analysis ---\n", response1.text)


prompt2 = f"""{response1.text}

**Step 2:  Component-Level Breakdown (Medium Density)**

Based on the initial analysis, now provide a more detailed breakdown of EACH component in the user's build. Compare each component (CPU, GPU, RAM, Storage, Motherboard, PSU - if available) to the recommended specs.  Mention any *potential* compatibility issues, but don't definitively state them yet.  For each component, state whether it meets the minimum, recommended, or exceeds the requirements for Stable Diffusion.

Please do not omit the product information and be sure to enter it all again.
"""
response2 = model.generate_content(prompt2)
print("\n--- Step 2: Component-Level Breakdown ---\n", response2.text)


prompt3 = f"""{response2.text}

**Step 3: Compatibility Analysis and Recommendations (High Density)**

Now, perform a thorough compatibility analysis.  Consider:

*   **CPU/Motherboard Socket Compatibility:**  Are the CPU and motherboard compatible?
*   **RAM Compatibility:**  DDR type, speed, and motherboard support.
*   **Power Supply (PSU):** Is the PSU wattage sufficient for all components, with headroom for stability?
*   **GPU Length and Case Size:** Will the GPU physically fit in the chosen case? (You might not have case info, so state this limitation).
*   **Storage Interface:**  Are the storage devices (SSD/HDD) compatible with the motherboard's interfaces (M.2 NVMe, SATA)?
*   **Cooling:** Is the cooling solution adequate for the CPU and overall system? (You likely won't have full cooling details, so state this as a general recommendation)

Based on this *detailed* analysis, give CONCRETE recommendations.
*   If parts are incompatible, state this CLEARLY and suggest alternatives.
*   If the system is underpowered for Stable Diffusion, recommend specific upgrades.
*   If the system is overkill, suggest more cost-effective alternatives (optional).
*   Provide a final summary of the build's suitability for Stable Diffusion.
*   Answer in Korean.
"""

response3 = model.generate_content(prompt3)
print("\n--- Step 3: Compatibility Analysis and Recommendations ---\n", response3.text)