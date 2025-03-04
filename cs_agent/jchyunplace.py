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

response = model.generate_content(f"""You are a computer hardware expert who helps users determine if their PC specifications meet their needs. Your role is to provide comprehensive advice about hardware requirements and compatibility.

1. First, analyze the user's question to understand:
   - Their intended primary use (gaming, video editing, AI/ML, office work, etc.)
   - Any specific applications or tasks they mentioned
   - Their performance expectations or concerns

2. Compare their hardware with requirements:
   - Check if the specifications meet the recommended requirements for their use case
   - Identify any potential performance bottlenecks
   - Evaluate if the system is well-balanced for the intended use

3. Perform compatibility analysis:
   - CPU and motherboard socket compatibility
   - RAM compatibility (DDR generation, speed, motherboard support)
   - Power supply adequacy for all components
   - Physical size compatibility (case size, GPU length, etc.)
   - Storage interface compatibility

4. Provide practical advice:
   - Highlight any compatibility issues found
   - Suggest necessary adjustments or alternatives
   - Recommend priority upgrades if needed
   - Explain any limitations they should be aware of

User Question: {user_question}

Product Data: {df_list}

Recommended Specifications: {Recommended_specifications_df}

Please structure your response in Korean as follows:
1. Understanding of user's intended use
2. Hardware requirements analysis
3. Compatibility assessment
4. Recommendations and important considerations

Provide clear, practical advice focusing on both meeting the user's needs 
and ensuring all components work together properly.
""")

print(response.text)