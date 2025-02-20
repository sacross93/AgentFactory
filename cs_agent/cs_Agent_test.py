from glob import glob
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
load_dotenv('./.env')
genai.configure(api_key=os.getenv('GEMINI_API_KEY_JY'))

model = genai.GenerativeModel('gemini-1.5-flash')

import requests
from bs4 import BeautifulSoup

urls = [
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=73892423&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=21694499&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=62653715&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=16741211&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=18610958&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=75075386&marketPlaceSeq=16",
    "https://shop.danawa.com/pc/?controller=estimateDeal&methods=productInformation&productSeq=10174200&marketPlaceSeq=16"
]

all_products_data = []

for url in urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.select_one("h1.pop_title_txt")
        product_title = title.get_text(strip=True) if title else "제목 없음"
        
        table_rows = soup.select("tbody tr")
        data = {'제품명': product_title}
        
        for row in table_rows:
            columns = row.find_all(['th', 'td'])
            
            if len(columns) == 2:
                key = columns[0].get_text(strip=True)
                value = columns[1].get_text(strip=True)
                data[key] = value
            elif len(columns) == 4:
                key1 = columns[0].get_text(strip=True)
                value1 = columns[1].get_text(strip=True)
                key2 = columns[2].get_text(strip=True)
                value2 = columns[3].get_text(strip=True)
                data[key1] = value1
                data[key2] = value2
        
        all_products_data.append(data)
        
        print(f"\n제품 URL: {url}")
        for k, v in data.items():
            print(f"{k}: {v}")
    else:
        print(f"페이지 요청 실패: {response.status_code} (URL: {url})")

user_question = "내 메인보드랑 호환되는 CPU를 골라줘."

response = model.generate_content(f"""You are a computer hardware expert specializing in PC assembly. Your role is to provide accurate and professional advice to customers.

Please analyze the provided product data considering the following aspects:

1. Compatibility Analysis:
   - For CPU and Motherboard: Verify matching socket types (e.g., AM5, LGA 1700)
   - For RAM: Confirm compatible memory types (DDR4/DDR5)
   - For Power Supply: Check required power capacity and PSU specifications

2. Performance Analysis:
   - Evaluate performance suitability for intended use
   - Analyze price-to-performance ratio
   - Consider thermal performance and power efficiency

3. Additional Considerations:
   - Product features, pros, and cons
   - Future upgrade possibilities
   - Compatible alternative recommendations

User Question: {user_question}

Product Data: {all_products_data}

Please structure your response as follows:
1. Current product specifications summary
2. Detailed compatibility analysis
3. Specific recommendations with reasoning
4. Important considerations and precautions

Provide your response in Korean, ensuring it is both professional and easily understandable for the customer.
Make sure to highlight any potential compatibility issues or special requirements.
""")

print(response.text)


# from pprint import pprint
# pprint(all_products_data)