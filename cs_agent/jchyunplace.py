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
    "https://www.jchyunplace.co.kr/shop/product_detail.html?pd_no=72838"
]

df_list = []
for url in urls:
    df = crawl_product_info(url)
    if df is not None:
        df_list.append(df)
    time.sleep(1)

print(df_list)
