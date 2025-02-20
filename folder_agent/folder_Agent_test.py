from glob import glob
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
load_dotenv('./.env')
genai.configure(api_key=os.getenv('GEMINI_API_KEY_JY'))

model = genai.GenerativeModel('gemini-1.5-flash')

def find_csv_files_method1(directory):
    csv_files = glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    return csv_files

find_dir = './folder_agent/data'
csv_files = find_csv_files_method1(find_dir)

for i in csv_files:
    data = pd.read_csv(i)
    condition = '2020년부터 2024년까지의 회계 자료들을 찾아줘 하나의 년도만 있어도 좋아'

    prompt = f"""You are an expert in finding the data that the user needs.
    
    Classify the data according to the conditions provided by the user using the following JSON format.

    Provided Condition: {condition}
    Provided Data: {data}

    Use this JSON schema:

    Information = {{'is_desired_data': Yes or No, 'reason': 'Provide a detailed explanation of why the data meets or does not meet the user's requirements. The explanation must be in Korean.', 'file_path': {i}}}
    Return Information
    """
    response = model.generate_content(prompt)
    
    log_dir = './folder_agent/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'classification_log.txt')
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n파일명: {i}\n")
        f.write(f"분류 결과:\n{response.text}\n")
        f.write("="*50 + "\n")
    
    time.sleep(3)