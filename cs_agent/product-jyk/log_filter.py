import re
from datetime import datetime

def filter_health_check_logs(log_file_path, output_file_path):
    """
    헬스 체크 관련 로그를 제외하고 나머지 로그만 필터링하여 저장합니다.
    
    Args:
        log_file_path (str): 원본 로그 파일 경로
        output_file_path (str): 필터링된 로그를 저장할 파일 경로
    """
    health_check_patterns = [
        r"헬스 체크 요청 받음",
        r"서버 상태 확인",
        r"상태 확인",
        r"health check",
        r"checkServerHealth"
    ]
    
    with open(log_file_path, 'r', encoding='utf-8') as infile:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 헬스 체크 관련 패턴이 포함된 라인은 건너뛰기
                if not any(pattern in line.lower() for pattern in health_check_patterns):
                    outfile.write(line)

def main():
    # 현재 시간을 포함한 로그 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = "api_server.log"
    output_file = f"filtered_logs_{timestamp}.log"
    
    try:
        filter_health_check_logs(input_file, output_file)
        print(f"로그 필터링 완료: {output_file}")
    except Exception as e:
        print(f"로그 필터링 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 