# 서버 설정
OLLAMA_BASE_URL = "http://192.168.110.102:11434"
DEFAULT_MODEL = "exaone3.5:32b"

# 검색 설정
MAX_SEARCHES = 5
DEFAULT_SEARCH_QUERIES = [
    "AMD 5600G League of Legends performance",
    "Can League of Legends run on AMD 5600G without dedicated GPU"
]

# UI 설정
FAQ_QUESTIONS = [
    "제플몰에서 가장 인기있는 CPU는 무엇인가요?",
    "게이밍 PC 구성 추천해주세요",
    "그래픽카드 없이 게임을 할 수 있나요?",
    "RAM은 얼마나 필요한가요?"
]

# 모델 옵션
MODEL_OPTIONS = ["gemma3:27b", "exaone3.5:32b", "qwen2.5:32b", "deepseek-r1:32b"]

# 대화 스타일 옵션
CONVERSATION_STYLES = ["표준", "상세한 설명", "간결한 요약", "전문가 수준"]