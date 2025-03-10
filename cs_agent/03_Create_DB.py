import duckdb
import pandas as pd
import json
import os

# 엑셀 파일 로드 (파일 존재 여부 확인)
raw_xlsx_dir = "./cs_agent/raw_xlsx/"

# 파일 패턴으로 가장 최신 파일 찾기 함수
def find_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.xlsx')]
    if not files:
        return None
    return max(files)  # 파일명 기준으로 가장 최신 파일 반환

# CPU 파일 로드
cpu_file = find_latest_file(raw_xlsx_dir, "CPU_")
if cpu_file:
    try:
        cpu = pd.read_excel(f"{raw_xlsx_dir}{cpu_file}")
        print(f"CPU 파일 로드: {cpu_file}")
    except Exception as e:
        print(f"CPU 파일 로드 중 오류 발생: {e}")
        cpu = pd.DataFrame()
else:
    print("경고: CPU 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    cpu = pd.DataFrame()

# 메인보드 파일 로드
mb_file = find_latest_file(raw_xlsx_dir, "Mainboard_")
if mb_file:
    try:
        motherboard = pd.read_excel(f"{raw_xlsx_dir}{mb_file}")
        print(f"메인보드 파일 로드: {mb_file}")
    except Exception as e:
        print(f"메인보드 파일 로드 중 오류 발생: {e}")
        motherboard = pd.DataFrame()
else:
    print("경고: 메인보드 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    motherboard = pd.DataFrame()

# 메모리 파일 로드
memory_file = find_latest_file(raw_xlsx_dir, "Memory_")
if memory_file:
    try:
        memory = pd.read_excel(f"{raw_xlsx_dir}{memory_file}")
        print(f"메모리 파일 로드: {memory_file}")
    except Exception as e:
        print(f"메모리 파일 로드 중 오류 발생: {e}")
        memory = pd.DataFrame()
else:
    print("경고: 메모리 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    memory = pd.DataFrame()

# 케이스 파일 로드
case_file = find_latest_file(raw_xlsx_dir, "Case_")
if case_file:
    try:
        case = pd.read_excel(f"{raw_xlsx_dir}{case_file}")
        print(f"케이스 파일 로드: {case_file}")
    except Exception as e:
        print(f"케이스 파일 로드 중 오류 발생: {e}")
        case = pd.DataFrame()
else:
    print("경고: 케이스 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    case = pd.DataFrame()

# GPU 파일 로드
gpu_file = find_latest_file(raw_xlsx_dir, "VGA_")
if gpu_file:
    try:
        gpu = pd.read_excel(f"{raw_xlsx_dir}{gpu_file}")
        print(f"GPU 파일 로드: {gpu_file}")
    except Exception as e:
        print(f"GPU 파일 로드 중 오류 발생: {e}")
        gpu = pd.DataFrame()
else:
    print("경고: GPU 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    gpu = pd.DataFrame()

# 파워 파일 로드
power_file = find_latest_file(raw_xlsx_dir, "Power_")
if power_file:
    try:
        power = pd.read_excel(f"{raw_xlsx_dir}{power_file}")
        print(f"파워 파일 로드: {power_file}")
    except Exception as e:
        print(f"파워 파일 로드 중 오류 발생: {e}")
        power = pd.DataFrame()
else:
    print("경고: 파워 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    power = pd.DataFrame()

# 스토리지 파일 로드
storage_file = find_latest_file(raw_xlsx_dir, "SSD_")
if storage_file:
    try:
        storage = pd.read_excel(f"{raw_xlsx_dir}{storage_file}")
        print(f"스토리지 파일 로드: {storage_file}")
    except Exception as e:
        print(f"스토리지 파일 로드 중 오류 발생: {e}")
        storage = pd.DataFrame()
else:
    print("경고: 스토리지 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    storage = pd.DataFrame()

# CPU 쿨러 파일 로드
cooler_file = find_latest_file(raw_xlsx_dir, "CpuCooler_")
if cooler_file:
    try:
        cooler = pd.read_excel(f"{raw_xlsx_dir}{cooler_file}")
        print(f"CPU 쿨러 파일 로드: {cooler_file}")
    except Exception as e:
        print(f"CPU 쿨러 파일 로드 중 오류 발생: {e}")
        cooler = pd.DataFrame()
else:
    print("경고: CPU 쿨러 엑셀 파일을 찾을 수 없습니다. 빈 DataFrame을 생성합니다.")
    cooler = pd.DataFrame()

# JSON 파일에서 컬럼 정보 로드
with open('columns_info.json', 'r', encoding='utf-8') as f:
    columns_info = json.load(f)

# 각 컴포넌트의 컬럼 정보 가져오기 (예외 처리 추가)
cpu_columns = columns_info.get('cpu', {})
motherboard_columns = columns_info.get('motherboard', {})
memory_columns = columns_info.get('memory', {})
case_columns = columns_info.get('case', {})
gpu_columns = columns_info.get('gpu', {})
power_columns = columns_info.get('power', {})
storage_columns = columns_info.get('storage', {})
# cooler_columns가 없는 경우 빈 딕셔너리 사용
cooler_columns = columns_info.get('cooler', {}) 
# 또는 'cpu_cooler' 키로 시도
if not cooler_columns and 'cpu_cooler' in columns_info:
    cooler_columns = columns_info['cpu_cooler']

# DuckDB 연결 생성
conn = duckdb.connect('./cs_agent/db/pc_parts.db')

# CPU 테이블 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS cpu (
    cpu_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    generation VARCHAR,                  -- 세대명
    intel_model VARCHAR,                 -- (인텔) 모델명
    amd_model VARCHAR,                   -- (AMD) 모델명
    cores INTEGER,                       -- 코어 갯수
    threads INTEGER,                     -- 쓰레드
    socket_type VARCHAR,                 -- 소켓 형태
    base_clock FLOAT,                    -- 동작 클럭
    turbo_clock FLOAT,                   -- 터보 클럭
    l3_cache VARCHAR,                    -- L3 캐시메모리
    integrated_graphics BOOLEAN,         -- 내장그래픽
    graphics_model VARCHAR,              -- 그래픽 코어 모델
    graphics_clock FLOAT,                -- 그래픽 코어 클럭
    pbp_mtp VARCHAR,                     -- PBP/MTP
    tdp INTEGER,                         -- 열 설계 전력(TDP)
    process VARCHAR,                     -- 제조공정
    optane BOOLEAN,                      -- 옵테인
    hyperthreading BOOLEAN,              -- 하이퍼스레드
    sensemi BOOLEAN,                     -- SENSEMI
    storemi BOOLEAN,                     -- StoreMI
    vr_ready BOOLEAN,                    -- VR Ready 프리미엄
    ryzen_master BOOLEAN,                -- Ryzen Master
    v_cache BOOLEAN,                     -- 3D V캐시
    memory_support VARCHAR,              -- 지원 메모리 규격
    memory_bus VARCHAR,                  -- 메모리 버스
    memory_channels INTEGER,             -- 메모리 채널
    package VARCHAR,                     -- 패키지
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR,                    -- 품질보증기준
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    ryzen_ai BOOLEAN,                    -- AMD Ryzen AI
    ppt VARCHAR,                         -- PPT
    intel_xtu BOOLEAN,                   -- 인텔 XTU
    intel_dlboost BOOLEAN                -- 인텔 딥러닝부스트
);
''')

# 메인보드 테이블 생성 (수정됨)
conn.execute('''
CREATE TABLE IF NOT EXISTS motherboard (
    mb_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    cpu_support VARCHAR,                 -- 사용 CPU
    socket_type VARCHAR,                 -- 소켓
    chipset VARCHAR,                     -- 칩셋
    form_factor VARCHAR,                 -- 보드 규격
    cpu_count INTEGER,                   -- CPU 장착 개수
    memory_support VARCHAR,              -- 지원 메모리
    memory_speed VARCHAR,                -- 속도
    memory_slots INTEGER,                -- 슬롯
    max_memory INTEGER,                  -- 지원 용량
    memory_channel INTEGER,              -- 지원 채널
    pcie_x16 INTEGER,                    -- PCI-Ex. x16
    pcie_x8 INTEGER,                     -- PCI-Ex. x8
    pcie_x4 INTEGER,                     -- PCI-Ex. x4
    pcie_x1 INTEGER,                     -- PCI-Ex. x1
    pcie_version VARCHAR,                -- PCIe 버전
    m2_slots INTEGER,                    -- M.2 슬롯
    m2_spec VARCHAR,                     -- M.2 규격
    sata3 INTEGER,                       -- SATA3
    sata_raid BOOLEAN,                   -- SATA RAID
    nvme_raid BOOLEAN,                   -- NVMe RAID
    thunderbolt_support BOOLEAN,         -- 썬더볼트
    wifi_support BOOLEAN,                -- 무선LAN
    bluetooth_support BOOLEAN,           -- 블루투스
    lan_speed VARCHAR,                   -- LAN 속도
    lan_ports INTEGER,                   -- LAN 포트
    audio_chipset VARCHAR,               -- 오디오 칩셋
    audio_channels VARCHAR,              -- 오디오 채널
    usb_31_gen2 INTEGER,                 -- USB 3.1 Gen2
    usb_31_gen1 INTEGER,                 -- USB 3.1 Gen1
    usb_20 INTEGER,                      -- USB 2.0
    usb_type_c INTEGER,                  -- USB Type-C
    display_port INTEGER,                -- 디스플레이포트
    hdmi INTEGER,                        -- HDMI
    dvi INTEGER,                         -- DVI
    vga INTEGER,                         -- D-SUB
    rgb_header BOOLEAN,                  -- RGB 헤더
    argb_header BOOLEAN,                 -- ARGB 헤더
    corsair_header BOOLEAN,              -- CORSAIR 헤더
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    polychrome_sync BOOLEAN,             -- POLYCHROME SYNC
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    fan_headers INTEGER,                 -- 팬 헤더
    cpu_fan_headers INTEGER,             -- CPU 팬 헤더
    pump_headers INTEGER,                -- 펌프 헤더
    temperature_sensors INTEGER,         -- 온도 센서
    debug_led BOOLEAN,                   -- 디버그 LED
    post_display BOOLEAN,                -- POST 디스플레이
    clear_cmos BOOLEAN,                  -- Clear CMOS
    bios_flashback BOOLEAN,              -- BIOS 플래시백
    dual_bios BOOLEAN,                   -- 듀얼 BIOS
    ez_mode BOOLEAN,                     -- EZ 모드
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 메모리 테이블 생성 (수정됨)
conn.execute('''
CREATE TABLE IF NOT EXISTS memory (
    memory_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    purpose VARCHAR,                     -- 용도
    classification VARCHAR,              -- 분류
    product_classification VARCHAR,      -- 제품 분류
    device_usage VARCHAR,                -- 사용 장치
    memory_standard VARCHAR,             -- 규격
    memory_type VARCHAR,                 -- 메모리 규격
    capacity INTEGER,                    -- 용량
    memory_capacity VARCHAR,             -- 메모리 용량
    clock INTEGER,                       -- 클럭
    operating_clock INTEGER,             -- 동작 클럭
    timing VARCHAR,                      -- 타이밍
    memory_timing VARCHAR,               -- 메모리 타이밍
    voltage FLOAT,                       -- 전압
    rated_voltage VARCHAR,               -- 정격전압
    package VARCHAR,                     -- 패키지
    package_composition VARCHAR,         -- 패키지 구성
    ecc BOOLEAN,                         -- ECC
    on_die_ecc BOOLEAN,                  -- 온다이ECC
    reg BOOLEAN,                         -- REG
    xmp BOOLEAN,                         -- XMP
    expo BOOLEAN,                        -- EXPO
    clock_driver BOOLEAN,                -- 클럭드라이버
    heatsink BOOLEAN,                    -- 방열판
    led BOOLEAN,                         -- LED
    led_color VARCHAR,                   -- LED색
    rgb_control BOOLEAN,                 -- RGB제어
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    polychrome_sync BOOLEAN,             -- POLYCHROME-SYNC
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 그래픽카드 테이블 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS gpu (
    gpu_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    chipset_manufacturer VARCHAR,        -- 칩셋 제조사
    gpu_type VARCHAR,                    -- GPU 종류
    chipset VARCHAR,                     -- GPU 칩셋
    process VARCHAR,                     -- 제조 공정
    core_clock INTEGER,                  -- 부스트 클럭
    memory_clock INTEGER,                -- 메모리 클럭
    memory_capacity INTEGER,             -- 메모리 용량
    memory_type VARCHAR,                 -- 메모리 종류
    memory_bus INTEGER,                  -- 메모리 버스
    cuda_cores INTEGER,                  -- CUDA 코어
    stream_processors INTEGER,           -- 스트림 프로세서
    rt_cores INTEGER,                    -- RT 코어
    tensor_cores INTEGER,                -- 텐서 코어
    interface VARCHAR,                   -- 인터페이스
    hdmi INTEGER,                        -- HDMI
    display_port INTEGER,                -- 디스플레이포트
    dvi INTEGER,                         -- DVI
    vga INTEGER,                         -- D-SUB
    power_pin VARCHAR,                   -- 전원 핀
    power_consumption INTEGER,           -- 소비전력
    recommended_psu INTEGER,             -- 권장 파워
    cooling_type VARCHAR,                -- 냉각 방식
    cooling_fan INTEGER,                 -- 냉각팬 개수
    length INTEGER,                      -- 길이
    width INTEGER,                       -- 너비
    height INTEGER,                      -- 높이
    backplate BOOLEAN,                   -- 백플레이트
    led BOOLEAN,                         -- LED 기능
    rgb BOOLEAN,                         -- RGB
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    polychrome_sync BOOLEAN,             -- POLYCHROME SYNC
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    directx VARCHAR,                     -- DirectX 지원
    opengl VARCHAR,                      -- OpenGL 지원
    opencl VARCHAR,                      -- OpenCL 지원
    vulkan VARCHAR,                      -- Vulkan 지원
    cuda VARCHAR,                        -- CUDA 지원
    physx BOOLEAN,                       -- PhysX
    sli_crossfire BOOLEAN,               -- SLI/CrossFire
    vr_ready BOOLEAN,                    -- VR Ready
    dlss BOOLEAN,                        -- DLSS
    ray_tracing BOOLEAN,                 -- 레이 트레이싱
    hdcp VARCHAR,                        -- HDCP
    multi_monitor INTEGER,               -- 멀티 모니터
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 파워 서플라이 테이블 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS power_supply (
    psu_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    product_category VARCHAR,            -- 제품분류
    wattage INTEGER,                     -- 정격출력
    pfc_type VARCHAR,                    -- PFC 방식
    plus12v FLOAT,                       -- +12V
    plus5v FLOAT,                        -- +5V
    plus3v3 FLOAT,                       -- +3.3V
    plus12v_rail VARCHAR,                -- +12V방식
    standby_power_1w BOOLEAN,            -- 대기전력1W
    efficiency VARCHAR,                  -- 80PLUS
    main_connector VARCHAR,              -- 메인커넥터
    ide_4pin INTEGER,                    -- 4핀 IDE
    pcie_8pin INTEGER,                   -- 8핀(6+2) PCI-E
    pcie_6pin INTEGER,                   -- 6핀 PCI-E
    aux_4pin INTEGER,                    -- 보조4핀
    aux_8pin INTEGER,                    -- 보조8핀
    sata_connectors INTEGER,             -- SATA
    flat_cable BOOLEAN,                  -- 플랫케이블
    reform_cable BOOLEAN,                -- 리폼 케이블
    modular_type VARCHAR,                -- 모듈러 타입
    fan_size INTEGER,                    -- 팬크기
    fan_count INTEGER,                   -- 쿨링팬 개수
    atx12v_standard VARCHAR,             -- ATX12V규격
    led_light BOOLEAN,                   -- LED라이트
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    polychrome BOOLEAN,                  -- POLYCHROME
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 케이스 테이블 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS case_chassis (
    case_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    case_type VARCHAR,                   -- 케이스 타입
    power_supply_type VARCHAR,           -- 지원 파워 규격
    power_included BOOLEAN,              -- 파워 포함 여부
    supported_mb_types VARCHAR,          -- 지원보드규격
    atx_support BOOLEAN,                 -- ATX
    matx_support BOOLEAN,                -- mATX
    itx_support BOOLEAN,                 -- MiniITX/ITX
    eatx_support BOOLEAN,                -- EATX
    cpu_cooler_height INTEGER,           -- CPU쿨러장착높이
    vga_length INTEGER,                  -- VGA장착길이
    fans_included INTEGER,               -- 기본 장착 쿨링팬
    fan_slots VARCHAR,                   -- 쿨링팬 설치공간
    front_fan VARCHAR,                   -- 전면 팬
    top_fan VARCHAR,                     -- 상단 팬
    rear_fan VARCHAR,                    -- 후면 팬
    bottom_fan VARCHAR,                  -- 하단 팬
    side_fan VARCHAR,                    -- 측면 팬
    radiator_support VARCHAR,            -- 수냉 쿨러 설치
    front_radiator VARCHAR,              -- 전면 라디에이터
    top_radiator VARCHAR,                -- 상단 라디에이터
    rear_radiator VARCHAR,               -- 후면 라디에이터
    side_radiator VARCHAR,               -- 측면 라디에이터
    usb_ports VARCHAR,                   -- USB 포트 정보
    usb_31_gen1 INTEGER,                 -- USB 3.1 Gen1
    usb_31_gen2 INTEGER,                 -- USB 3.1 Gen2
    usb_31_type_c INTEGER,               -- USB 3.1 Type-C
    usb_20 INTEGER,                      -- USB 2.0
    audio_ports BOOLEAN,                 -- 오디오 포트
    rgb_support BOOLEAN,                 -- RGB
    rgb_controller BOOLEAN,              -- RGB 컨트롤러
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    polychrome_sync BOOLEAN,             -- POLYCHROME SYNC
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    width INTEGER,                       -- 너비
    height INTEGER,                      -- 높이
    depth INTEGER,                       -- 깊이
    weight FLOAT,                        -- 무게
    material VARCHAR,                    -- 재질
    side_panel VARCHAR,                  -- 측면 패널
    dust_filter BOOLEAN,                 -- 먼지 필터
    psu_shroud BOOLEAN,                  -- PSU 쉬라우드
    hdd_bays INTEGER,                    -- 3.5인치 베이
    ssd_bays INTEGER,                    -- 2.5인치 베이
    expansion_slots INTEGER,             -- 확장 슬롯
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 스토리지 테이블 생성 (수정됨)
conn.execute('''
CREATE TABLE IF NOT EXISTS storage (
    storage_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    product_purpose VARCHAR,             -- 제품 용도
    capacity INTEGER,                    -- 용량
    interface VARCHAR,                   -- 인터페이스
    form_factor VARCHAR,                 -- 디스크 크기
    platter_count INTEGER,               -- 플래터 개수
    rpm INTEGER,                         -- 회전수
    buffer_size INTEGER,                 -- 버퍼 크기
    transfer_rate VARCHAR,               -- 전송 속도
    rv_sensor BOOLEAN,                   -- RV센서
    warranty_hours VARCHAR,              -- 사용 보증 시간
    helium_filled BOOLEAN,               -- 헬륨충전
    sed BOOLEAN,                         -- SED
    data_recovery BOOLEAN,               -- 데이터 복구
    smart BOOLEAN,                       -- S.M.A.R.T
    dsa BOOLEAN,                         -- DSA
    low_power BOOLEAN,                   -- 저전력
    ise BOOLEAN,                         -- ISE
    warranty_period VARCHAR,             -- A/S 기간
    read_speed INTEGER,                  -- 순차 읽기
    write_speed INTEGER,                 -- 순차 쓰기
    random_read INTEGER,                 -- 랜덤 읽기
    random_write INTEGER,                -- 랜덤 쓰기
    nand_type VARCHAR,                   -- NAND
    nand_layer INTEGER,                  -- NAND 레이어
    controller VARCHAR,                  -- 컨트롤러
    dram BOOLEAN,                        -- DRAM 탑재
    dram_type VARCHAR,                   -- DRAM 종류
    dram_capacity VARCHAR,               -- DRAM 용량
    cache_memory VARCHAR,                -- 캐시 메모리
    trim BOOLEAN,                        -- TRIM
    gc BOOLEAN,                          -- GC
    slc_caching BOOLEAN,                 -- SLC 캐싱
    wear_leveling BOOLEAN,               -- 웨어 레벨링
    tlc BOOLEAN,                         -- TLC
    mlc BOOLEAN,                         -- MLC
    qlc BOOLEAN,                         -- QLC
    pcie_lanes INTEGER,                  -- PCIe 레인수
    nvme_protocol BOOLEAN,               -- NVMe 프로토콜
    host_memory_buffer BOOLEAN,          -- HMB
    power_loss_protection BOOLEAN,       -- 전원 차단 보호
    hardware_encryption BOOLEAN,         -- 하드웨어 암호화
    mtbf INTEGER,                        -- MTBF
    tbw INTEGER,                         -- TBW
    dwpd FLOAT,                          -- DWPD
    heatsink BOOLEAN,                    -- 방열판
    thickness FLOAT,                     -- 두께
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# CPU 쿨러 테이블 생성 (수정됨)
conn.execute('''
CREATE TABLE IF NOT EXISTS cpu_cooler (
    cooler_id INTEGER PRIMARY KEY,
    model_name VARCHAR UNIQUE,
    manufacturer VARCHAR,                -- 수입/제조사
    cooler_type VARCHAR,                 -- 쿨러 타입 (공랭, 수랭)
    height INTEGER,                      -- 높이
    width INTEGER,                       -- 너비
    depth INTEGER,                       -- 깊이
    fan_size INTEGER,                    -- 팬 크기
    fan_count INTEGER,                   -- 팬 개수
    noise_level FLOAT,                   -- 소음 레벨
    rgb BOOLEAN,                         -- RGB 지원
    tdp_support INTEGER,                 -- 지원 TDP
    socket_support VARCHAR,              -- 지원 소켓
    water_block_material VARCHAR,        -- 수냉 워터블록 재질
    radiator_size VARCHAR,               -- 수냉 라디에이터 크기
    tube_length INTEGER,                 -- 수냉 튜브 길이
    fan_speed VARCHAR,                   -- 팬 속도
    bearing_type VARCHAR,                -- 베어링 타입
    fan_connector VARCHAR,               -- 팬 커넥터
    fan_pwm BOOLEAN,                     -- PWM 지원
    aura_sync BOOLEAN,                   -- AURA SYNC
    mystic_light BOOLEAN,                -- MYSTIC LIGHT
    rgb_fusion BOOLEAN,                  -- RGB FUSION
    polychrome_sync BOOLEAN,             -- POLYCHROME SYNC
    razer_chroma BOOLEAN,                -- RAZER CHROMA
    tt_rgb_plus BOOLEAN,                 -- TT RGB PLUS
    product_name VARCHAR,                -- 품명
    kc_certification VARCHAR,            -- KC 인증정보
    rated_voltage VARCHAR,               -- 정격전압
    power_consumption VARCHAR,           -- 소비전력
    energy_efficiency VARCHAR,           -- 에너지소비효율등급
    certification VARCHAR,               -- 법에 의한 인증, 허가 등을 받았음을 확인할 수 있는 경우 그에 대한 사항
    release_date VARCHAR,                -- 동일모델의 출시년월
    manufacturer_importer VARCHAR,       -- 제조자,수입품의 경우 수입자를 함께 표기
    origin VARCHAR,                      -- 제조국 또는 원산지
    manufacturer_info VARCHAR,           -- 제조사/수입품의 경우 수입자를 함께 표기
    country_of_origin VARCHAR,           -- 제조국
    as_manager VARCHAR,                  -- A/S 책임자
    customer_service VARCHAR,            -- 소비자상담 관련 전화번호
    as_contact VARCHAR,                  -- A/S 책임자와 전화번호
    size VARCHAR,                        -- 크기
    weight VARCHAR,                      -- 무게
    key_features TEXT,                   -- 주요사항
    warranty VARCHAR                     -- 품질보증기준
);
''')

# 호환성 테이블들 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS cpu_mb_compatibility (
    id INTEGER PRIMARY KEY,
    cpu_id INTEGER,
    mb_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (cpu_id) REFERENCES cpu(cpu_id),
    FOREIGN KEY (mb_id) REFERENCES motherboard(mb_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS mb_memory_compatibility (
    id INTEGER PRIMARY KEY,
    mb_id INTEGER,
    memory_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (mb_id) REFERENCES motherboard(mb_id),
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS mb_case_compatibility (
    id INTEGER PRIMARY KEY,
    mb_id INTEGER,
    case_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (mb_id) REFERENCES motherboard(mb_id),
    FOREIGN KEY (case_id) REFERENCES case_chassis(case_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS gpu_psu_compatibility (
    id INTEGER PRIMARY KEY,
    gpu_id INTEGER,
    psu_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (gpu_id) REFERENCES gpu(gpu_id),
    FOREIGN KEY (psu_id) REFERENCES power_supply(psu_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS gpu_case_compatibility (
    id INTEGER PRIMARY KEY,
    gpu_id INTEGER,
    case_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (gpu_id) REFERENCES gpu(gpu_id),
    FOREIGN KEY (case_id) REFERENCES case_chassis(case_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS cooler_case_compatibility (
    id INTEGER PRIMARY KEY,
    cooler_id INTEGER,
    case_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (cooler_id) REFERENCES cpu_cooler(cooler_id),
    FOREIGN KEY (case_id) REFERENCES case_chassis(case_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS cpu_cooler_compatibility (
    id INTEGER PRIMARY KEY,
    cpu_id INTEGER,
    cooler_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (cpu_id) REFERENCES cpu(cpu_id),
    FOREIGN KEY (cooler_id) REFERENCES cpu_cooler(cooler_id)
);
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS mb_storage_compatibility (
    id INTEGER PRIMARY KEY,
    mb_id INTEGER,
    storage_id INTEGER,
    compatible BOOLEAN,
    FOREIGN KEY (mb_id) REFERENCES motherboard(mb_id),
    FOREIGN KEY (storage_id) REFERENCES storage(storage_id)
);
''')

# 시스템 전체 호환성 테이블 생성
conn.execute('''
CREATE TABLE IF NOT EXISTS system_compatibility (
    system_id INTEGER PRIMARY KEY,
    cpu_id INTEGER,
    mb_id INTEGER,
    memory_id INTEGER,
    gpu_id INTEGER,
    storage_id INTEGER,
    psu_id INTEGER,
    case_id INTEGER,
    cooler_id INTEGER,
    total_power_consumption INTEGER,     -- 전체 시스템 소비 전력
    compatible BOOLEAN,                  -- 전체 호환성 여부
    compatibility_issues TEXT,           -- 호환성 문제 설명
    FOREIGN KEY (cpu_id) REFERENCES cpu(cpu_id),
    FOREIGN KEY (mb_id) REFERENCES motherboard(mb_id),
    FOREIGN KEY (memory_id) REFERENCES memory(memory_id),
    FOREIGN KEY (gpu_id) REFERENCES gpu(gpu_id),
    FOREIGN KEY (storage_id) REFERENCES storage(storage_id),
    FOREIGN KEY (psu_id) REFERENCES power_supply(psu_id),
    FOREIGN KEY (case_id) REFERENCES case_chassis(case_id),
    FOREIGN KEY (cooler_id) REFERENCES cpu_cooler(cooler_id)
);
''')

# 호환성 규칙 테이블 생성 (수정됨)
conn.execute('''
CREATE TABLE IF NOT EXISTS compatibility_rules (
    rule_id INTEGER PRIMARY KEY,
    component1_type VARCHAR,             -- 첫 번째 컴포넌트 타입 (cpu, mb, gpu 등)
    component2_type VARCHAR,             -- 두 번째 컴포넌트 타입
    rule_type VARCHAR,                   -- 규칙 타입 (exact_match, greater_than, less_than 등)
    component1_attribute VARCHAR,        -- 첫 번째 컴포넌트의 속성
    component2_attribute VARCHAR,        -- 두 번째 컴포넌트의 속성
    description VARCHAR                  -- 규칙 설명
);
''')

# 데이터베이스 연결 닫기
conn.close()

print("컴퓨터 부품 데이터베이스가 성공적으로 생성되었습니다.")