"""여러 파일이 함께 쓰는 공통 상수.

한 파일에서만 쓰는 상수는 그 파일 상단에 두고, 두 파일 이상이 쓰면 여기에 둔다.
"""

# 파일 경로를 다루는 기본 모듈
import os

# 프로젝트 최상위 폴더 (이 파일의 두 단계 위)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV 데이터가 있는 폴더
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 주문 데이터 파일 경로
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")

# 상품 데이터 파일 경로
PRODUCTS_PATH = os.path.join(DATA_DIR, "products.csv")

# 이미지 모양: 세로 16, 가로 16, 색 3(RGB)
IMAGE_SHAPE = (16, 16, 3)

# RFM 세그먼트 표시 순서. 결과표와 차트를 이 순서로 보여준다
SEGMENT_ORDER = ["VIP", "Loyal", "New", "Regular", "At Risk", "Churned"]
