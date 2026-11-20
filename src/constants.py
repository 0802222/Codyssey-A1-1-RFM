"""여러 파일이 함께 쓰는 공통 상수.

한 파일에서만 쓰는 상수는 그 파일 상단에 두고, 두 파일 이상이 쓰면 여기에 둔다.
"""

# 파일 경로를 다루는 기본 모듈
import os

# 프로젝트 최상위 폴더 (이 파일의 두 단계 위)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 파일이 있는 폴더
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 거래 데이터 파일 경로 (H&M 거래 샘플: 고객 1,000명의 구매 기록)
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions_sample.csv")

# 상품 데이터 파일 경로 (거래 샘플에 나온 상품 정보)
ARTICLES_PATH = os.path.join(DATA_DIR, "articles_sample.csv")

# 상품 이미지 배열 파일 경로 (32x32x3으로 줄인 이미지를 담은 npz)
IMAGES_PATH = os.path.join(DATA_DIR, "images_sample.npz")

# RFM 세그먼트 표시 순서. 결과표와 차트를 이 순서로 보여준다
SEGMENT_ORDER = ["VIP", "Loyal", "New", "Regular", "At Risk", "Churned"]
