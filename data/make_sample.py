"""H&M 원본 데이터에서 분석용 샘플을 만드는 스크립트.

원본(거래 3천만 건, 이미지 10만 장)은 너무 커서 고객 일부만 뽑아 쓴다.
고정 시드(42)를 쓰므로 몇 번을 실행해도 같은 고객 1,000명이 뽑힌다.
이 파일의 `rng`는 Random Number Generator(난수 생성기)의 줄임말이다.

원본 폴더 구조 (Kaggle H&M 대회 데이터)
    transactions_train.csv  거래 (날짜, 고객, 상품, 가격, 판매 채널)
    articles.csv            상품 정보 (상품명, 종류, 색 등)
    images/앞3자리/상품번호.jpg  상품 이미지

생성 파일 (data/ 폴더)
    transactions_sample.csv  고객 1,000명의 전체 거래
    articles_sample.csv      그 거래에 나온 상품 정보
    images_sample.npz        상품 이미지를 32x32로 줄인 배열

실행: python -m data.make_sample --raw-dir <H&M 원본 폴더>  (프로젝트 최상위 폴더에서)

생성 파일은 대회 데이터를 복제한 것이라 Git에 올리지 않는다. (.gitignore)
"""

# 명령줄 인자를 읽는 기본 모듈
import argparse

# 파일 경로를 다루는 기본 모듈
import os

# 이름 붙은 묶음(튜플)을 만드는 도구
from typing import NamedTuple

# 배열 계산 라이브러리
import numpy as np

# 표(DataFrame) 처리 라이브러리
import pandas as pd

# 이미지 파일을 NumPy 배열로 읽는 함수 (과제가 허용한 imread)
from matplotlib.pyplot import imread

# 공통 상수: 데이터 폴더, 결과 파일 경로
from src.constants import ARTICLES_PATH, DATA_DIR, IMAGES_PATH, TRANSACTIONS_PATH

# 난수 시드(Seed). 같은 값이면 항상 같은 고객이 뽑힌다. (42는 개발자 관습값이다)
SAMPLE_SEED = 42

# 뽑을 고객 수
N_SAMPLE_CUSTOMERS = 1000

# 줄인 이미지의 한 변 길이 (32 x 32)
IMAGE_SIZE = 32

# 원본 거래 파일을 한 번에 읽는 행 수 (메모리를 아끼려고 조각으로 나눠 읽는다)
CHUNK_SIZE = 1_000_000

# 상품 번호 자릿수 (이미지 파일 이름과 맞추려고 앞을 0으로 채운다)
ARTICLE_ID_LENGTH = 10

# 이미지 폴더 이름에 쓰는 상품 번호 앞자리 수 (images/앞3자리/...)
IMAGE_FOLDER_DIGITS = 3

# 색 채널 수 (RGB)
COLOR_CHANNELS = 3


# 샘플 만들기 결과 요약
class SampleSummary(NamedTuple):
    """샘플 개수 요약."""

    # 뽑힌 고객 수
    customer_count: int

    # 샘플 거래 행 수
    transaction_count: int

    # 샘플 상품 수
    article_count: int

    # 이미지가 있는 상품 수
    image_count: int


# 원본 거래 파일에서 고객 번호를 무작위로 고른다
# 고객 번호를 정렬한 뒤 같은 시드의 난수로 뽑으므로 항상 같은 고객이 나온다.
def pick_customer_ids(
    transactions_path: str,
    n_customers: int = N_SAMPLE_CUSTOMERS,
    seed: int = SAMPLE_SEED,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    # 전체 고객 번호를 모을 집합 (중복은 자동으로 사라진다)
    all_customer_ids: set[str] = set()

    # 조각마다 customer_id 열만 읽는다
    for chunk in pd.read_csv(
        transactions_path, usecols=["customer_id"], chunksize=chunk_size
    ):
        # 이 조각에 나온 고객 번호를 집합에 더한다
        all_customer_ids.update(chunk["customer_id"].unique())

    # 정렬해서 순서를 고정한다 (집합은 순서가 매번 달라질 수 있다)
    sorted_customer_ids = np.array(sorted(all_customer_ids))

    # 시드를 정한 난수 생성기를 만든다
    rng = np.random.default_rng(seed)

    # 고객 중 n_customers명을 중복 없이 무작위로 고른다
    return rng.choice(sorted_customer_ids, size=n_customers, replace=False)


# 고른 고객의 거래만 뽑는다
# 원본을 조각으로 읽으면서 고른 고객의 행만 모아 하나의 표로 합친다.
def extract_transactions(
    transactions_path: str,
    customer_ids: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    # 고른 고객의 행을 담을 목록
    picked_chunks = []

    # 조각마다 (상품 번호는 앞자리 0을 살리려고 문자열로 읽는다)
    for chunk in pd.read_csv(
        transactions_path, dtype={"article_id": "string"}, chunksize=chunk_size
    ):
        # 고른 고객의 행만 목록에 담는다
        picked_chunks.append(chunk[chunk["customer_id"].isin(customer_ids)])

    # 조각들을 하나의 표로 합쳐 돌려준다
    return pd.concat(picked_chunks, ignore_index=True)


# 상품 정보를 읽어 필요한 상품만 남긴다
# 상품 번호를 10자리로 맞춘 뒤 article_ids에 있는 상품만 돌려준다.
def load_articles(articles_path: str, article_ids: np.ndarray) -> pd.DataFrame:
    # 상품 정보를 읽는다 (상품 번호는 문자열로)
    articles = pd.read_csv(articles_path, dtype={"article_id": "string"})

    # 이미지 파일 이름에 맞게 번호 앞을 0으로 채워 10자리로 만든다
    articles["article_id"] = articles["article_id"].str.zfill(ARTICLE_ID_LENGTH)

    # 필요한 상품만 남기고 행 번호를 다시 매긴다
    return articles[articles["article_id"].isin(article_ids)].reset_index(drop=True)


# 이미지 한 장을 size x size 크기로 줄인다
# 세로, 가로를 같은 간격으로 size칸씩 뽑아서 줄인다. (슬라이싱 다운샘플링)
def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    # 흑백(세로, 가로) 이미지면
    if image.ndim == 2:
        # 같은 값을 3번 쌓아 컬러 모양으로 맞춘다
        image = np.stack([image] * COLOR_CHANNELS, axis=-1)

    # 색 채널만 앞의 3개(RGB)를 쓴다. 투명도 채널이 있으면 버린다
    image = image[..., :COLOR_CHANNELS]

    # 값이 0~1 소수형이면
    if image.dtype != np.uint8:
        # 0~255 정수 범위로 바꾼다
        image = np.rint(image * 255).astype(np.uint8)

    # 세로 방향에서 size칸을 고르게 뽑을 줄 번호를 만든다
    row_indexes = np.linspace(0, image.shape[0] - 1, size).astype(int)

    # 가로 방향에서 size칸을 고르게 뽑을 열 번호를 만든다
    col_indexes = np.linspace(0, image.shape[1] - 1, size).astype(int)

    # 줄 번호와 열 번호가 만나는 점만 골라 줄인 이미지를 만든다
    resized_image: np.ndarray = image[np.ix_(row_indexes, col_indexes)]

    # 줄인 이미지를 돌려준다
    return resized_image


# 상품 이미지를 줄여서 배열 하나로 묶는다
# 이미지 파일이 없는 상품은 건너뛴다. (원본에 이미지가 없는 상품이 있다)
def build_image_arrays(
    articles: pd.DataFrame, images_dir: str, image_size: int = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    # 줄인 이미지를 담을 목록
    resized_images = []

    # 이미지가 실제로 있는 상품 번호를 담을 목록
    image_article_ids = []

    # 파일을 한 장씩 여는 부분만 반복문이다
    for article_id in articles["article_id"]:
        # 이 상품의 이미지 경로 (images/앞3자리/상품번호.jpg)
        image_path = os.path.join(
            images_dir, article_id[:IMAGE_FOLDER_DIGITS], f"{article_id}.jpg"
        )

        # 이미지 파일이 없으면 건너뛴다
        if not os.path.exists(image_path):
            continue

        # 이미지를 읽어 줄인 뒤 목록에 담는다
        resized_images.append(resize_image(imread(image_path), image_size))

        # 이 상품은 이미지가 있으니 번호를 남긴다
        image_article_ids.append(article_id)

    # 상품 번호 배열과 (상품 수, 세로, 가로, 색) 이미지 배열을 돌려준다
    return np.array(image_article_ids, dtype=str), np.stack(resized_images)


# 원본에서 샘플 3개 파일을 만든다
# 고객 고르기 -> 거래 뽑기 -> 상품 정보 -> 이미지 줄이기 -> 파일 저장 순서로 한다.
def make_sample(
    raw_dir: str,
    output_dir: str = DATA_DIR,
    n_customers: int = N_SAMPLE_CUSTOMERS,
    seed: int = SAMPLE_SEED,
    image_size: int = IMAGE_SIZE,
    chunk_size: int = CHUNK_SIZE,
) -> SampleSummary:
    # 원본 파일 경로 3개
    raw_transactions_path = os.path.join(raw_dir, "transactions_train.csv")
    raw_articles_path = os.path.join(raw_dir, "articles.csv")
    raw_images_dir = os.path.join(raw_dir, "images")

    # 고객을 고르고 그 고객의 거래를 뽑는다
    customer_ids = pick_customer_ids(
        raw_transactions_path, n_customers, seed, chunk_size
    )
    transactions = extract_transactions(raw_transactions_path, customer_ids, chunk_size)

    # 거래에 나온 상품의 정보를 읽는다
    articles = load_articles(raw_articles_path, transactions["article_id"].unique())

    # 상품 이미지를 줄여 배열로 만든다
    image_article_ids, images = build_image_arrays(articles, raw_images_dir, image_size)

    # 결과 폴더가 없으면 만든다
    os.makedirs(output_dir, exist_ok=True)

    # 거래 샘플을 CSV로 저장한다
    transactions.to_csv(
        os.path.join(output_dir, os.path.basename(TRANSACTIONS_PATH)), index=False
    )

    # 상품 정보 샘플을 CSV로 저장한다
    articles.to_csv(
        os.path.join(output_dir, os.path.basename(ARTICLES_PATH)), index=False
    )

    # 상품 번호와 이미지 배열을 npz 파일 하나로 저장한다
    np.savez_compressed(
        os.path.join(output_dir, os.path.basename(IMAGES_PATH)),
        article_ids=image_article_ids,
        images=images,
    )

    # 개수를 요약해서 돌려준다
    return SampleSummary(
        customer_count=int(transactions["customer_id"].nunique()),
        transaction_count=len(transactions),
        article_count=len(articles),
        image_count=len(images),
    )


# 명령줄에서 실행했을 때의 진입점
# 원본 폴더를 인자로 받아 샘플을 만들고 개수를 출력한다.
def main() -> None:
    # 명령줄 인자 해석기를 만든다
    parser = argparse.ArgumentParser(description="H&M 원본에서 분석용 샘플을 만든다")

    # 원본 폴더 경로 인자
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(DATA_DIR, "raw"),
        help="transactions_train.csv, articles.csv, images/가 있는 폴더",
    )

    # 인자를 읽는다
    arguments = parser.parse_args()

    # 샘플을 만든다
    summary = make_sample(arguments.raw_dir)

    # 만든 개수를 출력한다
    print(f"고객 {summary.customer_count:,}명")
    print(f"거래 {summary.transaction_count:,}건")
    print(f"상품 {summary.article_count:,}개 (이미지 {summary.image_count:,}장)")


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 샘플을 만든다
    main()
