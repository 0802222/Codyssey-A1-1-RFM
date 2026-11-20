"""샘플 생성 스크립트(data/make_sample.py) 테스트.

실행: python -m unittest discover -s tests -v
"""

# 파일 경로를 다루는 기본 모듈
import os

# 임시 폴더를 만드는 기본 모듈
import tempfile

# 테스트를 돌리는 기본 모듈
import unittest

# 배열 계산 라이브러리
import numpy as np

# 표(DataFrame) 처리 라이브러리
import pandas as pd

# 이미지를 파일로 저장하는 함수 (테스트용 JPG를 만들 때만 쓴다)
from matplotlib.pyplot import imsave

# 테스트할 함수들
from data.make_sample import (
    build_image_arrays,
    extract_transactions,
    load_articles,
    make_sample,
    pick_customer_ids,
    resize_image,
)

# 가짜 고객 수
FAKE_CUSTOMER_COUNT = 10

# 고객 한 명당 가짜 거래 수
FAKE_ROWS_PER_CUSTOMER = 3

# 가짜 상품 수
FAKE_ARTICLE_COUNT = 6

# 이미지 파일이 있는 가짜 상품 수 (나머지 상품은 이미지가 없다)
FAKE_IMAGE_COUNT = 4


# 가짜 원본 폴더를 만드는 도우미
# H&M 원본과 같은 구조(transactions_train.csv, articles.csv, images/)로 만든다.
def make_fake_raw_dir(raw_dir: str) -> None:
    # 고객 번호 목록 (C00 ~ C09)
    customer_ids = [f"C{number:02d}" for number in range(FAKE_CUSTOMER_COUNT)]

    # 고객마다 거래 3건씩, 상품 번호는 1~6을 돌려 쓴다
    transactions = pd.DataFrame(
        {
            # 거래일
            "t_dat": ["2020-01-01"] * FAKE_CUSTOMER_COUNT * FAKE_ROWS_PER_CUSTOMER,
            # 고객 번호 (한 명이 3번 나온다)
            "customer_id": [
                customer_id
                for customer_id in customer_ids
                for _ in range(FAKE_ROWS_PER_CUSTOMER)
            ],
            # 상품 번호 (앞자리 0이 있는 10자리 문자열)
            "article_id": [
                f"{(row_number % FAKE_ARTICLE_COUNT) + 1:010d}"
                for row_number in range(FAKE_CUSTOMER_COUNT * FAKE_ROWS_PER_CUSTOMER)
            ],
            # 가격
            "price": [0.05] * FAKE_CUSTOMER_COUNT * FAKE_ROWS_PER_CUSTOMER,
            # 판매 채널
            "sales_channel_id": [1] * FAKE_CUSTOMER_COUNT * FAKE_ROWS_PER_CUSTOMER,
        }
    )

    # 거래 표를 CSV로 저장한다
    transactions.to_csv(os.path.join(raw_dir, "transactions_train.csv"), index=False)

    # 상품 번호는 숫자로 저장해 앞자리 0이 사라진 원본 상태를 흉내 낸다
    articles = pd.DataFrame(
        {
            # 상품 번호 (1~6)
            "article_id": list(range(1, FAKE_ARTICLE_COUNT + 1)),
            # 상품명
            "prod_name": [f"Item {number}" for number in range(FAKE_ARTICLE_COUNT)],
        }
    )

    # 상품 표를 CSV로 저장한다
    articles.to_csv(os.path.join(raw_dir, "articles.csv"), index=False)

    # 이미지는 앞의 4개 상품에만 만든다 (images/앞3자리/상품번호.jpg)
    image_dir = os.path.join(raw_dir, "images", "000")

    # 이미지 폴더를 만든다
    os.makedirs(image_dir)

    # 항상 같은 값이 나오는 난수 생성기 (rng = Random Number Generator)
    rng = np.random.default_rng(0)

    # 이미지가 있는 상품마다
    for article_number in range(1, FAKE_IMAGE_COUNT + 1):
        # 세로 8, 가로 6의 컬러 이미지를 만든다
        image = rng.integers(0, 256, size=(8, 6, 3)).astype(np.uint8)

        # JPG 파일로 저장한다
        imsave(os.path.join(image_dir, f"{article_number:010d}.jpg"), image)


# 고객 고르기 테스트 모음
class TestPickCustomerIds(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        # 임시 폴더를 만든다
        self.temp_dir = tempfile.TemporaryDirectory()

        # 가짜 원본 데이터를 만든다
        make_fake_raw_dir(self.temp_dir.name)

        # 거래 파일 경로
        self.transactions_path = os.path.join(
            self.temp_dir.name, "transactions_train.csv"
        )

    # 테스트마다 끝나고 실행되는 정리 단계
    def tearDown(self) -> None:
        # 임시 폴더를 지운다
        self.temp_dir.cleanup()

    # 요청한 수만큼, 중복 없이 고르는지
    def test_count_and_unique(self) -> None:
        # 고객 4명을 고른다
        customer_ids = pick_customer_ids(self.transactions_path, 4, seed=1)

        # 4명이어야 한다
        self.assertEqual(len(customer_ids), 4)

        # 중복이 없어야 한다
        self.assertEqual(len(set(customer_ids)), 4)

    # 같은 시드면 같은 고객이 나오는지
    def test_same_seed_same_result(self) -> None:
        # 같은 시드로 두 번 고른다
        first = pick_customer_ids(self.transactions_path, 4, seed=42)
        second = pick_customer_ids(self.transactions_path, 4, seed=42)

        # 결과가 같아야 한다
        np.testing.assert_array_equal(first, second)

    # 작은 조각으로 나눠 읽어도 결과가 같은지
    def test_chunk_size_does_not_change_result(self) -> None:
        # 조각을 크게, 작게 읽는다
        big_chunk = pick_customer_ids(self.transactions_path, 4, 42, chunk_size=1000)
        small_chunk = pick_customer_ids(self.transactions_path, 4, 42, chunk_size=3)

        # 결과가 같아야 한다
        np.testing.assert_array_equal(big_chunk, small_chunk)

    # 고객 수보다 많이 요청하면 에러가 나는지
    def test_too_many_customers_raises(self) -> None:
        # 전체 고객(10명)보다 많은 11명을 요청하면 에러여야 한다
        with self.assertRaises(ValueError):
            pick_customer_ids(self.transactions_path, 11, seed=1)


# 거래 뽑기 테스트
class TestExtractTransactions(unittest.TestCase):
    # 고른 고객의 거래만 남는지
    def test_keeps_only_picked_customers(self) -> None:
        # 임시 폴더에 가짜 원본을 만든다
        with tempfile.TemporaryDirectory() as temp_dir:
            # 가짜 원본 데이터를 만든다
            make_fake_raw_dir(temp_dir)

            # 거래 파일 경로
            transactions_path = os.path.join(temp_dir, "transactions_train.csv")

            # 고객 두 명의 거래를 작은 조각(4행)으로 나눠 읽는다
            picked = extract_transactions(
                transactions_path, np.array(["C01", "C07"]), chunk_size=4
            )

        # 고객 C01, C07만 있어야 한다
        self.assertEqual(set(picked["customer_id"]), {"C01", "C07"})

        # 고객 2명 x 거래 3건 = 6행이어야 한다
        self.assertEqual(len(picked), 2 * FAKE_ROWS_PER_CUSTOMER)

        # 상품 번호는 앞자리 0이 살아 있어야 한다
        self.assertTrue(
            all(len(article_id) == 10 for article_id in picked["article_id"])
        )


# 이미지 줄이기 테스트 모음
class TestResizeImage(unittest.TestCase):
    # 지정한 크기로 줄어들고 모서리 값이 보존되는지
    def test_shape_and_corners(self) -> None:
        # 4x6 컬러 이미지를 만든다
        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

        # 2x2로 줄인다
        resized = resize_image(image, 2)

        # 모양이 (2, 2, 3)이어야 한다
        self.assertEqual(resized.shape, (2, 2, 3))

        # 왼쪽 위 모서리는 원본의 왼쪽 위 값이어야 한다
        np.testing.assert_array_equal(resized[0, 0], image[0, 0])

        # 오른쪽 아래 모서리는 원본의 오른쪽 아래 값이어야 한다
        np.testing.assert_array_equal(resized[1, 1], image[3, 5])

    # 0~1 소수 이미지가 0~255 정수로 바뀌는지
    def test_float_image_becomes_uint8(self) -> None:
        # 값이 모두 1.0인 소수 이미지
        image = np.ones((4, 4, 3), dtype=np.float32)

        # 줄인다
        resized = resize_image(image, 2)

        # 자료형이 uint8이어야 한다
        self.assertEqual(resized.dtype, np.uint8)

        # 1.0은 255가 되어야 한다
        self.assertTrue((resized == 255).all())

    # 흑백, 투명도 포함 이미지도 (크기, 크기, 3)으로 맞춰지는지
    def test_gray_and_rgba_become_three_channels(self) -> None:
        # 흑백 이미지 (세로, 가로)
        gray = np.zeros((4, 4), dtype=np.uint8)

        # 투명도가 포함된 이미지 (세로, 가로, 4)
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)

        # 둘 다 (2, 2, 3)이어야 한다
        self.assertEqual(resize_image(gray, 2).shape, (2, 2, 3))
        self.assertEqual(resize_image(rgba, 2).shape, (2, 2, 3))


# 상품 정보 읽기 테스트
class TestLoadArticles(unittest.TestCase):
    # 번호가 10자리로 맞춰지고 필요한 상품만 남는지
    def test_zero_pad_and_filter(self) -> None:
        # 임시 폴더에 가짜 원본을 만든다
        with tempfile.TemporaryDirectory() as temp_dir:
            # 가짜 원본 데이터를 만든다
            make_fake_raw_dir(temp_dir)

            # 상품 1, 2만 남기도록 읽는다
            articles = load_articles(
                os.path.join(temp_dir, "articles.csv"),
                np.array(["0000000001", "0000000002"]),
            )

        # 상품 2개만 남아야 한다
        self.assertEqual(articles["article_id"].tolist(), ["0000000001", "0000000002"])


# 이미지 배열 만들기 테스트
class TestBuildImageArrays(unittest.TestCase):
    # 이미지가 없는 상품은 건너뛰는지
    def test_skips_missing_images(self) -> None:
        # 임시 폴더에 가짜 원본을 만든다
        with tempfile.TemporaryDirectory() as temp_dir:
            # 가짜 원본 데이터를 만든다
            make_fake_raw_dir(temp_dir)

            # 상품 6개를 모두 읽는다 (이미지는 4개에만 있다)
            articles = load_articles(
                os.path.join(temp_dir, "articles.csv"),
                np.array([f"{number:010d}" for number in range(1, 7)]),
            )

            # 이미지를 4x4로 줄여 배열로 만든다
            article_ids, images = build_image_arrays(
                articles, os.path.join(temp_dir, "images"), image_size=4
            )

        # 이미지가 있는 4개 상품만 남아야 한다
        self.assertEqual(len(article_ids), FAKE_IMAGE_COUNT)

        # 배열 모양은 (4, 4, 4, 3)이어야 한다
        self.assertEqual(images.shape, (FAKE_IMAGE_COUNT, 4, 4, 3))

        # 자료형은 uint8이어야 한다
        self.assertEqual(images.dtype, np.uint8)


# 샘플 만들기 전체 흐름 테스트
class TestMakeSample(unittest.TestCase):
    # 세 파일이 만들어지고 내용이 서로 맞는지
    def test_writes_three_files(self) -> None:
        # 가짜 원본 폴더와 결과 폴더를 만든다
        with (
            tempfile.TemporaryDirectory() as raw_dir,
            tempfile.TemporaryDirectory() as out_dir,
        ):
            # 가짜 원본 데이터를 만든다
            make_fake_raw_dir(raw_dir)

            # 고객 5명으로 샘플을 만든다
            summary = make_sample(
                raw_dir, out_dir, n_customers=5, seed=42, image_size=4, chunk_size=4
            )

            # 거래, 상품, 이미지 파일이 모두 있어야 한다
            transactions = pd.read_csv(
                os.path.join(out_dir, "transactions_sample.csv"),
                dtype={"article_id": "string"},
            )
            articles = pd.read_csv(
                os.path.join(out_dir, "articles_sample.csv"),
                dtype={"article_id": "string"},
            )
            with np.load(os.path.join(out_dir, "images_sample.npz")) as image_file:
                image_article_ids = image_file["article_ids"]
                images = image_file["images"]

        # 고객은 5명이어야 한다
        self.assertEqual(transactions["customer_id"].nunique(), 5)

        # 거래는 5명 x 3건 = 15행이어야 한다
        self.assertEqual(len(transactions), 5 * FAKE_ROWS_PER_CUSTOMER)

        # 거래에 나온 상품은 모두 상품 정보에 있어야 한다
        self.assertTrue(transactions["article_id"].isin(articles["article_id"]).all())

        # 이미지 상품은 상품 정보의 일부여야 한다
        self.assertTrue(set(image_article_ids).issubset(set(articles["article_id"])))

        # 이미지 개수와 상품 번호 개수가 같아야 한다
        self.assertEqual(len(image_article_ids), len(images))

        # 요약 값이 실제 결과와 같아야 한다
        self.assertEqual(summary.customer_count, 5)
        self.assertEqual(summary.transaction_count, len(transactions))
        self.assertEqual(summary.article_count, len(articles))
        self.assertEqual(summary.image_count, len(images))


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 테스트를 실행한다
    unittest.main()
