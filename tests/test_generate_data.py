"""합성 데이터 생성 스크립트(data/generate_data.py) 테스트.

실행: python -m unittest discover -s tests -v
"""

import os  # 파일 경로를 다루는 기본 모듈
import tempfile  # 임시 폴더를 만드는 기본 모듈
import unittest  # 테스트를 돌리는 기본 모듈

import numpy as np  # 배열 계산 라이브러리
import pandas as pd  # 표(DataFrame) 처리 라이브러리

from data import generate_data  # 테스트할 데이터 생성 스크립트
from src.constants import DATA_DIR, IMAGE_SHAPE  # 공통 상수: 저장 폴더, 이미지 모양


class TestProductHelpers(unittest.TestCase):  # 상품 표를 만드는 보조 함수 테스트
    def setUp(self) -> None:  # 테스트마다 같은 난수 생성기와 카테고리를 준비한다
        # 시드를 고정한 난수 생성기 (Random Number Generator)
        self.rng = np.random.default_rng(0)
        # 상품 수만큼 카테고리를 무작위로 정한다
        self.product_categories = self.rng.choice(
            list(generate_data.CATEGORIES), size=generate_data.N_PRODUCTS
        )

    def test_make_names(self) -> None:  # 상품명 만들기 테스트
        # 상품명을 만든다
        names = generate_data.make_names(self.rng, self.product_categories)
        # 상품 수만큼 만들어진다
        self.assertEqual(len(names), generate_data.N_PRODUCTS)
        # 상품명 앞부분은 그 카테고리의 꾸미는 말이다
        adjectives = generate_data.NAME_PARTS[self.product_categories[0]].adjectives
        self.assertIn(names[0].split(" ")[0], adjectives)

    def test_make_images(self) -> None:  # 이미지 배열 만들기 테스트
        # 가격은 모두 같은 값으로 둔다
        prices = np.full(generate_data.N_PRODUCTS, 30000.0)
        # 이미지를 만든다
        images = generate_data.make_images(self.rng, self.product_categories, prices)
        # 모양은 (상품 수, 16, 16, 3)이다
        self.assertEqual(images.shape, (generate_data.N_PRODUCTS,) + IMAGE_SHAPE)
        # 값 종류는 0~255 정수(uint8)이다
        self.assertEqual(images.dtype, np.uint8)

    def test_inject_missing(self) -> None:  # 결측치 넣기 테스트
        # 결측치가 없는 작은 상품 표를 만든다
        size = generate_data.N_PRODUCTS
        products = pd.DataFrame({"price": np.ones(size), "rating": np.ones(size)})
        # 결측치를 넣는다
        generate_data.inject_missing(self.rng, products)
        # 가격과 평점 모두 일부가 빈 값이 된다
        self.assertGreater(products["price"].isna().sum(), 0)
        self.assertGreater(products["rating"].isna().sum(), 0)
        # 전부 빈 값이 되지는 않는다
        self.assertLess(products["price"].isna().sum(), size)


class TestOrderHelpers(unittest.TestCase):  # 주문 표를 만드는 보조 함수 테스트
    def setUp(self) -> None:  # 테스트마다 같은 난수 생성기를 준비한다
        # 시드를 고정한 난수 생성기 (Random Number Generator)
        self.rng = np.random.default_rng(0)

    def test_activity_period(self) -> None:  # 활동 기간 정하기 테스트
        # 고객 유형마다 확인한다
        for customer_type in generate_data.CUSTOMER_TYPES:
            for _ in range(50):  # 난수가 달라도 규칙이 지켜지는지 여러 번 본다
                # 첫 구매일과 마지막 활동일을 정한다
                first_date, last_date = generate_data.activity_period(
                    self.rng, customer_type
                )
                # 마지막이 처음보다 앞서지 않는다
                self.assertLessEqual(first_date, last_date)
                # 둘 다 주문 기간 안에 있다
                self.assertGreaterEqual(first_date, generate_data.START_DATE)
                self.assertLessEqual(last_date, generate_data.END_DATE)

    def test_activity_period_new_customer(self) -> None:  # 신규 고객 기간 테스트
        # 신규 고객의 첫 구매일을 정한다
        first_date, last_date = generate_data.activity_period(self.rng, "new")
        # 마지막 날까지 활동한다
        self.assertEqual(last_date, generate_data.END_DATE)
        # 첫 구매는 최근 60일 안이다
        self.assertLess((generate_data.END_DATE - first_date).days, 60)

    def test_pick_quantity(self) -> None:  # 주문 수량 정하기 테스트
        # 수량을 많이 뽑는다
        quantities = [
            generate_data.pick_quantity(self.rng, "Electronics") for _ in range(2000)
        ]
        # 수량은 항상 1개 이상이다
        self.assertGreaterEqual(min(quantities), 1)
        # 대량 주문(10~30개)이 드물게 섞여 있다
        self.assertTrue(any(quantity >= 10 for quantity in quantities))
        # 대부분은 1~2개다
        self.assertGreater(
            sum(quantity <= 2 for quantity in quantities) / len(quantities), 0.8
        )


class TestGenerateOutput(unittest.TestCase):  # 생성 결과 전체 테스트
    def test_same_files_as_committed(self) -> None:  # 저장된 CSV와 같은지 테스트
        # 임시 폴더에 새로 생성한다 (원본 CSV는 건드리지 않는다)
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = generate_data.OUT_DIR  # 원래 저장 폴더를 기억한다
            generate_data.OUT_DIR = tmp  # 저장 폴더를 임시 폴더로 바꾼다
            try:
                generate_data.main()  # CSV를 만든다
            finally:
                generate_data.OUT_DIR = old_dir  # 저장 폴더를 되돌린다
            # 파일마다 저장된 CSV와 내용이 똑같은지 확인한다
            for name in ("products.csv", "orders.csv"):
                with open(os.path.join(tmp, name), "rb") as new_file:
                    with open(os.path.join(DATA_DIR, name), "rb") as old_file:
                        self.assertEqual(new_file.read(), old_file.read(), name)


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만
    unittest.main()  # 테스트를 실행한다
