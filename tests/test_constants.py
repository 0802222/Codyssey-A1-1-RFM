"""공통 상수 파일(src/constants.py) 테스트.

실행: python -m unittest discover -s tests -v
"""

# 파일 경로를 다루는 기본 모듈
import os

# 테스트를 돌리는 기본 모듈
import unittest

# 데이터 생성 스크립트 (상수를 가져다 쓰는 쪽)
from data import generate_data

# 공통 상수 파일과 분석 파이프라인
from src import constants, pipeline


# 공통 상수 테스트 모음
class TestConstants(unittest.TestCase):
    # 세그먼트 6개와 표시 순서 테스트
    def test_segment_order(self) -> None:
        # 세그먼트는 6개이고 표시 순서가 정해져 있다
        self.assertEqual(
            constants.SEGMENT_ORDER,
            ["VIP", "Loyal", "New", "Regular", "At Risk", "Churned"],
        )

    # 이미지 모양 테스트
    def test_img_shape(self) -> None:
        # 이미지는 세로 16, 가로 16, 색 3이다
        self.assertEqual(constants.IMAGE_SHAPE, (16, 16, 3))

    # 데이터 파일 경로 테스트
    def test_data_paths(self) -> None:
        # 주문 파일은 data 폴더의 orders.csv이다
        self.assertEqual(
            constants.ORDERS_PATH, os.path.join(constants.DATA_DIR, "orders.csv")
        )

        # 상품 파일은 data 폴더의 products.csv이다
        self.assertEqual(
            constants.PRODUCTS_PATH, os.path.join(constants.DATA_DIR, "products.csv")
        )

        # 데이터 폴더는 실제로 존재한다
        self.assertTrue(os.path.isdir(constants.DATA_DIR))

    # 다른 파일이 같은 상수를 쓰는지 테스트
    def test_shared_by_other_files(
        self,
    ) -> None:
        # 생성 스크립트와 파이프라인이 같은 이미지 모양을 쓴다
        self.assertIs(generate_data.IMAGE_SHAPE, constants.IMAGE_SHAPE)

        # 파이프라인이 같은 세그먼트 순서를 쓴다
        self.assertIs(pipeline.SEGMENT_ORDER, constants.SEGMENT_ORDER)

        # 생성 스크립트가 같은 폴더에 CSV를 저장한다
        self.assertEqual(generate_data.OUT_DIR, constants.DATA_DIR)


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 테스트를 실행한다
    unittest.main()
