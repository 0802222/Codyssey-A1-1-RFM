"""공통 상수 파일(src/constants.py) 테스트.

실행: python -m unittest discover -s tests -v
"""

# 파일 경로를 다루는 기본 모듈
import os

# 테스트를 돌리는 기본 모듈
import unittest

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

    # 데이터 파일 경로 테스트
    def test_data_paths(self) -> None:
        # 거래 파일은 data 폴더의 transactions_sample.csv이다
        self.assertEqual(
            constants.TRANSACTIONS_PATH,
            os.path.join(constants.DATA_DIR, "transactions_sample.csv"),
        )

        # 상품 파일은 data 폴더의 articles_sample.csv이다
        self.assertEqual(
            constants.ARTICLES_PATH,
            os.path.join(constants.DATA_DIR, "articles_sample.csv"),
        )

        # 이미지 파일은 data 폴더의 images_sample.npz이다
        self.assertEqual(
            constants.IMAGES_PATH,
            os.path.join(constants.DATA_DIR, "images_sample.npz"),
        )

        # 데이터 폴더는 실제로 존재한다
        self.assertTrue(os.path.isdir(constants.DATA_DIR))

    # 다른 파일이 같은 상수를 쓰는지 테스트
    def test_shared_by_other_files(self) -> None:
        # 파이프라인이 같은 세그먼트 순서를 쓴다
        self.assertIs(pipeline.SEGMENT_ORDER, constants.SEGMENT_ORDER)

        # 파이프라인이 같은 거래 파일 경로를 쓴다
        self.assertEqual(pipeline.TRANSACTIONS_PATH, constants.TRANSACTIONS_PATH)


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 테스트를 실행한다
    unittest.main()
