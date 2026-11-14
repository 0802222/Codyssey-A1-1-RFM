"""DataAnalyzer 단위 테스트.

실행: python -m unittest discover -s tests -v
"""

# 파일 경로를 다루는 기본 모듈
import os

# 파이썬 import 경로를 다루는 기본 모듈
import sys

# 임시 폴더를 만드는 기본 모듈
import tempfile

# 테스트를 돌리는 기본 모듈
import unittest

# 배열 계산 라이브러리
import numpy as np

# 표(DataFrame) 처리 라이브러리
import pandas as pd

# 프로젝트 루트를 import 경로에 추가한다
# tests 폴더의 한 단계 위 폴더를 추가한다
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 테스트할 분석 클래스를 가져온다
from src.pipeline import DataAnalyzer  # noqa: E402


# 이미지 배열을 CSV에 들어가는 문자열로 바꾸는 도우미
# NumPy 배열을 CSV에 저장하는 형식(공백 구분 문자열)으로 바꾼다.
def image_to_str(image_array: np.ndarray) -> str:
    # 1차원으로 펴서 숫자를 공백으로 잇는다
    return " ".join(map(str, image_array.ravel().tolist()))


# 파일 없이 표를 바로 넣은 분석기를 만드는 도우미
# 파일 없이 DataFrame을 바로 넣은 분석기를 만든다.
def make_analyzer(df: pd.DataFrame) -> DataAnalyzer:
    # 파일 경로 없이 분석기를 만든다
    analyzer = DataAnalyzer(data_path="")

    # 원본이 바뀌지 않게 복사해서 넣는다
    analyzer.df = df.copy()

    # 만든 분석기를 돌려준다
    return analyzer


# 데이터 불러오기 테스트 모음
class TestLoadData(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        # 임시 폴더를 만든다
        self.temp_dir = tempfile.TemporaryDirectory()

        orders = pd.DataFrame(
            # 가짜 주문 표
            {
                # 주문 번호
                "order_id": [1, 2, 3],
                # 고객 번호
                "customer_id": ["C1", "C1", "C2"],
                "order_date": [
                    "2025-01-01",
                    "2025-01-05",
                    "2025-01-03",
                    # 주문일(문자열)
                ],
                # 상품 번호
                "product_id": ["P1", "P2", "P1"],
                # 수량
                "quantity": [1, 2, 3],
            }
        )
        products = pd.DataFrame(
            # 가짜 상품 표
            {
                # 상품 번호
                "product_id": ["P1", "P2"],
                # 상품명
                "product_name": ["Red Shirt", "Blue Jeans Slim"],
                # 카테고리
                "category": ["Fashion", "Fashion"],
                # 가격
                "price": [10.0, 20.0],
            }
        )
        # 임시 주문 파일 경로
        self.orders_path = os.path.join(self.temp_dir.name, "orders.csv")

        # 임시 상품 파일 경로
        self.products_path = os.path.join(self.temp_dir.name, "products.csv")

        # 주문 표를 CSV로 저장한다
        orders.to_csv(self.orders_path, index=False)

        # 상품 표를 CSV로 저장한다
        products.to_csv(self.products_path, index=False)

    # 테스트마다 끝나고 실행되는 정리 단계
    def tearDown(self) -> None:
        # 임시 폴더를 지운다
        self.temp_dir.cleanup()

    # 주문 파일 하나만 읽는 경우
    def test_load_single_file(self) -> None:
        # 주문 파일만 읽는다
        df = DataAnalyzer(self.orders_path).load_data()

        # 3행이어야 한다
        self.assertEqual(len(df), 3)

    # 상품 파일을 합치고 날짜를 바꾸는 경우
    def test_load_merges_products_and_parses_dates(self) -> None:
        # 두 파일을 모두 지정한다
        analyzer = DataAnalyzer(self.orders_path, product_path=self.products_path)

        # 읽으면서 주문일을 날짜형으로 바꾼다
        df = analyzer.load_data(date_cols=["order_date"])

        # 행 수는 그대로 3이어야 한다
        self.assertEqual(len(df), 3)

        # 상품 컬럼이 합쳐져 있어야 한다
        self.assertIn("product_name", df.columns)

        # 주문일이 날짜형이어야 한다
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["order_date"]))

        # 돌려준 표가 객체에 저장된 표와 같아야 한다
        self.assertIs(analyzer.df, df)


# 결측치 처리 테스트 모음
class TestMissingValues(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            # 카테고리 A, B에 빈 값이 하나씩 있는 표
            {
                # 그룹 컬럼
                "category": ["A", "A", "A", "B", "B", "B"],
                "price": [
                    10.0,
                    20.0,
                    np.nan,
                    100.0,
                    300.0,
                    np.nan,
                    # 빈 값이 있는 가격
                ],
            }
        )

    # 결측치 현황표 테스트
    def test_missing_report(self) -> None:
        # 현황표를 만든다
        report = make_analyzer(self.df).missing_report()

        # price의 빈 값은 2개
        self.assertEqual(report.loc["price", "missing_count"], 2)

        # 비율은 6개 중 2개 = 33.3%
        self.assertAlmostEqual(report["missing_ratio"]["price"], 2 / 6 * 100)

    # 그룹별 중앙값 대치 테스트
    def test_group_median(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 카테고리별 중앙값으로 채운다
        analyzer.handle_missing_values(
            strategy="group_median", group_col="category", columns=["price"]
        )

        # A 그룹 중앙값 (10, 20 -> 15)
        self.assertEqual(analyzer.df.loc[2, "price"], 15.0)

        # B 그룹 중앙값 (100, 300 -> 200)
        self.assertEqual(analyzer.df.loc[5, "price"], 200.0)

        # 빈 값이 남지 않아야 한다
        self.assertEqual(analyzer.df["price"].isna().sum(), 0)

    # 그룹별 평균 대치 테스트
    def test_group_mean(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 카테고리별 평균으로 채운다
        analyzer.handle_missing_values(
            strategy="group_mean", group_col="category", columns=["price"]
        )

        # B 그룹 평균 (100, 300 -> 200)
        self.assertEqual(analyzer.df.loc[5, "price"], 200.0)

    # 전체 중앙값 대치 테스트
    def test_global_median_without_group(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 전체 중앙값으로 채운다
        analyzer.handle_missing_values(strategy="median", columns=["price"])

        # 10, 20, 100, 300의 중앙값은 60
        self.assertEqual(analyzer.df.loc[2, "price"], 60.0)

    # 빈 값이 있는 행 삭제 테스트
    def test_drop(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 빈 값이 있는 행을 지운다
        analyzer.handle_missing_values(strategy="drop", columns=["price"])

        # 6행 중 2행이 지워져 4행이 남는다
        self.assertEqual(len(analyzer.df), 4)

    # 없는 전략 이름 테스트
    def test_invalid_strategy(self) -> None:
        # 에러가 나야 통과다
        with self.assertRaises(ValueError):
            # 없는 전략을 넣는다
            make_analyzer(self.df).handle_missing_values(strategy="magic")

    # 그룹 컬럼 없이 그룹 전략을 쓰는 경우
    def test_group_strategy_requires_group_col(self) -> None:
        # 에러가 나야 통과다
        with self.assertRaises(ValueError):
            # group_col 없이 호출한다
            make_analyzer(self.df).handle_missing_values(strategy="group_mean")


# 이상치 처리 테스트 모음
class TestOutliers(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        # 1~10 과 극단값 100
        # 1부터 10까지와 튀는 값 100
        self.df = pd.DataFrame({"amount": list(range(1, 11)) + [100]})

    # IQR 정상 범위 계산 테스트
    def test_iqr_bounds(self) -> None:
        # 메서드로 범위를 구한다
        lower_bound, upper_bound = make_analyzer(self.df).get_iqr_bounds(
            "amount", threshold=1.5
        )

        # 직접 Q1, Q3를 구한다
        q1, q3 = self.df["amount"].quantile([0.25, 0.75])

        # 직접 IQR을 구한다
        iqr = q3 - q1

        # 하한이 손계산과 같아야 한다
        self.assertAlmostEqual(lower_bound, q1 - 1.5 * iqr)

        # 상한이 손계산과 같아야 한다
        self.assertAlmostEqual(upper_bound, q3 + 1.5 * iqr)

    # 이상치 찾기 테스트
    def test_detect_outliers(self) -> None:
        # 이상치 행을 찾는다
        outliers = make_analyzer(self.df).detect_outliers("amount")

        # 100만 이상치여야 한다
        self.assertEqual(outliers["amount"].tolist(), [100])

    # 임계값에 따라 결과가 달라지는지 테스트
    def test_threshold_changes_result(self) -> None:
        # 임계값을 아주 크게 주면 이상치가 없어야 한다
        # 임계값을 100으로 크게 준다
        outliers = make_analyzer(self.df).detect_outliers("amount", threshold=100)

        # 이상치가 없어야 한다
        self.assertEqual(len(outliers), 0)

    # 경계값으로 자르기 테스트
    def test_handle_outliers_clip(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 상한만 필요하다
        _, upper_bound = analyzer.get_iqr_bounds("amount")

        # 범위 밖 값을 경계값으로 자른다
        analyzer.handle_outliers("amount", method="clip")

        # 행 수는 그대로여야 한다
        self.assertEqual(len(analyzer.df), 11)

        # 최대값이 상한으로 잘려 있어야 한다
        self.assertAlmostEqual(analyzer.df["amount"].max(), upper_bound)

    # 이상치 행 삭제 테스트
    def test_handle_outliers_remove(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 이상치 행을 지운다
        analyzer.handle_outliers("amount", method="remove")

        # 11행 중 1행이 지워져 10행이다
        self.assertEqual(len(analyzer.df), 10)

        # 남은 최대값은 10이다
        self.assertEqual(analyzer.df["amount"].max(), 10)


# 이미지, 텍스트 피처 테스트 모음
class TestFeatureEngineering(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        # 항상 같은 값이 나오는 난수 생성기
        rng = np.random.default_rng(0)

        # 4x4 컬러 이미지 3장을 만든다
        self.images = rng.integers(0, 256, size=(3, 4, 4, 3)).astype(np.uint8)

        self.df = pd.DataFrame(
            # 이미지와 상품 정보가 든 표
            {
                "product_name": [
                    "Red Shirt",
                    "Wireless Gaming Mouse Pro",
                    "Tea",
                    # 상품명
                ],
                # 이미지를 문자열로 바꿔 넣는다
                "image_pixels": [image_to_str(image) for image in self.images],
                # 가격
                "price": [10.0, 20.0, 30.0],
                # 수량
                "quantity": [1, 2, 3],
            }
        )

    # 문자열을 배열로 되돌리는 테스트
    def test_parse_images_shape(self) -> None:
        # 문자열을 이미지 배열로 바꾼다
        parsed_images = make_analyzer(self.df).parse_images(
            "image_pixels", image_shape=(4, 4, 3)
        )

        # 모양이 (이미지 수, 4, 4, 3)이어야 한다
        self.assertEqual(parsed_images.shape, (3, 4, 4, 3))

        # 원래 이미지와 값이 완전히 같아야 한다
        np.testing.assert_array_equal(parsed_images, self.images)

    # 이미지 줄이기 테스트
    def test_parse_images_downsample(self) -> None:
        # 2칸마다 1칸씩 뽑는다
        parsed_images = make_analyzer(self.df).parse_images(
            "image_pixels", image_shape=(4, 4, 3), downsample_step=2
        )

        # 4x4가 2x2로 줄어야 한다
        self.assertEqual(parsed_images.shape, (3, 2, 2, 3))

        # 직접 슬라이싱한 결과와 같아야 한다
        np.testing.assert_array_equal(parsed_images, self.images[:, ::2, ::2, :])

    # 이미지 평균, 표준편차 테스트
    def test_image_mean_std(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 평균, 표준편차 컬럼을 추가한다
        analyzer.add_image_features("image_pixels", image_shape=(4, 4, 3))

        # 이미지를 하나씩 확인한다
        for row_number, image in enumerate(self.images):
            # 평균이 NumPy 계산과 같아야 한다
            self.assertAlmostEqual(
                analyzer.df.loc[row_number, "image_brightness_mean"], image.mean()
            )

            # 표준편차도 같아야 한다
            self.assertAlmostEqual(
                analyzer.df.loc[row_number, "image_brightness_std"], image.std()
            )

    # 밝기 비율, 엣지 컬럼 테스트
    def test_image_advanced_features(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 4구간으로 나눠 컬럼을 추가한다
        analyzer.add_image_advanced_features(
            "image_pixels", image_shape=(4, 4, 3), n_brightness_bins=4
        )

        # 기대하는 비율 컬럼 이름 4개
        ratio_columns = [
            f"image_brightness_bin_{bin_number}_ratio" for bin_number in range(4)
        ]

        # 비율 컬럼과 엣지 컬럼을
        for column in ratio_columns + ["image_edge_strength"]:
            # 모두 만들어졌는지 확인한다
            self.assertIn(column, analyzer.df.columns)

        # 히스토그램 비율의 합은 1
        # 구간별 비율을 더하면 1이어야 한다
        np.testing.assert_allclose(analyzer.df[ratio_columns].sum(axis=1), 1.0)

    # 단색 이미지는 엣지가 없어야 한다
    def test_edge_is_zero_for_flat_image(self) -> None:
        # 모든 픽셀이 128인 이미지 1장
        flat = np.full((1, 4, 4, 3), 128, dtype=np.uint8)

        # 그 이미지로 분석기를 만든다
        analyzer = make_analyzer(pd.DataFrame({"image_pixels": [image_to_str(flat)]}))

        # 엣지 컬럼을 추가한다
        analyzer.add_image_advanced_features("image_pixels", image_shape=(4, 4, 3))

        # 이웃 픽셀 차이가 없으니 0이어야 한다
        self.assertEqual(analyzer.df.loc[0, "image_edge_strength"], 0.0)

    # 상품명 단어 수, 글자 수 테스트
    def test_text_features(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 단어 수, 글자 수 컬럼을 추가한다
        analyzer.add_text_features("product_name")

        # 단어 수: 2, 4, 1
        self.assertEqual(analyzer.df["name_word_count"].tolist(), [2, 4, 1])

        # 글자 수(공백 포함): 9, 25, 3
        self.assertEqual(analyzer.df["name_length"].tolist(), [9, 25, 3])

    # 주문 금액 계산 테스트
    def test_add_amount(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 가격 x 수량 컬럼을 추가한다
        analyzer.add_amount("price", "quantity")

        # 10x1, 20x2, 30x3
        self.assertEqual(analyzer.df["amount"].tolist(), [10.0, 40.0, 90.0])


# 통계 테스트 모음
class TestStatistics(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        # y는 x의 2배인 표
        self.df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]}
        )

    # 요약 통계 테스트
    def test_summary_statistics(self) -> None:
        # x의 요약 통계를 구한다
        summary = make_analyzer(self.df).summary_statistics(["x"])

        # 있어야 할 통계 이름들을
        for column in ["mean", "median", "std", "q1", "q3", "iqr"]:
            # 하나씩 확인한다
            self.assertIn(column, summary.columns)

        # 1~5의 중앙값은 3
        self.assertEqual(summary.loc["x", "median"], 3.0)

        # 25% 지점은 2
        self.assertEqual(summary.loc["x", "q1"], 2.0)

        # Q3(4) - Q1(2) = 2
        self.assertEqual(summary.loc["x", "iqr"], 2.0)

    # 상관계수 테스트
    def test_correlation(self) -> None:
        # x와 y의 상관계수를 구한다
        corr = make_analyzer(self.df).correlation(["x", "y"])

        # y가 x의 2배라 완전히 같이 움직여 1이다
        self.assertAlmostEqual(corr["y"]["x"], 1.0)


# RFM 테스트 모음
class TestRFM(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            # 고객 A, B, C의 주문 6건
            {
                # 주문 번호
                "order_id": [1, 2, 3, 4, 5, 6],
                # 고객 번호
                "customer_id": ["A", "A", "A", "B", "C", "C"],
                "order_date": pd.to_datetime(
                    # 주문일
                    [
                        "2025-01-01",
                        "2025-02-01",
                        # A의 주문일 3개
                        "2025-03-01",
                        "2025-01-10",
                        "2025-02-20",
                        # B 1개, C 2개
                        "2025-02-25",
                    ]
                ),
                # 주문 금액
                "amount": [100.0, 200.0, 300.0, 50.0, 10.0, 20.0],
            }
        )

    # R, F, M 값 계산 테스트
    def test_rfm_values(self) -> None:
        # RFM을 계산한다
        rfm = make_analyzer(self.df).calculate_rfm(
            "customer_id", "order_date", "amount", order_col="order_id"
        )

        # 기준일은 마지막 거래일 2025-03-01
        # A는 기준일에 샀으니 0일
        self.assertEqual(rfm.loc["A", "recency"], 0)

        # B는 1월 10일 -> 50일 전
        self.assertEqual(rfm.loc["B", "recency"], 50)

        # C는 2월 25일 -> 4일 전
        self.assertEqual(rfm.loc["C", "recency"], 4)

        # A는 3번 샀다
        self.assertEqual(rfm.loc["A", "frequency"], 3)

        # C는 2번 샀다
        self.assertEqual(rfm.loc["C", "frequency"], 2)

        # A의 총액은 100+200+300
        self.assertEqual(rfm.loc["A", "monetary"], 600.0)

    # 기준일을 직접 정하는 테스트
    def test_reference_date_param(self) -> None:
        # 기준일을 2025-03-11로 정해서 계산한다
        rfm = make_analyzer(self.df).calculate_rfm(
            "customer_id", "order_date", "amount", reference_date="2025-03-11"
        )

        # A의 마지막 구매(3월 1일)에서 10일 지났다
        self.assertEqual(rfm.loc["A", "recency"], 10)

    # 점수 방향 테스트 (좋을수록 높은 점수)
    def test_scores_direction(self) -> None:
        # 3구간으로 점수를 매긴다
        rfm = make_analyzer(self.df).calculate_rfm(
            "customer_id", "order_date", "amount", n_score_levels=3
        )

        # 세 점수 컬럼을
        for column in ["r_score", "f_score", "m_score"]:
            # 모두 1~3 안에 있는지 확인한다
            self.assertTrue(rfm[column].between(1, 3).all())

        # 최근에 산 A가 오래전에 산 B보다 R 점수가 높다
        # 최근일수록 R 점수가 높아야 한다
        self.assertGreater(rfm["r_score"]["A"], rfm["r_score"]["B"])

        # 많이 쓸수록 M 점수가 높아야 한다
        self.assertGreater(rfm["m_score"]["A"], rfm["m_score"]["C"])

    # 세그먼트 판정 규칙 테스트
    def test_segment_rules(self) -> None:
        rfm = pd.DataFrame(
            # 점수를 직접 정한 가짜 RFM 표
            {
                # R 점수
                "r_score": [5, 4, 5, 1, 1, 3],
                # F 점수
                "f_score": [5, 3, 1, 4, 1, 2],
                # M 점수
                "m_score": [5, 3, 1, 4, 1, 3],
                # 고객 이름 대신 기대 세그먼트를 이름으로 쓴다
            },
            index=["vip", "loyal", "new", "risk", "churn", "mid"],
        )
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # 가짜 RFM 표를 직접 넣는다
        analyzer.rfm = rfm

        # 세그먼트를 나눈다
        result = analyzer.segment_customers()

        # 5, 5, 5 -> VIP
        self.assertEqual(result.loc["vip", "segment"], "VIP")

        # 4, 3, 3 -> Loyal
        self.assertEqual(result.loc["loyal", "segment"], "Loyal")

        # 5, 1, 1 -> New
        self.assertEqual(result.loc["new", "segment"], "New")

        # 1, 4, 4 -> At Risk
        self.assertEqual(result.loc["risk", "segment"], "At Risk")

        # 1, 1, 1 -> Churned
        self.assertEqual(result.loc["churn", "segment"], "Churned")

        # 3, 2, 3 -> 어디에도 안 맞아 Regular
        self.assertEqual(result.loc["mid", "segment"], "Regular")

    # RFM 계산 전에 세그먼트를 나누는 경우
    def test_segment_without_rfm_raises(self) -> None:
        # 에러가 나야 통과다
        with self.assertRaises(ValueError):
            # RFM 없이 호출한다
            make_analyzer(self.df).segment_customers()

    # 세그먼트 요약표 테스트
    def test_segment_summary(self) -> None:
        # 분석기를 만든다
        analyzer = make_analyzer(self.df)

        # RFM을 계산한다
        analyzer.calculate_rfm("customer_id", "order_date", "amount", n_score_levels=3)

        # 세그먼트를 나눈다
        analyzer.segment_customers()

        # 요약표를 만든다
        summary = analyzer.segment_summary()

        # 고객 수의 합은 3명이어야 한다
        self.assertEqual(summary["customers"].sum(), 3)

        # 매출 비중의 합은 100%여야 한다
        self.assertAlmostEqual(summary["revenue_share"].sum(), 100.0)


# 월별 매출, 코호트 테스트 모음
class TestTimeAnalysis(unittest.TestCase):
    # 테스트마다 먼저 실행되는 준비 단계
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            # 고객 A, B, C의 주문 5건
            {
                # 고객 번호
                "customer_id": ["A", "A", "B", "B", "C"],
                # 주문일
                "order_date": pd.to_datetime(
                    [
                        "2025-01-05",
                        "2025-02-10",
                        "2025-01-20",
                        "2025-03-01",
                        "2025-02-15",
                    ]
                ),
                # 주문 금액
                "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    # 월별 매출 테스트
    def test_monthly_sales(self) -> None:
        # 월별 매출을 구한다
        monthly = make_analyzer(self.df).monthly_sales("order_date", "amount")

        # 1월 10+30, 2월 20+50, 3월 40
        self.assertEqual(monthly.tolist(), [40.0, 70.0, 40.0])

    # 코호트 재구매율 테스트
    def test_cohort_retention(self) -> None:
        # 재구매율 표를 만든다
        retention = make_analyzer(self.df).cohort_retention("customer_id", "order_date")

        # 첫 달(0번째 달)의 재구매율은 항상 1
        # 0개월 차는 모두 100%여야 한다
        self.assertTrue((retention[0] == 1.0).all())

        # 1월 코호트(A, B): 1달 뒤 A만 구매 → 0.5, 2달 뒤 B만 구매 → 0.5
        # 1월에 처음 산 고객 묶음의 행
        jan = retention.xs(pd.Period("2025-01", "M"))

        # 1달 뒤에는 2명 중 1명이 샀다
        self.assertAlmostEqual(jan[1], 0.5)

        # 2달 뒤에도 2명 중 1명이 샀다
        self.assertAlmostEqual(jan[2], 0.5)


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 테스트를 실행한다
    unittest.main()
