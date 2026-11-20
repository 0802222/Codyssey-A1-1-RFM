"""이커머스 데이터 분석 파이프라인.

H&M 거래 샘플을 불러와 전처리(결측치, 이상치), 피처 엔지니어링(이미지, 텍스트),
요약 통계, RFM 고객 세분화까지 수행하는 DataAnalyzer 클래스를 제공한다.

실행: python -m src.pipeline
"""

# 정해진 문자열만 받는 타입 힌트
from typing import Literal

# 배열 계산 라이브러리
import numpy as np

# 표(DataFrame) 처리 라이브러리
import pandas as pd

# 공통 상수: 데이터 파일 경로, 세그먼트 표시 순서
from src.constants import (
    ARTICLES_PATH,
    IMAGES_PATH,
    SEGMENT_ORDER,
    TRANSACTIONS_PATH,
)

# IQR 정상 범위를 정하는 기본 배수
IQR_THRESHOLD = 1.5

# 작은 쪽에서 25% 지점
Q1_QUANTILE = 0.25

# 작은 쪽에서 75% 지점
Q3_QUANTILE = 0.75

# RFM 점수 단계 수 (1~5점)
RFM_SCORE_LEVELS = 5

# 이미지 밝기를 나누는 기본 구간 수
BRIGHTNESS_BINS = 4

# H&M 원본 컬럼 이름을 읽기 쉬운 이름으로 바꾸는 규칙 (원본 이름: 새 이름)
COLUMN_NAMES = {"t_dat": "order_date", "prod_name": "product_name"}


class DataAnalyzer:
    """이커머스 거래/상품/이미지 데이터를 분석하는 클래스."""

    # 객체를 만들 때 한 번 실행된다
    # 분석할 거래/상품 CSV와 이미지 npz 경로를 저장한다.
    def __init__(
        self,
        data_path: str,
        product_path: str | None = None,
        image_path: str | None = None,
    ) -> None:
        # 거래(구매) 데이터 경로
        self.data_path = data_path

        # 상품 데이터 경로 (없으면 거래 데이터만 사용)
        self.product_path = product_path

        # 이미지 배열(npz) 경로 (없으면 이미지 피처를 만들 수 없다)
        self.image_path = image_path

        # 분석 대상 표. load_data()를 하기 전에는 빈 표이다
        self.df: pd.DataFrame = pd.DataFrame()

        # load_images()로 읽은 상품 번호. 읽기 전에는 None이다
        self.image_article_ids: np.ndarray | None = None

        # load_images()로 읽은 이미지 배열 (상품 수, 세로, 가로, 색)
        # 읽기 전에는 None이다
        self.images: np.ndarray | None = None

        # calculate_rfm() 결과. 계산 전에는 None이다
        self.rfm: pd.DataFrame | None = None

    # 1. 데이터 로드
    #
    # - CSV를 읽어 거래 데이터를 self.df에 담는다.
    # - 상품 번호와 고객 번호는 앞자리 0이 사라지지 않게 문자열로 읽는다.
    # - column_names 규칙대로 컬럼 이름을 바꾼다. 없으면 H&M 기본 규칙을 쓴다.
    # - 수량(quantity) 컬럼이 없으면 한 행이 상품 1개이므로 1로 채운다.
    # - 상품 데이터가 있으면 merge_key로 합친다.
    def load_data(
        self,
        date_cols: list[str] | None = None,
        merge_key: str = "article_id",
        column_names: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        # 이름 바꾸기 규칙을 안 정했으면 기본 규칙을 쓴다
        rename_rules = COLUMN_NAMES if column_names is None else column_names

        # 번호 컬럼은 앞자리 0을 살리려고 문자열로 읽는다
        string_columns = {merge_key: "string", "customer_id": "string"}

        # 거래 CSV를 표로 읽고 컬럼 이름을 바꾼다
        df = pd.read_csv(self.data_path, dtype=string_columns).rename(
            columns=rename_rules
        )

        # 상품 파일 경로가 있으면
        if self.product_path:
            # 상품 CSV도 읽고 컬럼 이름을 바꾼다
            products = pd.read_csv(
                self.product_path, dtype={merge_key: "string"}
            ).rename(columns=rename_rules)

            # 상품 번호가 같은 행끼리 옆으로 붙인다
            df = df.merge(products, on=merge_key, how="left")

        # 수량 컬럼이 없으면
        if "quantity" not in df.columns:
            # 한 행이 상품 1개이므로 모두 1로 채운다
            df["quantity"] = 1

        # 데이터 전처리 (데이터 타입 변환 : 문자열 -> 날짜형)
        for date_col in date_cols or []:
            # 문자열을 날짜형으로 바꾼다
            df[date_col] = pd.to_datetime(df[date_col])

        # 완성한 표를 객체에 저장한다
        self.df = df

        # 저장한 표를 돌려준다
        return self.df

    # 같은 고객이 같은 날 산 행을 하나의 주문으로 묶는 키를 만든다
    # 주문 번호가 없는 데이터에서 Frequency(주문 횟수)를 세기 위해 쓴다.
    # 주문 키 = 고객 번호 + "_" + 거래일(YYYY-MM-DD)
    def add_order_key(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        output_col: str = "order_key",
    ) -> pd.DataFrame:
        # 거래일을 "2020-01-01" 모양의 글자로 바꾼다
        date_texts = pd.to_datetime(self.df[date_col]).dt.strftime("%Y-%m-%d")

        # 고객 번호와 거래일 글자를 이어 붙여 주문 키를 만든다
        self.df[output_col] = self.df[customer_col].astype(str) + "_" + date_texts

        # 컬럼을 추가한 표를 돌려준다
        return self.df

    # 2. 결측치 처리
    #
    # - 컬럼마다 빈 값이 얼마나 되는지 보여준다.
    # - 컬럼별 결측치 개수와 비율(%)을 돌려준다.
    def missing_report(self) -> pd.DataFrame:
        # 컬럼별 빈 값 개수를 센다
        missing_counts = self.df.isna().sum()

        # 개수와 비율(%)을 표로 만든다
        report = pd.DataFrame(
            {
                "missing_count": missing_counts,
                "missing_ratio": missing_counts / len(self.df) * 100,
            }
        )

        # 빈 값이 있는 컬럼만 많은 순으로 돌려준다
        return report[report["missing_count"] > 0].sort_values(
            "missing_count", ascending=False
        )

    # 빈 값을 채우거나 지운다
    # 결측치를 채운다.
    #
    # strategy
    #     group_mean, group_median: group_col 그룹별 평균/중앙값으로 채운다.
    #         그룹 전체가 비어 있으면 전체 평균/중앙값으로 채운다.
    #     mean, median, mode: 전체 통계로 채운다.
    #     drop: 결측치가 있는 행을 지운다.
    # columns: 처리할 컬럼 목록. 없으면 결측치가 있는 수치형 컬럼 전체.
    def handle_missing_values(
        self,
        strategy: str = "group_median",
        group_col: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        # 쓸 수 있는 전략 목록
        valid_strategies = {
            "group_mean",
            "group_median",
            "mean",
            "median",
            "mode",
            "drop",
        }

        # 목록에 없는 전략이면
        if strategy not in valid_strategies:
            # 에러로 알려준다
            raise ValueError(
                f"strategy는 {sorted(valid_strategies)} 중 하나여야 한다: {strategy}"
            )

        # 그룹 전략인데 그룹 컬럼이 없으면
        if strategy.startswith("group_") and group_col is None:
            # 에러로 알려준다
            raise ValueError("group 전략은 group_col이 필요하다")

        # 처리할 컬럼을 안 정했으면
        if columns is None:
            # 숫자 컬럼만 고른다
            numeric_columns = self.df.select_dtypes(include="number").columns

            # 그중 빈 값이 있는 컬럼만 남긴다
            columns = [name for name in numeric_columns if self.df[name].isna().any()]

        # 지우는 전략이면
        if strategy == "drop":
            # 빈 값이 있는 행을 지우고 번호를 다시 매긴다
            self.df = self.df.dropna(subset=columns).reset_index(drop=True)

            # 바로 끝낸다
            return self.df

        # 컬럼을 하나씩 처리한다
        for column in columns:
            # 그룹별로 채우는 전략이면
            if strategy.startswith("group_"):
                # "mean" 또는 "median"만 남긴다
                statistic_name = strategy.replace("group_", "")

                # 그룹별로 해당 컬럼을 묶는다
                grouped = self.df.groupby(group_col)[column]

                # 평균으로 채우는 경우
                if statistic_name == "mean":
                    # 행마다 자기 그룹의 평균을 붙인다
                    group_value = grouped.transform("mean")
                # 중앙값으로 채우는 경우
                else:
                    # 행마다 자기 그룹의 중앙값을 붙인다
                    group_value = grouped.transform("median")

                # 빈 칸을 자기 그룹 값으로 채운다
                self.df[column] = self.df[column].fillna(group_value)

                # 그룹 전체가 비었을 때 쓸 전체 통계
                fallback_value = self.df[column].agg(statistic_name)
            # 최빈값(가장 자주 나온 값)으로 채우는 전략이면
            elif strategy == "mode":
                # 가장 자주 나온 값을 고른다
                fallback_value = self.df[column].mode().iloc[0]
            # mean, median 전략이면
            else:
                # 전체 평균 또는 중앙값을 구한다
                fallback_value = self.df[column].agg(strategy)

            # 아직 남은 빈 칸을 마저 채운다
            self.df[column] = self.df[column].fillna(fallback_value)

        # 채운 표를 돌려준다
        return self.df

    # 3. 이상치 처리 (IQR)

    # IQR로 정상 범위를 계산한다
    # IQR 방식의 정상 범위 (하한, 상한)를 계산한다.
    def get_iqr_bounds(
        self, column: str, threshold: float = IQR_THRESHOLD
    ) -> tuple[float, float]:
        # 작은 쪽에서 25% 지점의 값
        q1 = self.df[column].quantile(Q1_QUANTILE)

        # 작은 쪽에서 75% 지점의 값
        q3 = self.df[column].quantile(Q3_QUANTILE)

        # 가운데 50%가 퍼진 폭
        iqr = q3 - q1

        # (하한, 상한)을 일반 소수로 바꿔 돌려준다
        return float(q1 - threshold * iqr), float(q3 + threshold * iqr)

    # 정상 범위 밖 행을 찾는다
    # 정상 범위를 벗어난 행을 돌려준다.
    def detect_outliers(
        self, column: str, threshold: float = IQR_THRESHOLD
    ) -> pd.DataFrame:
        # 하한과 상한을 구한다
        lower_bound, upper_bound = self.get_iqr_bounds(column, threshold)

        # 하한보다 작거나 상한보다 큰 행만 돌려준다
        is_outlier = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

        return self.df[is_outlier]

    # 이상치를 자르거나 지운다
    # 이상치를 처리한다.
    #
    # method
    #     clip: 범위 밖 값을 하한/상한 값으로 바꾼다. 행 수는 그대로다.
    #     remove: 범위 밖 행을 지운다.
    def handle_outliers(
        self, column: str, threshold: float = IQR_THRESHOLD, method: str = "clip"
    ) -> pd.DataFrame:
        # 하한과 상한을 구한다
        lower_bound, upper_bound = self.get_iqr_bounds(column, threshold)

        # 경계값으로 자르는 방법이면
        if method == "clip":
            # 범위 밖 값을 하한/상한으로 바꾼다
            self.df[column] = self.df[column].clip(lower_bound, upper_bound)
        # 지우는 방법이면
        elif method == "remove":
            # 정상 범위 안이면 True인 표시를 만든다
            is_in_range = self.df[column].between(lower_bound, upper_bound)

            # 정상 행만 남기고 번호를 다시 매긴다
            self.df = self.df[is_in_range].reset_index(drop=True)
        # 그 밖의 값이면
        else:
            # 에러로 알려준다
            raise ValueError(f"method는 clip 또는 remove여야 한다: {method}")

        # 처리한 표를 돌려준다
        return self.df

    # 4. 피처 엔지니어링

    # 이미지 배열 파일(npz)을 읽는다
    # 상품 번호 배열(article_ids)과 이미지 배열(images)을 읽어 객체에 저장한다.
    #
    # images 모양은 (상품 수, 세로, 가로, 색)이고, 상품 번호와 같은 순서다.
    def load_images(
        self, image_path: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # 경로를 안 주면 객체를 만들 때 정한 경로를 쓴다
        path = image_path or self.image_path

        # 경로가 어디에도 없으면
        if path is None:
            # 에러로 알려준다
            raise ValueError("이미지 파일 경로가 필요하다")

        # npz 파일을 열어 두 배열을 메모리로 꺼낸다
        with np.load(path) as image_file:
            # 상품 번호 배열
            article_ids = image_file["article_ids"]

            # 이미지 배열
            images = image_file["images"]

        # 읽은 값을 객체에 저장한다
        self.image_article_ids = article_ids
        self.images = images

        # 읽은 값을 돌려준다
        return article_ids, images

    # 이미지가 읽혔는지 확인하고 (상품 번호, 이미지)를 돌려준다
    def _require_images(self) -> tuple[np.ndarray, np.ndarray]:
        # 아직 안 읽었으면
        if self.images is None or self.image_article_ids is None:
            # 에러로 알려준다
            raise ValueError("load_images()를 먼저 실행해야 한다")

        # 읽어 둔 값을 돌려준다
        return self.image_article_ids, self.images

    # 상품별 피처를 거래 표의 각 행에 붙인다
    # 상품 번호가 같은 행에는 같은 값이 붙고, 이미지가 없는 상품은 NaN이 된다.
    def _attach_article_features(self, feature_table: pd.DataFrame) -> None:
        # 거래 표의 상품 번호를 글자로 맞춘다
        article_ids = self.df["article_id"].astype(str)

        # 피처 컬럼을 하나씩
        for column in feature_table.columns:
            # 상품 번호로 찾아 거래 표에 붙인다
            self.df[column] = article_ids.map(feature_table[column])

    # 이미지 평균과 표준편차 컬럼을 만든다
    # 이미지별 밝기 평균과 표준편차 컬럼을 추가한다.
    #
    # 컬럼 이름은 image_brightness_mean, image_brightness_std이다.
    # 이미지 전체를 한 번에 계산하므로 반복문을 쓰지 않는다.
    def add_image_features(self) -> pd.DataFrame:
        # 읽어 둔 상품 번호와 이미지를 가져온다
        article_ids, images = self._require_images()

        # 이미지를 소수 배열로 바꾼다
        float_images = images.astype(np.float64)

        # 이미지 번호(0번 축)를 뺀 나머지 축 전체
        pixel_axes = tuple(range(1, float_images.ndim))

        feature_table = pd.DataFrame(
            # 상품 번호를 행 이름으로 하는 피처 표
            {
                # 이미지마다 평균 밝기
                "image_brightness_mean": float_images.mean(axis=pixel_axes),
                # 이미지마다 밝기가 퍼진 정도
                "image_brightness_std": float_images.std(axis=pixel_axes),
            },
            index=article_ids,
        )

        # 거래 표에 붙인다
        self._attach_article_features(feature_table)

        # 컬럼을 추가한 표를 돌려준다
        return self.df

    # 밝기 비율과 엣지 컬럼을 만든다
    # 밝기 구간별 픽셀 비율(image_brightness_bin_N_ratio)과 엣지 강도를 추가한다.
    #
    # 엣지 강도는 image_edge_strength 컬럼이다.
    def add_image_advanced_features(
        self, n_brightness_bins: int = BRIGHTNESS_BINS
    ) -> pd.DataFrame:
        # 읽어 둔 상품 번호와 이미지를 가져온다
        article_ids, images = self._require_images()

        # 이미지를 소수 배열로 바꾼다
        float_images = images.astype(np.float64)

        # (N, H, W) 흑백 이미지
        gray_images = (
            float_images.mean(axis=3) if float_images.ndim == 4 else float_images
        )

        # 밝기 구간 번호를 구한 뒤, 구간마다 픽셀 비율을 센다
        # 픽셀마다 밝기 구간 번호(0 ~ n_brightness_bins-1)를 정한다
        bin_numbers = np.clip(
            (gray_images / 256 * n_brightness_bins).astype(int),
            0,
            n_brightness_bins - 1,
        )

        # 이미지마다 구간별 픽셀 비율을 구한다
        bin_ratios = (
            bin_numbers[..., np.newaxis] == np.arange(n_brightness_bins)
        ).mean(axis=(1, 2))

        # 구간별 비율 컬럼을 담을 딕셔너리
        feature_columns: dict[str, np.ndarray] = {}

        # 구간을 하나씩 돌며
        for bin_number in range(n_brightness_bins):
            # 구간별 비율을 컬럼으로 넣는다
            feature_columns[f"image_brightness_bin_{bin_number}_ratio"] = bin_ratios[
                :, bin_number
            ]

        # 이웃 픽셀 간 밝기 차이의 평균 = 엣지 강도
        # 가로 이웃 픽셀과의 밝기 차이 평균
        horizontal_diff = np.abs(np.diff(gray_images, axis=2)).mean(axis=(1, 2))

        # 세로 이웃 픽셀과의 밝기 차이 평균
        vertical_diff = np.abs(np.diff(gray_images, axis=1)).mean(axis=(1, 2))

        # 가로, 세로 평균을 엣지 강도로 쓴다
        feature_columns["image_edge_strength"] = (horizontal_diff + vertical_diff) / 2

        # 상품 번호를 행 이름으로 하는 피처 표를 만든다
        feature_table = pd.DataFrame(feature_columns, index=article_ids)

        # 거래 표에 붙인다
        self._attach_article_features(feature_table)

        # 컬럼을 추가한 표를 돌려준다
        return self.df

    # 상품명의 단어 수와 글자 수 컬럼을 만든다
    # 상품명의 단어 수(name_word_count)와 글자 수(name_length) 컬럼을 추가한다.
    def add_text_features(self, text_col: str = "product_name") -> pd.DataFrame:
        # 빈 값은 빈 글자로 바꾸고 문자열로 맞춘다
        texts = self.df[text_col].fillna("").astype(str)

        # 공백으로 자른 조각 수 = 단어 수
        # (pandas-stubs가 split 결과의 len 타입을 못 읽어 경고를 무시한다)
        self.df["name_word_count"] = texts.str.split().str.len()  # type: ignore[misc]

        # 공백을 포함한 글자 수
        self.df["name_length"] = texts.str.len()

        # 컬럼을 추가한 표를 돌려준다
        return self.df

    # 주문 금액 컬럼을 만든다
    # 주문 금액 = 가격 x 수량 컬럼을 추가한다.
    def add_amount(
        self,
        price_col: str = "price",
        quantity_col: str = "quantity",
        output_col: str = "amount",
    ) -> pd.DataFrame:
        # 가격과 수량을 곱해 주문 금액을 만든다
        self.df[output_col] = self.df[price_col] * self.df[quantity_col]

        # 컬럼을 추가한 표를 돌려준다
        return self.df

    # 5. 통계 분석

    # 컬럼별 요약 통계표를 만든다
    # 평균, 중앙값, 표준편차, 사분위수, IQR을 표로 돌려준다.
    def summary_statistics(self, columns: list[str]) -> pd.DataFrame:
        # 요약할 컬럼만 고른다
        selected = self.df[columns]

        summary = pd.DataFrame(
            # 통계를 컬럼으로 하는 표를 만든다
            {
                # 평균
                "mean": selected.mean(),
                # 중앙값
                "median": selected.median(),
                # 표준편차
                "std": selected.std(),
                # 최소값
                "min": selected.min(),
                # 25% 지점
                "q1": selected.quantile(Q1_QUANTILE),
                # 75% 지점
                "q3": selected.quantile(Q3_QUANTILE),
                # 최대값
                "max": selected.max(),
            }
        )
        # IQR = Q3 - Q1
        summary["iqr"] = summary["q3"] - summary["q1"]

        # 통계표를 돌려준다
        return summary

    # 상관계수 표를 만든다
    # 상관계수 행렬을 돌려준다.
    def correlation(
        self,
        columns: list[str],
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
    ) -> pd.DataFrame:
        # 컬럼끼리 서로 얼마나 함께 움직이는지 계산한다
        return self.df[columns].corr(method=method)

    # 6. RFM 고객 세분화

    # 고객별 R, F, M을 계산한다
    # 고객별 Recency, Frequency, Monetary 값과 1~n_score_levels 점수를 계산한다.
    #
    # reference_date: Recency 기준일. 없으면 데이터의 마지막 거래일.
    # order_col: 주문 번호 컬럼. 있으면 고유 주문 수, 없으면 행 수를 Frequency로 쓴다.
    def calculate_rfm(
        self,
        customer_col: str,
        date_col: str,
        amount_col: str,
        reference_date: str | pd.Timestamp | None = None,
        order_col: str | None = None,
        n_score_levels: int = RFM_SCORE_LEVELS,
    ) -> pd.DataFrame:
        # 기준일을 안 정했으면
        if reference_date is None:
            # 데이터의 마지막 거래일을 기준일로 쓴다
            reference_date = pd.Timestamp(self.df[date_col].max())
        else:
            # 정해 준 기준일을 날짜형으로 맞춘다
            reference_date = pd.Timestamp(reference_date)

        # 고객별로 묶는다
        grouped = self.df.groupby(customer_col)

        rfm = pd.DataFrame(
            # 고객 한 명이 한 행인 표를 만든다
            {
                # 기준일 - 마지막 구매일 (일 수)
                "recency": (reference_date - grouped[date_col].max()).dt.days,
                # 주문 횟수 (주문 번호가 있으면 중복 없이 센다)
                "frequency": (
                    grouped[order_col].nunique() if order_col else grouped.size()
                ),
                # 총 구매 금액
                "monetary": grouped[amount_col].sum(),
            }
        )

        # 순위를 먼저 매겨 동점을 없앤 뒤 같은 인원 수로 n_score_levels 구간을 나눈다
        # Recency는 작을수록 좋으므로 음수로 바꿔 순위를 매긴다
        # 점수 이름: 1, 2, ..., n_score_levels
        score_labels = list(range(1, n_score_levels + 1))

        # 최근일수록 높은 점수
        rfm["r_score"] = pd.qcut(
            (-rfm["recency"]).rank(method="first"),
            n_score_levels,
            labels=score_labels,
        ).astype(int)

        # 자주 샀을수록 높은 점수
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            n_score_levels,
            labels=score_labels,
        ).astype(int)

        # 많이 썼을수록 높은 점수
        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"),
            n_score_levels,
            labels=score_labels,
        ).astype(int)

        # 세 점수의 합
        rfm["rfm_total"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)

        # 결과를 객체에 저장한다
        self.rfm = rfm

        # 저장한 표를 돌려준다
        return self.rfm

    # 점수로 고객을 세그먼트로 나눈다
    # RFM 점수로 고객을 6개 세그먼트로 나눈다.
    #
    # 아래 규칙은 위에서부터 먼저 맞는 것이 적용된다.
    #
    # VIP      R>=4, F>=4, M>=4  최근에, 자주, 많이 산다
    # Loyal    R>=3, F>=3        꾸준히 산다
    # New      R>=4, F<=2        최근에 처음 샀다
    # At Risk  R<=2, F>=3        자주 샀지만 최근에 안 온다
    # Churned  R<=2, F<=2        오래전에 조금 사고 떠났다
    # Regular  나머지
    def segment_customers(self) -> pd.DataFrame:
        # RFM을 아직 계산하지 않았으면
        if self.rfm is None:
            # 에러로 알려준다
            raise ValueError("calculate_rfm()을 먼저 실행해야 한다")

        # 세 점수 컬럼을 꺼낸다
        # 최근성 점수
        r_scores = self.rfm["r_score"]

        # 빈도 점수
        f_scores = self.rfm["f_score"]

        # 금액 점수
        m_scores = self.rfm["m_score"]

        # 위에서부터 먼저 맞는 조건이 이긴다
        segment_conditions = [
            # VIP
            (r_scores >= 4) & (f_scores >= 4) & (m_scores >= 4),
            # Loyal
            (r_scores >= 3) & (f_scores >= 3),
            # New
            (r_scores >= 4) & (f_scores <= 2),
            # At Risk
            (r_scores <= 2) & (f_scores >= 3),
            # Churned
            (r_scores <= 2) & (f_scores <= 2),
        ]

        # 조건 순서와 같은 세그먼트 이름
        segment_names = ["VIP", "Loyal", "New", "At Risk", "Churned"]

        # 처음 맞는 조건의 이름을 고르고, 없으면 Regular
        self.rfm["segment"] = np.select(
            segment_conditions, segment_names, default="Regular"
        )

        # 세그먼트 컬럼이 추가된 표를 돌려준다
        return self.rfm

    # 세그먼트별 요약표를 만든다
    # 세그먼트별 고객 수, 평균 R/F/M, 매출 비중을 정리한다.
    def segment_summary(self) -> pd.DataFrame:
        # 세그먼트를 아직 안 나눴으면
        if self.rfm is None or "segment" not in self.rfm:
            # 에러로 알려준다
            raise ValueError("segment_customers()를 먼저 실행해야 한다")

        # 세그먼트별로 묶어 통계를 낸다
        summary = self.rfm.groupby("segment").agg(
            # 고객 수
            customers=("recency", "size"),
            # 평균 Recency
            avg_recency=("recency", "mean"),
            # 평균 주문 횟수
            avg_frequency=("frequency", "mean"),
            # 평균 구매 금액
            avg_monetary=("monetary", "mean"),
            # 세그먼트 전체 매출
            total_revenue=("monetary", "sum"),
        )

        # 전체 고객 중 비중(%)
        summary["customer_share"] = (
            summary["customers"] / summary["customers"].sum() * 100
        )

        # 전체 매출 중 비중(%)
        summary["revenue_share"] = (
            summary["total_revenue"] / summary["total_revenue"].sum() * 100
        )

        # 정해 둔 순서대로, 실제로 있는 세그먼트만 고른다
        ordered_segments = [name for name in SEGMENT_ORDER if name in summary.index]

        # 그 순서로 정렬해 돌려준다
        return summary.loc[ordered_segments]

    # 7. 시계열, 코호트

    # 월별 매출을 구한다
    # 월별 매출 합계를 돌려준다.
    def monthly_sales(
        self, date_col: str = "order_date", amount_col: str = "amount"
    ) -> pd.Series:
        # 주문일을 월 단위로 묶어 금액을 더한다
        return self.df.groupby(self.df[date_col].dt.to_period("M"))[amount_col].sum()

    # 코호트별 재구매율을 구한다
    # 첫 구매 월(코호트)별로 N개월 뒤에도 구매한 고객 비율을 계산한다.
    def cohort_retention(
        self, customer_col: str = "customer_id", date_col: str = "order_date"
    ) -> pd.DataFrame:
        # 필요한 두 컬럼만 복사한다
        orders = self.df[[customer_col, date_col]].copy()

        # 주문일을 월 단위로 바꾼다
        orders["order_month"] = orders[date_col].dt.to_period("M")

        # 고객마다 가장 이른 월 = 첫 구매 월
        orders["cohort"] = orders.groupby(customer_col)["order_month"].transform("min")

        # 연*12+월 로 바꿔 빼면 몇 개월 차이인지 반복문 없이 구할 수 있다
        # 주문 월을 하나의 숫자로 바꾼다
        order_month_number = orders[date_col].dt.year * 12 + orders[date_col].dt.month

        # 첫 구매 월도 같은 방식으로 바꾼다
        cohort_month_number = orders["cohort"].dt.year * 12 + orders["cohort"].dt.month

        # 첫 구매 후 몇 개월 지났는지
        orders["months_since_first_order"] = order_month_number - cohort_month_number

        # 코호트(행) x 경과 개월(열)별 고객 수
        cohort_customer_counts = (
            orders.groupby(["cohort", "months_since_first_order"])[customer_col]
            .nunique()
            .unstack()
        )

        # 0개월 차 고객 수로 나눠 비율로 바꾼다
        return cohort_customer_counts.div(cohort_customer_counts[0], axis=0)


# 기본 데이터로 전체 흐름을 실행한다
# 기본 데이터로 전체 파이프라인을 실행하고 결과를 출력한다.
def main() -> None:
    # 출력 한 줄의 너비
    pd.set_option("display.width", 160)

    # 컬럼을 줄이지 않고 모두 보여준다
    pd.set_option("display.max_columns", None)

    # 소수는 천 단위 쉼표와 소수 4자리로 보여준다
    pd.set_option("display.float_format", "{:,.4f}".format)

    # 분석기를 만든다
    analyzer = DataAnalyzer(
        TRANSACTIONS_PATH, product_path=ARTICLES_PATH, image_path=IMAGES_PATH
    )

    # 거래와 상품을 합쳐 읽고 거래일을 날짜형으로 바꾼다
    df = analyzer.load_data(date_cols=["order_date"])

    # 표의 크기를 출력한다
    print(f"[1] 데이터 로드: {df.shape[0]:,}행 x {df.shape[1]}열")

    # 고객 수와 기간을 출력한다
    print(
        f"    고객 {df['customer_id'].nunique():,}명, "
        f"기간 {df['order_date'].min().date()} ~ {df['order_date'].max().date()}\n"
    )

    # 구매 금액 컬럼을 만든다
    analyzer.add_amount("price", "quantity")

    # 이미지 배열을 읽는다
    _, images = analyzer.load_images()

    # 이미지 배열의 모양을 출력한다
    print(f"[2] 이미지 로드: 배열 모양 {images.shape}, 자료형 {images.dtype}")

    # 이미지 평균, 표준편차 컬럼을 만든다
    analyzer.add_image_features()

    # 밝기 비율, 엣지 컬럼을 만든다
    analyzer.add_image_advanced_features(n_brightness_bins=BRIGHTNESS_BINS)

    # 상품명 단어 수, 글자 수 컬럼을 만든다
    analyzer.add_text_features("product_name")

    # 피처 엔지니어링 결과를 알려준다
    print(
        "    피처 추가: image_brightness_mean, image_brightness_std, "
        "image_brightness_bin_0~3_ratio, image_edge_strength, "
        "name_word_count, name_length\n"
    )

    # 결측치 단계 제목
    print("[3] 결측치 현황")

    # 컬럼별 빈 값 현황을 출력한다
    print(analyzer.missing_report(), "\n")

    # 이미지 피처 컬럼 이름 목록
    image_feature_columns = [
        "image_brightness_mean",
        "image_brightness_std",
        "image_edge_strength",
    ]

    # 상품 종류별 중앙값으로 이미지 피처의 빈 값을 채운다
    analyzer.handle_missing_values(
        strategy="group_median",
        group_col="product_type_name",
        columns=image_feature_columns,
    )

    # 남은 빈 값 수를 센다
    remaining_missing = int(analyzer.df[image_feature_columns].isna().sum().sum())

    # 남은 빈 값 수를 출력한다
    print(
        f"    product_type_name별 중앙값 대치 후 남은 결측치: {remaining_missing}개\n"
    )

    # 구매 금액의 정상 범위를 구한다
    lower_bound, upper_bound = analyzer.get_iqr_bounds(
        "amount", threshold=IQR_THRESHOLD
    )

    # 범위 밖 이상치 행을 찾는다
    outliers = analyzer.detect_outliers("amount", threshold=IQR_THRESHOLD)

    # 범위와 이상치 수를 출력한다
    print(
        f"[4] amount IQR 정상 범위: {lower_bound:.4f} ~ {upper_bound:.4f}, "
        f"이상치 {len(outliers):,}건 ({len(outliers) / len(df) * 100:.1f}%)"
    )

    # 이상치를 지우지 않는 이유를 알려준다
    print(
        "    RFM 매출 합계를 보존하기 위해 원본 amount는 유지하고, "
        "통계용 amount_clipped 컬럼을 따로 만든다\n"
    )

    # 이상치를 경계값으로 자른 컬럼을 따로 만든다
    analyzer.df["amount_clipped"] = analyzer.df["amount"].clip(lower_bound, upper_bound)

    # 통계 단계 제목
    print("[5] 요약 통계")

    # 통계를 낼 컬럼 목록
    statistic_columns = ["price", "amount_clipped"] + image_feature_columns

    # 요약 통계표를 출력한다
    print(analyzer.summary_statistics(statistic_columns), "\n")

    # 상관계수 제목
    print("    상관계수")

    # 상관계수 표를 출력한다
    print(
        analyzer.correlation(
            ["price", "image_brightness_mean", "image_brightness_std", "name_length"]
        ),
        "\n",
    )

    # 같은 고객이 같은 날 산 행을 하나의 주문으로 묶는 키를 만든다
    analyzer.add_order_key()

    # 고객별 R, F, M 점수를 계산한다
    analyzer.calculate_rfm("customer_id", "order_date", "amount", order_col="order_key")

    # 점수로 세그먼트를 나눈다
    analyzer.segment_customers()

    # 기준일과 함께 제목을 출력한다
    print(f"[6] RFM 세분화 (기준일: {analyzer.df['order_date'].max().date()})")

    # 세그먼트별 요약표를 출력한다
    print(analyzer.segment_summary(), "\n")

    # 월별 매출 제목
    print("[7] 월별 매출 (최근 6개월)")

    # 마지막 6개월 매출을 출력한다
    print(analyzer.monthly_sales("order_date", "amount").tail(6))


# 이 파일을 직접 실행했을 때만
if __name__ == "__main__":
    # 전체 흐름을 실행한다
    main()
