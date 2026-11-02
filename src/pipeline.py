"""이커머스 데이터 분석 파이프라인.

원본 CSV를 불러와 전처리(결측치, 이상치), 피처 엔지니어링(이미지, 텍스트),
요약 통계, RFM 고객 세분화까지 수행하는 DataAnalyzer 클래스를 제공한다.

실행: python -m src.pipeline
"""

import os  # 파일 경로를 다루는 기본 모듈

import numpy as np  # 배열 계산 라이브러리
import pandas as pd  # 표(DataFrame) 처리 라이브러리

# RFM 세그먼트 표시 순서
SEGMENT_ORDER = ["VIP", "Loyal", "New", "Regular", "At Risk", "Churned"]  # 결과표를 이 순서로 보여준다


class DataAnalyzer:
    """이커머스 거래/상품 데이터를 분석하는 클래스."""

    def __init__(self, data_path, product_path=None):  # 객체를 만들 때 한 번 실행된다
        self.data_path = data_path  # 거래(주문) 데이터 경로
        self.product_path = product_path  # 상품 데이터 경로 (없으면 거래 데이터만 사용)
        self.df: pd.DataFrame = None  # type: ignore  # 분석 대상 표. load_data()를 하기 전에는 비어 있다
        self.rfm: pd.DataFrame | None = None  # calculate_rfm() 결과. 계산 전에는 None이다

    # 1. 데이터 로드

    def load_data(self, date_cols=None, merge_key="product_id"):  # CSV를 읽어 self.df에 담는다
        """CSV를 읽는다. 상품 데이터가 있으면 merge_key로 합친다."""
        df = pd.read_csv(self.data_path)  # 주문 CSV를 표로 읽는다
        if self.product_path:  # 상품 파일 경로가 있으면
            products = pd.read_csv(self.product_path)  # 상품 CSV도 읽는다
            df = df.merge(products, on=merge_key, how="left")  # 상품 번호가 같은 행끼리 옆으로 붙인다
        for col in date_cols or []:  # 날짜로 바꿀 컬럼을 하나씩 꺼낸다
            df[col] = pd.to_datetime(df[col])  # 문자열을 날짜형으로 바꾼다
        self.df = df  # 완성한 표를 객체에 저장한다
        return self.df  # 저장한 표를 돌려준다

    # 2. 결측치 처리

    def missing_report(self):  # 컬럼마다 빈 값이 얼마나 되는지 보여준다
        """컬럼별 결측치 개수와 비율(%)을 돌려준다."""
        count = self.df.isna().sum()  # 컬럼별 빈 값 개수를 센다
        report = pd.DataFrame({"missing_count": count, "missing_ratio": count / len(self.df) * 100})  # 개수와 비율(%)을 표로 만든다
        return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)  # 빈 값이 있는 컬럼만 많은 순으로 돌려준다

    def handle_missing_values(self, strategy="group_median", group_col=None, columns=None):  # 빈 값을 채우거나 지운다
        """결측치를 채운다.

        strategy
            group_mean, group_median: group_col 그룹별 평균/중앙값으로 채운다.
                그룹 전체가 비어 있으면 전체 평균/중앙값으로 채운다.
            mean, median, mode: 전체 통계로 채운다.
            drop: 결측치가 있는 행을 지운다.
        columns: 처리할 컬럼 목록. 없으면 결측치가 있는 수치형 컬럼 전체.
        """
        valid = {"group_mean", "group_median", "mean", "median", "mode", "drop"}  # 쓸 수 있는 전략 목록
        if strategy not in valid:  # 목록에 없는 전략이면
            raise ValueError(f"strategy는 {sorted(valid)} 중 하나여야 한다: {strategy}")  # 에러로 알려준다
        if strategy.startswith("group_") and group_col is None:  # 그룹 전략인데 그룹 컬럼이 없으면
            raise ValueError("group 전략은 group_col이 필요하다")  # 에러로 알려준다

        if columns is None:  # 처리할 컬럼을 안 정했으면
            numeric = self.df.select_dtypes(include="number").columns  # 숫자 컬럼만 고른다
            columns = [c for c in numeric if self.df[c].isna().any()]  # 그중 빈 값이 있는 컬럼만 남긴다

        if strategy == "drop":  # 지우는 전략이면
            self.df = self.df.dropna(subset=columns).reset_index(drop=True)  # 빈 값이 있는 행을 지우고 번호를 다시 매긴다
            return self.df  # 바로 끝낸다

        for col in columns:  # 컬럼을 하나씩 처리한다
            if strategy.startswith("group_"):  # 그룹별로 채우는 전략이면
                agg = strategy.replace("group_", "")  # "mean" 또는 "median"만 남긴다
                grouped = self.df.groupby(group_col)[col]  # 그룹별로 해당 컬럼을 묶는다
                if agg == "mean":  # 평균으로 채우는 경우
                    group_value = grouped.transform("mean")  # 행마다 자기 그룹의 평균을 붙인다
                else:  # 중앙값으로 채우는 경우
                    group_value = grouped.transform("median")  # 행마다 자기 그룹의 중앙값을 붙인다
                self.df[col] = self.df[col].fillna(group_value)  # 빈 칸을 자기 그룹 값으로 채운다
                fallback = self.df[col].agg(agg)  # 그룹 전체가 비었을 때 쓸 전체 통계
            elif strategy == "mode":  # 최빈값(가장 자주 나온 값)으로 채우는 전략이면
                fallback = self.df[col].mode().iloc[0]  # 가장 자주 나온 값을 고른다
            else:  # mean, median 전략이면
                fallback = self.df[col].agg(strategy)  # 전체 평균 또는 중앙값을 구한다
            self.df[col] = self.df[col].fillna(fallback)  # 아직 남은 빈 칸을 마저 채운다
        return self.df  # 채운 표를 돌려준다

    # 3. 이상치 처리 (IQR)

    def get_iqr_bounds(self, column, threshold=1.5):  # IQR로 정상 범위를 계산한다
        """IQR 방식의 정상 범위 (하한, 상한)를 계산한다."""
        q1 = self.df[column].quantile(0.25)  # 작은 쪽에서 25% 지점의 값
        q3 = self.df[column].quantile(0.75)  # 작은 쪽에서 75% 지점의 값
        iqr = q3 - q1  # 가운데 50%가 퍼진 폭
        return float(q1 - threshold * iqr), float(q3 + threshold * iqr)  # (하한, 상한)을 일반 소수로 바꿔 돌려준다

    def detect_outliers(self, column, threshold=1.5):  # 정상 범위 밖 행을 찾는다
        """정상 범위를 벗어난 행을 돌려준다."""
        lower, upper = self.get_iqr_bounds(column, threshold)  # 하한과 상한을 구한다
        return self.df[(self.df[column] < lower) | (self.df[column] > upper)]  # 하한보다 작거나 상한보다 큰 행만 돌려준다

    def handle_outliers(self, column, threshold=1.5, method="clip"):  # 이상치를 자르거나 지운다
        """이상치를 처리한다.

        method
            clip: 범위 밖 값을 하한/상한 값으로 바꾼다. 행 수는 그대로다.
            remove: 범위 밖 행을 지운다.
        """
        lower, upper = self.get_iqr_bounds(column, threshold)  # 하한과 상한을 구한다
        if method == "clip":  # 경계값으로 자르는 방법이면
            self.df[column] = self.df[column].clip(lower, upper)  # 범위 밖 값을 하한/상한으로 바꾼다
        elif method == "remove":  # 지우는 방법이면
            mask = self.df[column].between(lower, upper)  # 정상 범위 안이면 True인 표시를 만든다
            self.df = self.df[mask].reset_index(drop=True)  # 정상 행만 남기고 번호를 다시 매긴다
        else:  # 그 밖의 값이면
            raise ValueError(f"method는 clip 또는 remove여야 한다: {method}")  # 에러로 알려준다
        return self.df  # 처리한 표를 돌려준다

    # 4. 피처 엔지니어링

    def parse_images(self, image_col="image_pixels", shape=(16, 16, 3), step=1):  # 픽셀 문자열을 이미지 배열로 바꾼다
        """공백으로 구분된 픽셀 문자열 컬럼을 (N, H, W, C) 배열로 바꾼다.

        모든 행의 문자열을 하나로 이어 np.fromstring으로 한 번에 읽고 reshape한다.
        step > 1이면 arr[:, ::step, ::step]으로 다운샘플링한다.
        """
        joined = " ".join(self.df[image_col].astype(str))  # 모든 행의 문자열을 공백으로 이어 붙인다
        flat = np.fromstring(joined, dtype=np.int64, sep=" ")  # 문자열을 1차원 숫자 배열로 한 번에 읽는다
        images = flat.reshape((len(self.df),) + tuple(shape)).astype(np.uint8)  # (이미지 수, 세로, 가로, 색) 모양으로 바꾼다
        return images[:, ::step, ::step]  # step칸마다 1칸씩 뽑아 이미지를 줄여 돌려준다

    def add_image_features(self, image_col="image_pixels", shape=(16, 16, 3), step=1):  # 이미지 평균과 표준편차 컬럼을 만든다
        """이미지별 픽셀 평균(img_mean)과 표준편차(img_std) 컬럼을 추가한다."""
        images = self.parse_images(image_col, shape, step).astype(np.float64)  # 이미지를 소수 배열로 바꾼다
        axes = tuple(range(1, images.ndim))  # 이미지 번호(0번 축)를 뺀 나머지 축 전체
        self.df["img_mean"] = images.mean(axis=axes)  # 이미지마다 평균 밝기를 구한다
        self.df["img_std"] = images.std(axis=axes)  # 이미지마다 밝기가 퍼진 정도를 구한다
        return self.df  # 컬럼을 추가한 표를 돌려준다

    def add_image_advanced_features(self, image_col="image_pixels", shape=(16, 16, 3), bins=4, step=1):  # 밝기 비율과 엣지 컬럼을 만든다
        """밝기 히스토그램 비율(img_hist_0..)과 엣지 강도(img_edge) 컬럼을 추가한다."""
        images = self.parse_images(image_col, shape, step).astype(np.float64)  # 이미지를 소수 배열로 바꾼다
        gray = images.mean(axis=3) if images.ndim == 4 else images  # (N, H, W) 흑백

        # 밝기 구간 번호를 구한 뒤, 구간마다 픽셀 비율을 센다
        bin_idx = np.clip((gray / 256 * bins).astype(int), 0, bins - 1)  # 픽셀마다 밝기 구간 번호(0 ~ bins-1)를 정한다
        hist = (bin_idx[..., np.newaxis] == np.arange(bins)).mean(axis=(1, 2))  # 이미지마다 구간별 픽셀 비율을 구한다
        for i in range(bins):  # 구간을 하나씩 돌며
            self.df[f"img_hist_{i}"] = hist[:, i]  # 구간별 비율을 컬럼으로 넣는다

        # 이웃 픽셀 간 밝기 차이의 평균 = 엣지 강도
        gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))  # 가로 이웃 픽셀과의 밝기 차이 평균
        gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))  # 세로 이웃 픽셀과의 밝기 차이 평균
        self.df["img_edge"] = (gx + gy) / 2  # 가로, 세로 평균을 엣지 강도로 쓴다
        return self.df  # 컬럼을 추가한 표를 돌려준다

    def add_text_features(self, text_col="product_name"):  # 상품명의 단어 수와 글자 수 컬럼을 만든다
        """상품명의 단어 수(name_word_count)와 글자 수(name_length) 컬럼을 추가한다."""
        text = self.df[text_col].fillna("").astype(str)  # 빈 값은 빈 글자로 바꾸고 문자열로 맞춘다
        self.df["name_word_count"] = text.str.split().str.len()  # type: ignore  # 공백으로 자른 조각 수 = 단어 수 (pandas 타입 힌트 한계로 경고 무시)
        self.df["name_length"] = text.str.len()  # 공백을 포함한 글자 수
        return self.df  # 컬럼을 추가한 표를 돌려준다

    def add_amount(self, price_col="price", qty_col="quantity", out_col="amount"):  # 주문 금액 컬럼을 만든다
        """주문 금액 = 가격 x 수량 컬럼을 추가한다."""
        self.df[out_col] = self.df[price_col] * self.df[qty_col]  # 가격과 수량을 곱해 주문 금액을 만든다
        return self.df  # 컬럼을 추가한 표를 돌려준다

    # 5. 통계 분석

    def summary_statistics(self, columns):  # 컬럼별 요약 통계표를 만든다
        """평균, 중앙값, 표준편차, 사분위수, IQR을 표로 돌려준다."""
        data = self.df[columns]  # 요약할 컬럼만 고른다
        summary = pd.DataFrame({  # 통계를 컬럼으로 하는 표를 만든다
            "mean": data.mean(),  # 평균
            "median": data.median(),  # 중앙값
            "std": data.std(),  # 표준편차
            "min": data.min(),  # 최소값
            "q1": data.quantile(0.25),  # 25% 지점
            "q3": data.quantile(0.75),  # 75% 지점
            "max": data.max(),  # 최대값
        })
        summary["iqr"] = summary["q3"] - summary["q1"]  # IQR = Q3 - Q1
        return summary  # 통계표를 돌려준다

    def correlation(self, columns, method="pearson"):  # 상관계수 표를 만든다
        """상관계수 행렬을 돌려준다."""
        return self.df[columns].corr(method=method)  # 컬럼끼리 서로 얼마나 함께 움직이는지 계산한다

    # 6. RFM 고객 세분화

    def calculate_rfm(self, customer_col, date_col, amount_col, reference_date=None, order_col=None, n_bins=5):  # 고객별 R, F, M을 계산한다
        """고객별 Recency, Frequency, Monetary 값과 1~n_bins 점수를 계산한다.

        reference_date: Recency 기준일. 없으면 데이터의 마지막 거래일.
        order_col: 주문 번호 컬럼. 있으면 고유 주문 수, 없으면 행 수를 Frequency로 쓴다.
        """
        if reference_date is None:  # 기준일을 안 정했으면
            reference_date = self.df[date_col].max()  # 데이터의 마지막 거래일을 기준일로 쓴다
        reference_date = pd.Timestamp(reference_date)  # 날짜형으로 맞춘다

        grouped = self.df.groupby(customer_col)  # 고객별로 묶는다
        rfm = pd.DataFrame({  # 고객 한 명이 한 행인 표를 만든다
            "recency": (reference_date - grouped[date_col].max()).dt.days,  # 기준일 - 마지막 구매일 (일 수)
            "frequency": grouped[order_col].nunique() if order_col else grouped.size(),  # 주문 횟수 (주문 번호가 있으면 중복 없이 센다)
            "monetary": grouped[amount_col].sum(),  # 총 구매 금액
        })

        # 순위를 먼저 매겨 동점을 없앤 뒤 같은 인원 수로 n_bins 구간을 나눈다
        # Recency는 작을수록 좋으므로 음수로 바꿔 순위를 매긴다
        labels = list(range(1, n_bins + 1))  # 점수 이름: 1, 2, ..., n_bins
        rfm["r_score"] = pd.qcut((-rfm["recency"]).rank(method="first"), n_bins, labels=labels).astype(int)  # 최근일수록 높은 점수
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), n_bins, labels=labels).astype(int)  # 자주 샀을수록 높은 점수
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), n_bins, labels=labels).astype(int)  # 많이 썼을수록 높은 점수
        rfm["rfm_total"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)  # 세 점수의 합

        self.rfm = rfm  # 결과를 객체에 저장한다
        return self.rfm  # 저장한 표를 돌려준다

    def segment_customers(self):  # 점수로 고객을 세그먼트로 나눈다
        """RFM 점수로 고객을 6개 세그먼트로 나눈다. (위 규칙부터 먼저 적용)

        VIP      R>=4, F>=4, M>=4  최근에, 자주, 많이 산다
        Loyal    R>=3, F>=3        꾸준히 산다
        New      R>=4, F<=2        최근에 처음 샀다
        At Risk  R<=2, F>=3        자주 샀지만 최근에 안 온다
        Churned  R<=2, F<=2        오래전에 조금 사고 떠났다
        Regular  나머지
        """
        if self.rfm is None:  # RFM을 아직 계산하지 않았으면
            raise ValueError("calculate_rfm()을 먼저 실행해야 한다")  # 에러로 알려준다
        r, f, m = self.rfm["r_score"], self.rfm["f_score"], self.rfm["m_score"]  # 세 점수 컬럼을 꺼낸다
        conditions = [  # 위에서부터 먼저 맞는 조건이 이긴다
            (r >= 4) & (f >= 4) & (m >= 4),  # VIP
            (r >= 3) & (f >= 3),  # Loyal
            (r >= 4) & (f <= 2),  # New
            (r <= 2) & (f >= 3),  # At Risk
            (r <= 2) & (f <= 2),  # Churned
        ]
        choices = ["VIP", "Loyal", "New", "At Risk", "Churned"]  # 조건 순서와 같은 세그먼트 이름
        self.rfm["segment"] = np.select(conditions, choices, default="Regular")  # 처음 맞는 조건의 이름을 고르고, 없으면 Regular
        return self.rfm  # 세그먼트 컬럼이 추가된 표를 돌려준다

    def segment_summary(self):  # 세그먼트별 요약표를 만든다
        """세그먼트별 고객 수, 평균 R/F/M, 매출 비중을 정리한다."""
        if self.rfm is None or "segment" not in self.rfm:  # 세그먼트를 아직 안 나눴으면
            raise ValueError("segment_customers()를 먼저 실행해야 한다")  # 에러로 알려준다
        summary = self.rfm.groupby("segment").agg(  # 세그먼트별로 묶어 통계를 낸다
            customers=("recency", "size"),  # 고객 수
            avg_recency=("recency", "mean"),  # 평균 Recency
            avg_frequency=("frequency", "mean"),  # 평균 주문 횟수
            avg_monetary=("monetary", "mean"),  # 평균 구매 금액
            total_revenue=("monetary", "sum"),  # 세그먼트 전체 매출
        )
        summary["customer_share"] = summary["customers"] / summary["customers"].sum() * 100  # 전체 고객 중 비중(%)
        summary["revenue_share"] = summary["total_revenue"] / summary["total_revenue"].sum() * 100  # 전체 매출 중 비중(%)
        order = [s for s in SEGMENT_ORDER if s in summary.index]  # 정해 둔 순서대로, 실제로 있는 세그먼트만 고른다
        return summary.loc[order]  # 그 순서로 정렬해 돌려준다

    # 7. 시계열, 코호트

    def monthly_sales(self, date_col="order_date", amount_col="amount"):  # 월별 매출을 구한다
        """월별 매출 합계를 돌려준다."""
        return self.df.groupby(self.df[date_col].dt.to_period("M"))[amount_col].sum()  # 주문일을 월 단위로 묶어 금액을 더한다

    def cohort_retention(self, customer_col="customer_id", date_col="order_date"):  # 코호트별 재구매율을 구한다
        """첫 구매 월(코호트)별로 N개월 뒤에도 구매한 고객 비율을 계산한다."""
        df = self.df[[customer_col, date_col]].copy()  # 필요한 두 컬럼만 복사한다
        df["order_month"] = df[date_col].dt.to_period("M")  # 주문일을 월 단위로 바꾼다
        df["cohort"] = df.groupby(customer_col)["order_month"].transform("min")  # 고객마다 가장 이른 월 = 첫 구매 월

        # 연*12+월 로 바꿔 빼면 몇 개월 차이인지 반복문 없이 구할 수 있다
        order_num = df[date_col].dt.year * 12 + df[date_col].dt.month  # 주문 월을 하나의 숫자로 바꾼다
        cohort_num = df["cohort"].dt.year * 12 + df["cohort"].dt.month  # 첫 구매 월도 같은 방식으로 바꾼다
        df["months_since"] = order_num - cohort_num  # 첫 구매 후 몇 개월 지났는지

        counts = df.groupby(["cohort", "months_since"])[customer_col].nunique().unstack()  # 코호트(행) x 경과 개월(열)별 고객 수
        return counts.div(counts[0], axis=0)  # 0개월 차 고객 수로 나눠 비율로 바꾼다


def main():  # 기본 데이터로 전체 흐름을 실행한다
    """기본 데이터로 전체 파이프라인을 실행하고 결과를 출력한다."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 최상위 폴더 경로
    orders_path = os.path.join(root, "data", "orders.csv")  # 주문 파일 경로
    products_path = os.path.join(root, "data", "products.csv")  # 상품 파일 경로

    pd.set_option("display.width", 160)  # 출력 한 줄의 너비
    pd.set_option("display.max_columns", None)  # 컬럼을 줄이지 않고 모두 보여준다
    pd.set_option("display.float_format", "{:,.2f}".format)  # 소수는 천 단위 쉼표와 소수 2자리로 보여준다

    analyzer = DataAnalyzer(orders_path, product_path=products_path)  # 분석기를 만든다
    df = analyzer.load_data(date_cols=["order_date", "registered_date"])  # 데이터를 읽고 날짜 컬럼을 날짜형으로 바꾼다
    print(f"[1] 데이터 로드: {df.shape[0]:,}행 x {df.shape[1]}열")  # 표의 크기를 출력한다
    print(df.drop(columns="image_pixels").head(3), "\n")  # 긴 이미지 컬럼을 빼고 앞 3행을 보여준다

    print("[2] 결측치 현황")  # 결측치 단계 제목
    print(analyzer.missing_report(), "\n")  # 컬럼별 빈 값 현황을 출력한다
    analyzer.handle_missing_values(strategy="group_median", group_col="category", columns=["price", "rating"])  # 카테고리별 중앙값으로 채운다
    print(f"    category별 중앙값으로 대치 후 남은 결측치: {int(analyzer.df[['price', 'rating']].isna().sum().sum())}개\n")  # 남은 빈 값 수를 확인한다

    analyzer.add_amount("price", "quantity")  # 주문 금액 컬럼을 만든다
    analyzer.add_image_features("image_pixels", shape=(16, 16, 3))  # 이미지 평균, 표준편차 컬럼을 만든다
    analyzer.add_image_advanced_features("image_pixels", shape=(16, 16, 3), bins=4)  # 밝기 비율, 엣지 컬럼을 만든다
    analyzer.add_text_features("product_name")  # 상품명 단어 수, 글자 수 컬럼을 만든다
    print("[3] 피처 엔지니어링: img_mean, img_std, img_hist_0~3, img_edge, name_word_count, name_length 추가")  # 단계 제목
    print(analyzer.df[["product_name", "img_mean", "img_std", "img_edge", "name_word_count", "name_length"]].head(3), "\n")  # 새 컬럼 앞 3행을 보여준다

    lower, upper = analyzer.get_iqr_bounds("amount", threshold=1.5)  # 주문 금액의 정상 범위를 구한다
    outliers = analyzer.detect_outliers("amount", threshold=1.5)  # 범위 밖 이상치 행을 찾는다
    print(f"[4] amount IQR 정상 범위: {lower:,.0f} ~ {upper:,.0f}, 이상치 {len(outliers)}건 ({len(outliers) / len(df) * 100:.1f}%)")  # 범위와 이상치 수를 출력한다
    print("    RFM 매출 합계를 보존하기 위해 원본 amount는 유지하고, 통계용 amount_clipped 컬럼을 따로 만든다\n")  # 이상치를 지우지 않는 이유를 알려준다
    analyzer.df["amount_clipped"] = analyzer.df["amount"].clip(lower, upper)  # 이상치를 경계값으로 자른 컬럼을 따로 만든다

    print("[5] 요약 통계")  # 통계 단계 제목
    print(analyzer.summary_statistics(["price", "quantity", "amount", "rating", "img_mean"]), "\n")  # 요약 통계표를 출력한다
    print("    상관계수")  # 상관계수 제목
    print(analyzer.correlation(["price", "quantity", "amount", "rating", "img_mean", "img_std"]), "\n")  # 상관계수 표를 출력한다

    analyzer.calculate_rfm("customer_id", "order_date", "amount", order_col="order_id")  # 고객별 R, F, M 점수를 계산한다
    analyzer.segment_customers()  # 점수로 세그먼트를 나눈다
    print(f"[6] RFM 세분화 (기준일: {analyzer.df['order_date'].max().date()})")  # 기준일과 함께 제목을 출력한다
    print(analyzer.segment_summary(), "\n")  # 세그먼트별 요약표를 출력한다

    print("[7] 월별 매출 (최근 6개월)")  # 월별 매출 제목
    print(analyzer.monthly_sales("order_date", "amount").tail(6))  # 마지막 6개월 매출을 출력한다


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만
    main()  # 전체 흐름을 실행한다
