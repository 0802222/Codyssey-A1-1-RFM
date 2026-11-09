"""이커머스 합성 데이터 생성 스크립트.

고정 시드(42)를 사용하므로 몇 번을 실행해도 같은 데이터가 나온다.
이 파일의 `rng`는 Random Number Generator(난수 생성기)의 줄임말이다.
`rng.choice`, `rng.integers` 같은 메서드로 무작위 값을 뽑는다.

생성 파일
    data/products.csv  상품 200개
        (상품명, 카테고리, 가격, 평점, 등록일, 16x16 RGB 이미지 배열)
    data/orders.csv    주문 약 5,000건 (실제 5,160건)
        (주문번호, 고객, 주문일, 상품, 수량, 지역, 결제수단)

실행: python -m data.generate_data  (프로젝트 최상위 폴더에서)
"""

import os  # 파일 경로를 다루는 기본 모듈
from typing import NamedTuple  # 이름 붙은 묶음(튜플)을 만드는 도구

import numpy as np  # 배열 계산 라이브러리
import pandas as pd  # 표(DataFrame) 처리 라이브러리

from src.constants import DATA_DIR, IMAGE_SHAPE  # 공통 상수: 저장 폴더, 이미지 모양

# 난수 시드(Seed). 같은 값이면 항상 같은 데이터가 나온다. 42는 개발자 관습값이다
SEED = 42
N_PRODUCTS = 200  # 만들 상품 수
N_CUSTOMERS = 1000  # 만들 고객 수
OUT_DIR = DATA_DIR  # CSV를 저장할 폴더 (공통 상수를 그대로 쓴다)
START_DATE = pd.Timestamp("2024-01-01")  # 주문 기간 시작일
END_DATE = pd.Timestamp("2025-12-31")  # 주문 기간 마지막일

PRICE_MISSING_RATE = 0.06  # 가격을 빈 값으로 만들 비율 (6%)
RATING_MISSING_RATE = 0.10  # 평점을 빈 값으로 만들 비율 (10%)
BULK_ORDER_RATE = 0.015  # 모든 주문이 대량 주문(이상치)이 될 확률 (1.5%)
BULK_ORDER_EXTRA_RATE = 0.02  # Food, Books 주문이 대량 주문이 될 추가 확률 (2%)
BULK_PRONE_CATEGORIES = ("Food", "Books")  # 대량 주문이 더 잘 나오는 카테고리


class NameParts(NamedTuple):  # 상품명 재료 한 묶음
    """카테고리 하나의 상품명 재료."""

    adjectives: list[str]  # 꾸미는 말 목록
    nouns: list[str]  # 물건 이름 목록


class CustomerType(NamedTuple):  # 고객 유형 한 가지의 행동 규칙
    """고객 유형 하나의 행동 규칙."""

    share: float  # 전체 고객 중 이 유형의 비율
    extra_order_mean: float  # 첫 주문 말고 추가로 하는 주문 수의 평균
    category_weights: list[float]  # 카테고리 선호도. CATEGORIES 순서이고 합이 1


# 카테고리별 기준 가격(원)과 이미지 기본 색(RGB)
CATEGORIES = {
    # 전자기기: 비싸고 어두운 색
    "Electronics": {"base_price": 180000, "base_color": (90, 100, 120)},
    "Fashion": {"base_price": 45000, "base_color": (200, 120, 140)},  # 패션
    "Beauty": {"base_price": 30000, "base_color": (230, 180, 200)},  # 뷰티: 밝은 색
    "Home": {"base_price": 40000, "base_color": (170, 150, 110)},  # 홈
    "Food": {"base_price": 15000, "base_color": (220, 160, 60)},  # 식품: 가장 저렴
    "Books": {"base_price": 18000, "base_color": (120, 160, 200)},  # 도서
}

NAME_PARTS = {  # 카테고리별 상품명 재료
    "Electronics": NameParts(
        ["Wireless", "Smart", "Portable", "Ultra", "Pro"],
        ["Earbuds", "Speaker", "Keyboard", "Mouse", "Charger", "Monitor"],
    ),
    "Fashion": NameParts(
        ["Classic", "Slim", "Oversized", "Cotton", "Vintage"],
        ["T-Shirt", "Jeans", "Hoodie", "Jacket", "Sneakers", "Skirt"],
    ),
    "Beauty": NameParts(
        ["Hydrating", "Daily", "Vegan", "Mild", "Glow"],
        ["Toner", "Serum", "Sunscreen", "Lip Balm", "Cream", "Cleanser"],
    ),
    "Home": NameParts(
        ["Modern", "Compact", "Soft", "Eco", "Large"],
        ["Cushion", "Lamp", "Towel Set", "Storage Box", "Mug", "Blanket"],
    ),
    "Food": NameParts(
        ["Organic", "Premium", "Spicy", "Roasted", "Fresh"],
        ["Coffee Beans", "Granola", "Nuts", "Green Tea", "Snack Box", "Olive Oil"],
    ),
    "Books": NameParts(
        ["Illustrated", "Essential", "Beginner", "Complete", "Pocket"],
        ["Python Guide", "Novel", "Cookbook", "Travel Essay", "Data Science", "Poetry"],
    ),
}

CUSTOMER_TYPES = {  # 고객 유형별 행동 규칙
    # 단골: 많이, 자주, 전자기기 선호
    "vip": CustomerType(0.08, 18, [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]),
    # 충성: 꾸준히 구매
    "loyal": CustomerType(0.20, 8, [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]),
    # 일반: 가끔 구매
    "regular": CustomerType(0.32, 3, [0.10, 0.25, 0.20, 0.15, 0.20, 0.10]),
    # 신규: 최근에 처음 구매
    "new": CustomerType(0.15, 0.5, [0.10, 0.25, 0.20, 0.15, 0.20, 0.10]),
    # 이탈: 일찍 떠남
    "churned": CustomerType(0.25, 1, [0.05, 0.25, 0.20, 0.15, 0.25, 0.10]),
}

# 지역 7곳
REGIONS = ["Seoul", "Gyeonggi", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju"]
REGION_PROBS = [0.30, 0.27, 0.12, 0.10, 0.08, 0.07, 0.06]  # 지역별 뽑힐 확률 (합 1)
PAYMENTS = ["Card", "Kakao Pay", "Naver Pay", "Bank Transfer"]  # 결제 수단 4가지
PAYMENT_PROBS = [0.50, 0.22, 0.20, 0.08]  # 결제 수단별 뽑힐 확률 (합 1)


def make_names(rng: np.random.Generator, product_categories: np.ndarray) -> list[str]:
    """카테고리에 맞는 상품명(꾸미는 말 + 물건 이름 + 에디션)을 만든다."""
    # 상품명 앞부분(꾸미는 말)
    adjectives = [
        rng.choice(NAME_PARTS[name].adjectives) for name in product_categories
    ]
    # 상품명 중간(물건 이름)
    nouns = [rng.choice(NAME_PARTS[name].nouns) for name in product_categories]
    # 상품명 뒷부분(에디션)
    editions = rng.choice(
        ["", " Mini", " Plus", " Edition 2", " Gift Set"],
        size=N_PRODUCTS,
        p=[0.5, 0.15, 0.15, 0.1, 0.1],
    )
    # 세 부분을 이어 상품명을 만든다
    return [
        f"{adjective} {noun}{edition}"
        for adjective, noun, edition in zip(adjectives, nouns, editions)
    ]


def make_images(
    rng: np.random.Generator, product_categories: np.ndarray, prices: np.ndarray
) -> np.ndarray:
    """상품마다 16x16 RGB 이미지를 만든다.

    카테고리 기본색 + 가로 그라데이션 + 상품마다 다른 노이즈를 더한다.
    비싼 상품일수록 어둡게 만들어 가격과 밝기 사이에 약한 관계를 둔다.
    """
    image_width = IMAGE_SHAPE[1]  # 가로 크기만 꺼낸다 (그라데이션이 가로로 변한다)
    # 카테고리 기본색을 이미지 모양에 맞게 늘린다
    base_colors = np.array(
        [CATEGORIES[name]["base_color"] for name in product_categories], dtype=float
    )[:, None, None, :]
    # 왼쪽은 어둡고 오른쪽은 밝은 띠
    gradient = (
        np.linspace(-30, 30, image_width)[None, None, :, None]
        * rng.uniform(0.2, 1.5, N_PRODUCTS)[:, None, None, None]
    )
    log_prices = np.log(prices)  # 가격 차이가 너무 크니 로그로 줄인다
    # 비쌀수록 더 어둡게 하는 값
    darkness = -18 * (log_prices - log_prices.mean()) / log_prices.std()
    # 상품마다 세기가 다른 잡음
    noise = (
        rng.normal(0, 1, (N_PRODUCTS,) + IMAGE_SHAPE)
        * rng.uniform(5, 40, N_PRODUCTS)[:, None, None, None]
    )
    # 모두 더해 0~255로 자르고 정수로 바꾼다
    images: np.ndarray = np.clip(
        base_colors + gradient + darkness[:, None, None, None] + noise, 0, 255
    ).astype(np.uint8)
    return images  # 이미지 배열을 돌려준다


def inject_missing(rng: np.random.Generator, products: pd.DataFrame) -> None:
    """상품 표에 결측치(빈 값)를 넣는다. 가격 약 6%, 평점 약 10%."""
    # 가격 약 6%를 빈 값으로 만든다
    products.loc[rng.random(N_PRODUCTS) < PRICE_MISSING_RATE, "price"] = np.nan
    # 평점 약 10%를 빈 값으로 만든다
    products.loc[rng.random(N_PRODUCTS) < RATING_MISSING_RATE, "rating"] = np.nan


def make_products(rng: np.random.Generator) -> pd.DataFrame:  # 상품 표를 만든다
    """상품 표와 이미지 배열을 만든다.

    난수를 뽑는 순서가 바뀌면 결과가 달라지므로 호출 순서를 지켜야 한다.
    (카테고리 -> 가격 -> 상품명 -> 등록일 -> 평점 -> 이미지 -> 결측치)
    """
    category_names = list(CATEGORIES)  # 카테고리 이름 목록
    # 상품마다 카테고리를 무작위로 정한다
    product_categories = rng.choice(category_names, size=N_PRODUCTS)

    # 상품마다 카테고리 기준 가격을 가져온다
    base_prices = np.array(
        [CATEGORIES[name]["base_price"] for name in product_categories]
    )
    # 기준 가격에 무작위 배수를 곱하고 100원 단위로 반올림한다
    prices = np.round(base_prices * rng.lognormal(0, 0.45, N_PRODUCTS), -2)

    product_names = make_names(rng, product_categories)  # 상품명을 만든다

    first_register_date = pd.Timestamp("2023-06-01")  # 등록일 범위의 시작
    last_register_date = pd.Timestamp("2025-10-31")  # 등록일 범위의 끝
    # 등록일 범위의 일 수
    register_span_days = (last_register_date - first_register_date).days
    # 범위 안에서 무작위 등록일
    registered_dates = first_register_date + pd.to_timedelta(
        rng.integers(0, register_span_days, N_PRODUCTS), unit="D"
    )

    # 평균 4.1 근처 평점을 1~5로 자르고 소수 1자리로 맞춘다
    ratings = np.clip(rng.normal(4.1, 0.45, N_PRODUCTS), 1, 5).round(1)

    images = make_images(rng, product_categories, prices)  # 이미지를 만든다

    products = pd.DataFrame(
        {  # 상품 표를 만든다
            # 상품 번호: P0001, P0002, ...
            "product_id": [f"P{number:04d}" for number in range(1, N_PRODUCTS + 1)],
            "product_name": product_names,  # 상품명
            "category": product_categories,  # 카테고리
            "price": prices,  # 가격
            "rating": ratings,  # 평점
            # 등록일을 문자열로 바꿔 저장한다
            "registered_date": registered_dates.strftime("%Y-%m-%d"),
            # 이미지를 768개 숫자를 공백으로 이은 문자열로 저장한다
            "image_pixels": [" ".join(map(str, image.ravel())) for image in images],
        }
    )

    inject_missing(rng, products)  # 결측치를 넣는다
    return products  # 상품 표를 돌려준다


# 날짜별 주문 가중치를 정한다
def seasonal_weight(dates: pd.DatetimeIndex) -> np.ndarray:
    """11~12월은 더 많이, 2월은 조금 적게 주문이 들어오게 하는 가중치."""
    month = dates.month  # 날짜에서 월만 꺼낸다
    # 11~12월 1.8배, 2월 0.8배, 나머지 1배
    return np.where(month.isin([11, 12]), 1.8, np.where(month == 2, 0.8, 1.0))


def activity_period(
    rng: np.random.Generator, customer_type: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """고객 유형별 활동 기간(첫 구매일, 마지막 활동일)을 정한다."""
    if customer_type == "new":  # 신규 고객이면
        # 최근 60일 안에 처음 구매한다
        first_order_date = END_DATE - pd.Timedelta(days=int(rng.integers(0, 60)))
        last_active_date = END_DATE  # 마지막 날까지 활동한다
    elif customer_type == "churned":  # 이탈 고객이면
        # 앞쪽 시기에 처음 구매한다
        first_order_date = START_DATE + pd.Timedelta(days=int(rng.integers(0, 420)))
        # 처음 구매 후 120일 안에 떠난다
        last_active_date = first_order_date + pd.Timedelta(
            days=int(rng.integers(0, 120))
        )
    elif customer_type == "regular":  # 일반 고객이면
        # 처음 구매일을 정한다
        first_order_date = START_DATE + pd.Timedelta(days=int(rng.integers(0, 540)))
        # 마지막 구매는 끝나기 30~270일 전이다
        last_active_date = END_DATE - pd.Timedelta(days=int(rng.integers(30, 270)))
        # 마지막이 처음보다 앞서지 않게 맞춘다
        last_active_date = max(last_active_date, first_order_date)
    else:  # vip, loyal이면
        # 첫 해 안에 처음 구매한다
        first_order_date = START_DATE + pd.Timedelta(days=int(rng.integers(0, 365)))
        # 끝나기 직전까지 활동한다
        last_active_date = END_DATE - pd.Timedelta(days=int(rng.integers(0, 25)))
    # 첫 구매일과 마지막 활동일을 돌려준다
    return first_order_date, last_active_date


def pick_quantity(rng: np.random.Generator, category_name: str) -> int:
    """주문 수량을 정한다. 대부분 1~2개이고 드물게 10~30개 대량 주문이 나온다."""
    quantity = 1 + rng.poisson(0.35)  # 수량은 대부분 1~2개
    # 대량 주문(이상치): 모든 주문 1.5%, Food/Books는 2% 더
    is_bulk_order = rng.random() < BULK_ORDER_RATE or (
        category_name in BULK_PRONE_CATEGORIES and rng.random() < BULK_ORDER_EXTRA_RATE
    )
    if is_bulk_order:  # 드물게 대량 주문이 되는 경우
        quantity = int(rng.integers(10, 31))  # 10~30개로 크게 만든다
    return int(quantity)  # 정수로 돌려준다


def make_orders(rng: np.random.Generator, products: pd.DataFrame) -> pd.DataFrame:
    """고객 유형별 행동 규칙에 따라 주문 표를 만든다.

    난수를 뽑는 순서가 바뀌면 결과가 달라지므로 고객마다
    지역 -> 활동 기간 -> 주문 횟수 -> 주문일 -> 주문 내용 순서를 지켜야 한다.
    """
    # 주문 기간 안의 모든 날짜
    period_days = pd.date_range(START_DATE, END_DATE, freq="D")
    # 날짜마다 주문 가중치를 구한다
    period_day_weights = seasonal_weight(period_days)
    category_names = list(CATEGORIES)  # 카테고리 이름 목록
    # 상품별 카테고리 배열
    product_categories = products["category"].to_numpy()

    type_names = list(CUSTOMER_TYPES)  # 고객 유형 이름 목록
    # 유형별 고객 비율
    type_shares = [CUSTOMER_TYPES[name].share for name in type_names]
    # 고객마다 유형을 무작위로 정한다
    customer_types = rng.choice(type_names, size=N_CUSTOMERS, p=type_shares)

    order_rows = []  # 주문 한 건씩 모아 둘 목록
    # 고객을 한 명씩 처리한다
    for customer_index, customer_type in enumerate(customer_types):
        rule = CUSTOMER_TYPES[customer_type]  # 이 고객 유형의 행동 규칙
        # 고객 번호: C00001, C00002, ...
        customer_id = f"C{customer_index + 1:05d}"
        # 고객의 지역을 정한다
        region = rng.choice(REGIONS, p=REGION_PROBS)
        # 활동 기간을 정한다
        first_order_date, last_active_date = activity_period(rng, customer_type)

        # 주문 횟수 = 1회 + 무작위 추가 횟수
        order_count = 1 + rng.poisson(rule.extra_order_mean)
        # 활동 기간 안의 날짜만 표시한다
        is_active_day = (period_days >= first_order_date) & (
            period_days <= last_active_date
        )
        active_days = period_days[is_active_day]  # 활동 기간의 날짜
        active_day_weights = period_day_weights[is_active_day]  # 그 날짜들의 가중치
        # 첫 구매일 + 가중치대로 뽑은 나머지 주문일
        order_dates = (
            [first_order_date]
            + list(
                rng.choice(
                    active_days,
                    size=order_count - 1,
                    p=active_day_weights / active_day_weights.sum(),
                )
            )
            if order_count > 1
            else [first_order_date]
        )

        for order_date in order_dates:  # 주문일마다 주문 한 건을 만든다
            # 고객 선호도대로 카테고리를 고른다
            category_name = rng.choice(category_names, p=rule.category_weights)
            # 그 카테고리 상품 중 하나의 행 번호를 고른다
            product_row = rng.choice(
                np.flatnonzero(product_categories == category_name)
            )
            quantity = pick_quantity(rng, category_name)  # 수량을 정한다
            order_rows.append(
                {  # 주문 한 건을 목록에 추가한다
                    "customer_id": customer_id,  # 고객 번호
                    # 주문일(문자열)
                    "order_date": pd.Timestamp(order_date).strftime("%Y-%m-%d"),
                    # 상품 번호
                    "product_id": products.loc[product_row, "product_id"],
                    "quantity": quantity,  # 수량
                    "region": region,  # 지역
                    # 결제 수단
                    "payment_method": rng.choice(PAYMENTS, p=PAYMENT_PROBS),
                }
            )

    # 표로 만들고 날짜, 고객 순으로 정렬한다
    orders = (
        pd.DataFrame(order_rows)
        .sort_values(["order_date", "customer_id"])
        .reset_index(drop=True)
    )
    # 맨 앞에 주문 번호(100001부터)를 넣는다
    orders.insert(0, "order_id", np.arange(100001, 100001 + len(orders)))
    return orders  # 주문 표를 돌려준다


def main() -> None:  # 두 CSV 파일을 만든다
    """상품과 주문 CSV를 만들어 data 폴더에 저장한다."""
    # rng = Random Number Generator(난수 생성기). 시드를 고정해서 만든다
    rng = np.random.default_rng(SEED)
    products = make_products(rng)  # 상품 표를 만든다
    orders = make_orders(rng, products)  # 주문 표를 만든다
    # 상품 표를 CSV로 저장한다
    products.to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
    # 주문 표를 CSV로 저장한다
    orders.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
    # 만든 표의 크기를 출력한다
    print(f"products.csv: {products.shape}, orders.csv: {orders.shape}")


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만
    main()  # CSV 생성을 실행한다
