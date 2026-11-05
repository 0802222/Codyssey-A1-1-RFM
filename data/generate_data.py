"""이커머스 합성 데이터 생성 스크립트.

고정 시드(42)를 사용하므로 몇 번을 실행해도 같은 데이터가 나온다.

생성 파일
    data/products.csv  상품 200개
        (상품명, 카테고리, 가격, 평점, 등록일, 16x16 RGB 이미지 배열)
    data/orders.csv    주문 약 6,000건
        (주문번호, 고객, 주문일, 상품, 수량, 지역, 결제수단)

실행: python data/generate_data.py
"""

import os  # 파일 경로를 다루는 기본 모듈

import numpy as np  # 배열 계산 라이브러리
import pandas as pd  # 표(DataFrame) 처리 라이브러리

SEED = 42  # 난수 시드. 같은 값이면 항상 같은 데이터가 나온다
N_PRODUCTS = 200  # 만들 상품 수
N_CUSTOMERS = 1000  # 만들 고객 수
IMG_SHAPE = (16, 16, 3)  # 이미지 모양: 세로 16, 가로 16, 색 3(RGB)
START = pd.Timestamp("2024-01-01")  # 주문 기간 시작일
END = pd.Timestamp("2025-12-31")  # 주문 기간 마지막일

# 카테고리별 기준 가격(원)과 이미지 기본 색(RGB)
CATEGORIES = {
    # 전자기기: 비싸고 어두운 색
    "Electronics": {"price": 180000, "color": (90, 100, 120)},
    "Fashion": {"price": 45000, "color": (200, 120, 140)},  # 패션
    "Beauty": {"price": 30000, "color": (230, 180, 200)},  # 뷰티: 밝은 색
    "Home": {"price": 40000, "color": (170, 150, 110)},  # 홈
    "Food": {"price": 15000, "color": (220, 160, 60)},  # 식품: 가장 저렴
    "Books": {"price": 18000, "color": (120, 160, 200)},  # 도서
}

NAME_PARTS = {  # 카테고리별 상품명 재료: (꾸미는 말 목록, 물건 이름 목록)
    "Electronics": (
        ["Wireless", "Smart", "Portable", "Ultra", "Pro"],
        ["Earbuds", "Speaker", "Keyboard", "Mouse", "Charger", "Monitor"],
    ),
    "Fashion": (
        ["Classic", "Slim", "Oversized", "Cotton", "Vintage"],
        ["T-Shirt", "Jeans", "Hoodie", "Jacket", "Sneakers", "Skirt"],
    ),
    "Beauty": (
        ["Hydrating", "Daily", "Vegan", "Mild", "Glow"],
        ["Toner", "Serum", "Sunscreen", "Lip Balm", "Cream", "Cleanser"],
    ),
    "Home": (
        ["Modern", "Compact", "Soft", "Eco", "Large"],
        ["Cushion", "Lamp", "Towel Set", "Storage Box", "Mug", "Blanket"],
    ),
    "Food": (
        ["Organic", "Premium", "Spicy", "Roasted", "Fresh"],
        ["Coffee Beans", "Granola", "Nuts", "Green Tea", "Snack Box", "Olive Oil"],
    ),
    "Books": (
        ["Illustrated", "Essential", "Beginner", "Complete", "Pocket"],
        ["Python Guide", "Novel", "Cookbook", "Travel Essay", "Data Science", "Poetry"],
    ),
}

# 고객 유형: (비율, 평균 추가 주문 수, 선호 카테고리 가중치)
CUSTOMER_TYPES = {
    # 단골: 많이, 자주, 전자기기 선호
    "vip": (0.08, 18, [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]),
    "loyal": (0.20, 8, [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]),  # 충성: 꾸준히 구매
    "regular": (0.32, 3, [0.10, 0.25, 0.20, 0.15, 0.20, 0.10]),  # 일반: 가끔 구매
    "new": (0.15, 0.5, [0.10, 0.25, 0.20, 0.15, 0.20, 0.10]),  # 신규: 최근에 처음 구매
    "churned": (0.25, 1, [0.05, 0.25, 0.20, 0.15, 0.25, 0.10]),  # 이탈: 일찍 떠남
}

# 지역 7곳
REGIONS = ["Seoul", "Gyeonggi", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju"]
REGION_P = [0.30, 0.27, 0.12, 0.10, 0.08, 0.07, 0.06]  # 지역별 뽑힐 확률 (합 1)
PAYMENTS = ["Card", "Kakao Pay", "Naver Pay", "Bank Transfer"]  # 결제 수단 4가지
PAYMENT_P = [0.50, 0.22, 0.20, 0.08]  # 결제 수단별 뽑힐 확률 (합 1)


def make_products(rng: np.random.Generator) -> pd.DataFrame:  # 상품 표를 만든다
    """상품 표와 이미지 배열을 만든다."""
    cat_names = list(CATEGORIES)  # 카테고리 이름 목록
    # 상품마다 카테고리를 무작위로 정한다
    category = rng.choice(cat_names, size=N_PRODUCTS)

    # 상품마다 카테고리 기준 가격을 가져온다
    base_price = np.array([CATEGORIES[c]["price"] for c in category])
    # 기준 가격에 무작위 배수를 곱하고 100원 단위로 반올림한다
    price = np.round(base_price * rng.lognormal(0, 0.45, N_PRODUCTS), -2)

    # 상품명 앞부분(꾸미는 말)
    adjectives = [rng.choice(NAME_PARTS[c][0]) for c in category]
    nouns = [rng.choice(NAME_PARTS[c][1]) for c in category]  # 상품명 중간(물건 이름)
    # 상품명 뒷부분(에디션)
    editions = rng.choice(
        ["", " Mini", " Plus", " Edition 2", " Gift Set"],
        size=N_PRODUCTS,
        p=[0.5, 0.15, 0.15, 0.1, 0.1],
    )
    # 세 부분을 이어 상품명을 만든다
    name = [f"{a} {n}{e}" for a, n, e in zip(adjectives, nouns, editions)]

    # 등록일 범위의 일 수
    days = (pd.Timestamp("2025-10-31") - pd.Timestamp("2023-06-01")).days
    # 범위 안에서 무작위 등록일
    registered = pd.Timestamp("2023-06-01") + pd.to_timedelta(
        rng.integers(0, days, N_PRODUCTS), unit="D"
    )

    # 평균 4.1 근처 평점을 1~5로 자르고 소수 1자리로 맞춘다
    rating = np.clip(rng.normal(4.1, 0.45, N_PRODUCTS), 1, 5).round(1)

    # 이미지: 카테고리 기본색 + 가로 그라데이션 + 상품마다 다른 노이즈
    # 비싼 상품일수록 어둡게 만들어 가격과 밝기 사이에 약한 관계를 둔다
    h, w, _ = IMG_SHAPE  # 세로, 가로 크기를 꺼낸다 (색 개수는 안 쓴다)
    # 카테고리 기본색을 이미지 모양에 맞게 늘린다
    base = np.array([CATEGORIES[c]["color"] for c in category], dtype=float)[
        :, None, None, :
    ]
    # 왼쪽은 어둡고 오른쪽은 밝은 띠
    gradient = (
        np.linspace(-30, 30, w)[None, None, :, None]
        * rng.uniform(0.2, 1.5, N_PRODUCTS)[:, None, None, None]
    )
    log_price = np.log(price)  # 가격 차이가 너무 크니 로그로 줄인다
    # 비쌀수록 더 어둡게 하는 값
    darkness = -18 * (log_price - log_price.mean()) / log_price.std()
    # 상품마다 세기가 다른 잡음
    noise = (
        rng.normal(0, 1, (N_PRODUCTS,) + IMG_SHAPE)
        * rng.uniform(5, 40, N_PRODUCTS)[:, None, None, None]
    )
    # 모두 더해 0~255로 자르고 정수로 바꾼다
    images = np.clip(
        base + gradient + darkness[:, None, None, None] + noise, 0, 255
    ).astype(np.uint8)

    products = pd.DataFrame(
        {  # 상품 표를 만든다
            # 상품 번호: P0001, P0002, ...
            "product_id": [f"P{i:04d}" for i in range(1, N_PRODUCTS + 1)],
            "product_name": name,  # 상품명
            "category": category,  # 카테고리
            "price": price,  # 가격
            "rating": rating,  # 평점
            # 등록일을 문자열로 바꿔 저장한다
            "registered_date": registered.strftime("%Y-%m-%d"),
            # 이미지를 768개 숫자를 공백으로 이은 문자열로 저장한다
            "image_pixels": [" ".join(map(str, img.ravel())) for img in images],
        }
    )

    # 결측치 만들기: 가격 6%, 평점 10%
    # 가격 약 6%를 빈 값으로 만든다
    products.loc[rng.random(N_PRODUCTS) < 0.06, "price"] = np.nan
    # 평점 약 10%를 빈 값으로 만든다
    products.loc[rng.random(N_PRODUCTS) < 0.10, "rating"] = np.nan
    return products  # 상품 표를 돌려준다


# 날짜별 주문 가중치를 정한다
def seasonal_weight(dates: pd.DatetimeIndex) -> np.ndarray:
    """11~12월은 더 많이, 2월은 조금 적게 주문이 들어오게 하는 가중치."""
    month = dates.month  # 날짜에서 월만 꺼낸다
    # 11~12월 1.8배, 2월 0.8배, 나머지 1배
    return np.where(month.isin([11, 12]), 1.8, np.where(month == 2, 0.8, 1.0))


# 주문 표를 만든다
def make_orders(rng: np.random.Generator, products: pd.DataFrame) -> pd.DataFrame:
    """고객 유형별 행동 규칙에 따라 주문 표를 만든다."""
    all_days = pd.date_range(START, END, freq="D")  # 기간 안의 모든 날짜
    weights_all = seasonal_weight(all_days)  # 날짜마다 주문 가중치를 구한다
    cat_names = list(CATEGORIES)  # 카테고리 이름 목록
    product_cat = products["category"].to_numpy()  # 상품별 카테고리 배열

    type_names = list(CUSTOMER_TYPES)  # 고객 유형 이름 목록
    type_p = [CUSTOMER_TYPES[t][0] for t in type_names]  # 유형별 비율
    # 고객마다 유형을 무작위로 정한다
    cust_types = rng.choice(type_names, size=N_CUSTOMERS, p=type_p)

    rows = []  # 주문 한 건씩 모아 둘 목록
    for idx, ctype in enumerate(cust_types):  # 고객을 한 명씩 처리한다
        # 추가 주문 평균과 카테고리 선호도를 꺼낸다
        _, extra_mean, cat_w = CUSTOMER_TYPES[ctype]
        customer_id = f"C{idx + 1:05d}"  # 고객 번호: C00001, C00002, ...
        region = rng.choice(REGIONS, p=REGION_P)  # 고객의 지역을 정한다

        # 고객 유형별 활동 기간(첫 구매일 ~ 마지막 활동일)
        if ctype == "new":  # 신규 고객이면
            # 최근 60일 안에 처음 구매한다
            first = END - pd.Timedelta(days=int(rng.integers(0, 60)))
            last = END  # 마지막 날까지 활동한다
        elif ctype == "churned":  # 이탈 고객이면
            # 앞쪽 시기에 처음 구매한다
            first = START + pd.Timedelta(days=int(rng.integers(0, 420)))
            # 처음 구매 후 120일 안에 떠난다
            last = first + pd.Timedelta(days=int(rng.integers(0, 120)))
        elif ctype == "regular":  # 일반 고객이면
            # 처음 구매일을 정한다
            first = START + pd.Timedelta(days=int(rng.integers(0, 540)))
            # 마지막 구매는 끝나기 30~270일 전이다
            last = END - pd.Timedelta(days=int(rng.integers(30, 270)))
            last = max(last, first)  # 마지막이 처음보다 앞서지 않게 맞춘다
        else:  # vip, loyal이면
            # 첫 해 안에 처음 구매한다
            first = START + pd.Timedelta(days=int(rng.integers(0, 365)))
            # 끝나기 직전까지 활동한다
            last = END - pd.Timedelta(days=int(rng.integers(0, 25)))

        n_orders = 1 + rng.poisson(extra_mean)  # 주문 횟수 = 1회 + 무작위 추가 횟수
        # 활동 기간 안의 날짜만 표시한다
        mask = (all_days >= first) & (all_days <= last)
        days, w = all_days[mask], weights_all[mask]  # 그 기간의 날짜와 가중치
        # 첫 구매일 + 가중치대로 뽑은 나머지 주문일
        dates = (
            [first] + list(rng.choice(days, size=n_orders - 1, p=w / w.sum()))
            if n_orders > 1
            else [first]
        )

        for date in dates:  # 주문일마다 주문 한 건을 만든다
            cat = rng.choice(cat_names, p=cat_w)  # 고객 선호도대로 카테고리를 고른다
            # 그 카테고리 상품 중 하나를 고른다
            pid_idx = rng.choice(np.flatnonzero(product_cat == cat))
            quantity = 1 + rng.poisson(0.35)  # 수량은 대부분 1~2개
            # 대량 주문(이상치): 모든 주문 1.5%, Food/Books는 2% 더
            # 드물게 대량 주문이 되는 경우
            if rng.random() < 0.015 or (
                cat in ("Food", "Books") and rng.random() < 0.02
            ):
                quantity = int(rng.integers(10, 31))  # 10~30개로 크게 만든다
            rows.append(
                {  # 주문 한 건을 목록에 추가한다
                    "customer_id": customer_id,  # 고객 번호
                    "order_date": pd.Timestamp(date).strftime(
                        "%Y-%m-%d"
                    ),  # 주문일(문자열)
                    "product_id": products.loc[pid_idx, "product_id"],  # 상품 번호
                    "quantity": quantity,  # 수량
                    "region": region,  # 지역
                    "payment_method": rng.choice(PAYMENTS, p=PAYMENT_P),  # 결제 수단
                }
            )

    # 표로 만들고 날짜, 고객 순으로 정렬한다
    orders = (
        pd.DataFrame(rows)
        .sort_values(["order_date", "customer_id"])
        .reset_index(drop=True)
    )
    # 맨 앞에 주문 번호(100001부터)를 넣는다
    orders.insert(0, "order_id", np.arange(100001, 100001 + len(orders)))
    return orders  # 주문 표를 돌려준다


def main() -> None:  # 두 CSV 파일을 만든다
    """상품과 주문 CSV를 만들어 data 폴더에 저장한다."""
    rng = np.random.default_rng(SEED)  # 시드를 고정한 난수 생성기
    out_dir = os.path.dirname(os.path.abspath(__file__))  # 이 파일이 있는 data 폴더
    products = make_products(rng)  # 상품 표를 만든다
    orders = make_orders(rng, products)  # 주문 표를 만든다
    # 상품 표를 CSV로 저장한다
    products.to_csv(os.path.join(out_dir, "products.csv"), index=False)
    # 주문 표를 CSV로 저장한다
    orders.to_csv(os.path.join(out_dir, "orders.csv"), index=False)
    # 만든 표의 크기를 출력한다
    print(f"products.csv: {products.shape}, orders.csv: {orders.shape}")


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만
    main()  # CSV 생성을 실행한다
