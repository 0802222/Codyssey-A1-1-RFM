# 분석 데이터 설명

[구현 상세 3.1 Data](implementation.md#31-data)의 상세 내용이다. 데이터 출처와 사용 조건, 샘플을 만든 방법, 컬럼 설명을 정리했다.

<br>
<br>

# 1. 데이터 출처와 사용 조건

- 출처는 Kaggle 대회 [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)이다. H&M의 실제 거래 기록, 상품 정보, 상품 이미지로 이루어져 있다.
- 대회 규칙에 따라 이 과제는 아래처럼 데이터를 다룬다.
    - 7.A: 대회, 포럼, 비상업적인 학술·교육 목적으로만 사용할 수 있다. 이 과제는 교육 과정의 학습용이다.
    - 7.B: 데이터를 복사, 게시, 제3자에게 재배포할 수 없다. 그래서 원본과 샘플 파일을 저장소에 올리지 않고(`.gitignore`), 샘플을 만드는 코드(`data/make_sample.py`)만 올렸다.
- 원본의 규모는 아래와 같다.

원본 파일 3종의 규모

| 파일 | 내용 | 규모 |
|---|---|---|
| `transactions_train.csv` | 거래 (날짜, 고객, 상품, 가격, 판매 채널) | 31,788,324행, 고객 1,362,281명, 2018-09-20 ~ 2020-09-22 |
| `articles.csv` | 상품 정보 25개 컬럼 | 105,542개 |
| `images/앞3자리/상품번호.jpg` | 상품 이미지 | 105,100장 (상품보다 442개 적다) |

<br>
<br>

# 2. 샘플을 만든 방법

원본(약 34GB)은 개인 노트북에서 다루기 어려워 고객 일부만 뽑아 쓴다. [data/make_sample.py](../data/make_sample.py)가 아래 순서로 샘플 3개 파일을 `data/`에 만든다.

샘플 만들기 순서와 사용한 함수

| 순서 | 하는 일 | 함수 | 결과 |
|---|---|---|---|
| 1 | 원본 거래를 100만 행씩 조각으로 읽어 전체 고객 번호를 모으고, 정렬한 뒤 시드 42로 1,000명을 무작위로 고른다 | `pick_customer_ids()` | 고객 1,000명 |
| 2 | 원본 거래를 다시 조각으로 읽으며 고른 고객의 행을 모두 모은다 | `extract_transactions()` | 거래 23,422행 (`transactions_sample.csv`, 2.5MB) |
| 3 | 거래에 나온 상품의 정보만 남긴다. 상품 번호는 이미지 파일 이름에 맞게 앞을 0으로 채워 10자리로 만든다 | `load_articles()` | 상품 13,752개 (`articles_sample.csv`, 4.6MB) |
| 4 | 이미지를 읽어 32x32로 줄이고 하나의 배열로 쌓는다. 파일이 없는 상품은 건너뛴다 | `build_image_arrays()`, `resize_image()` | 이미지 13,717장 (`images_sample.npz`, 18.2MB) |

- 고객 단위로 뽑는 이유: `Recency`, `Frequency`, `Monetary`는 한 고객의 전체 구매 이력으로 계산한다. 거래 행을 무작위로 뽑으면 이력이 끊겨 `RFM`이 틀어진다.
- 고객 수를 1,000명으로 정한 이유: `RFM` 점수는 고객을 5등분해서 매기므로, 1,000명이면 한 구간이 200명이라 결과를 손으로 검산하기 쉽다.
- 시드 42는 난수의 출발값이다. 값 자체에 기술적 의미는 없고 개발자들이 관습적으로 쓰는 숫자다. 중요한 것은 고정했다는 점이다. 고정해야 누가 실행해도 같은 고객 1,000명이 뽑힌다.
- 이미지를 줄이는 방법: 세로와 가로를 같은 간격으로 32칸씩 뽑는다(`np.linspace`로 줄 번호를 만들고 `np.ix_`로 고른다). 흑백이면 같은 값을 3번 쌓아 컬러 모양으로 맞추고, 투명도 채널은 버린다.
- 이미지가 없는 상품 35개(13,752 - 13,717)는 원본에 파일이 없다. 이 상품을 산 거래 86행은 이미지 피처가 비고, [결측치 처리](implementation.md#332-결측치-상품-종류별-중앙값으로-대치)에서 채운다.
- `npz`는 NumPy 배열 여러 개를 한 파일에 압축해 담는 형식이다. 상품 번호 배열(`article_ids`)과 이미지 배열(`images`, 모양 `(13717, 32, 32, 3)`, `uint8`)이 들어 있다. `np.load()`로 읽는다.

<br>
<br>

# 3. 원본에 없는 값을 만든 방법

- 주문 번호: 원본에 주문 번호가 없다. `add_order_key()`가 "고객 번호_거래일"을 이어 붙여 `order_key`를 만든다. 같은 고객이 같은 날 산 행은 주문 1건이다. 23,422행이 7,075건이 된다.
- 수량: 원본에 수량이 없다. 행 하나가 상품 1개의 구매라서 `quantity`를 모두 1로 채운다.
- 구매 금액: `amount = price x quantity`로 만든다. 수량이 모두 1이라 값은 `price`와 같다.
- 가격 단위: 원본 `price`는 이미 0~1 범위로 정규화되어 있어 통화 단위가 없다. 이 과제의 금액은 모두 고객과 상품을 비교하기 위한 상대값이다.
- `Recency` 기준일은 데이터의 마지막 거래일인 2020-09-22다.

<br>
<br>

# 4. 데이터 항목(컬럼) 설명

거래 5개 + 상품 25개 - 합치는 키 1개 + 파생 2개 = 31개 컬럼을 유형별로 묶은 설명표:

<table>
<thead>
<tr><th>유형</th><th>컬럼</th><th>출처 파일</th><th>설명</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">식별자</td><td><code>customer_id</code></td><td>거래</td><td>고객 번호. 원본에서 이미 암호화된 값이다 (1,000명)</td></tr>
<tr><td><code>article_id</code></td><td>거래, 상품</td><td>상품 번호 (13,752개, 두 파일을 합치는 키)</td></tr>
<tr><td>날짜형</td><td><code>order_date</code></td><td>거래</td><td>거래일 (원본 이름 <code>t_dat</code>, <code>Recency</code> 기준). 734개 날짜</td></tr>
<tr><td rowspan="3">수치형</td><td><code>price</code></td><td>거래</td><td>정규화된 가격 (0.0005 ~ 0.3373)</td></tr>
<tr><td><code>quantity</code></td><td>파생</td><td>수량. 모두 1</td></tr>
<tr><td><code>amount</code></td><td>파생</td><td>구매 금액 (<code>price x quantity</code>)</td></tr>
<tr><td rowspan="12">범주형</td><td><code>sales_channel_id</code></td><td>거래</td><td>판매 채널 (2종: 1, 2)</td></tr>
<tr><td><code>product_code</code></td><td>상품</td><td>상품 코드 (8,360종, 같은 상품의 색 변형이 `article_id`로 나뉜다)</td></tr>
<tr><td><code>product_type_no</code>, <code>product_type_name</code></td><td>상품</td><td>상품 종류 번호와 이름 (96종). 결측치 대치의 그룹 기준</td></tr>
<tr><td><code>product_group_name</code></td><td>상품</td><td>상품군 (12종, 예: <code>Garment Upper body</code>)</td></tr>
<tr><td><code>graphical_appearance_no</code>, <code>graphical_appearance_name</code></td><td>상품</td><td>무늬 종류 번호와 이름 (29종)</td></tr>
<tr><td><code>colour_group_code</code>, <code>colour_group_name</code></td><td>상품</td><td>색 번호와 이름 (50종)</td></tr>
<tr><td><code>perceived_colour_value_id</code>, <code>perceived_colour_value_name</code></td><td>상품</td><td>색의 밝기·진하기 구분 번호와 이름 (8종)</td></tr>
<tr><td><code>perceived_colour_master_id</code>, <code>perceived_colour_master_name</code></td><td>상품</td><td>대표색 번호와 이름 (18종)</td></tr>
<tr><td><code>department_no</code>, <code>department_name</code></td><td>상품</td><td>부서 번호(264종)와 이름(218종)</td></tr>
<tr><td><code>index_code</code>, <code>index_name</code></td><td>상품</td><td>매장 구분 코드와 이름 (10종)</td></tr>
<tr><td><code>index_group_no</code>, <code>index_group_name</code></td><td>상품</td><td>매장 대분류 번호와 이름 (5종, 예: <code>Ladieswear</code>)</td></tr>
<tr><td><code>section_no</code>, <code>section_name</code>, <code>garment_group_no</code>, <code>garment_group_name</code></td><td>상품</td><td>섹션(54종)과 의류 그룹(21종)의 번호와 이름</td></tr>
<tr><td rowspan="2">텍스트</td><td><code>product_name</code></td><td>상품</td><td>상품명. 단어 수, 글자 수 피처의 재료 (8,466종)</td></tr>
<tr><td><code>detail_desc</code></td><td>상품</td><td>상품 상세 설명. 72행(0.31%) 비어 있다</td></tr>
<tr><td>이미지 배열</td><td><code>images_sample.npz</code></td><td>별도 파일</td><td>상품 13,717장의 32x32x3 <code>uint8</code> 배열과 상품 번호. 표 컬럼은 아니고 <code>load_images()</code>로 읽는다</td></tr>
</tbody>
</table>

- 파이프라인이 추가로 만드는 컬럼은 이미지 7개(`image_brightness_mean`, `image_brightness_std`, `image_brightness_bin_0~3_ratio`, `image_edge_strength`)와 텍스트 2개(`name_word_count`, `name_length`)이다. 이 컬럼을 모두 합치면 표는 40개 컬럼이 된다.
- 범주형 컬럼에는 같은 뜻의 번호(`*_no`, `*_code`, `*_id`)와 이름(`*_name`)이 짝으로 들어 있다. 분석에서는 읽기 쉬운 이름 컬럼을 쓴다.
