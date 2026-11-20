# DataAnalyzer 메서드 상세

[README 3.2 분석 클래스 `DataAnalyzer` 구성](../README.md#32-분석-클래스-dataanalyzer-구성)의 상세 내용이다. 메서드별 주요 인자와 사용 예시를 정리했다. 코드는 [src/pipeline.py](../src/pipeline.py)에 있다.

<br>
<br>

# 1. 메서드 목록

`DataAnalyzer`의 공개 메서드 19개를 단계별로 정리한 목록표. 객체를 만들 때 거래 파일 경로(`data_path`), 상품 파일 경로(`product_path`), 이미지 파일 경로(`image_path`)를 받는다.

<table>
<thead>
<tr><th>단계</th><th>메서드</th><th>주요 인자</th><th>하는 일</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">로드</td><td><code>load_data()</code></td><td><code>date_cols</code>, <code>merge_key</code>, <code>column_names</code></td><td>거래 CSV와 상품 CSV를 읽고 <code>article_id</code>로 합친다. 컬럼 이름을 바꾸고(<code>t_dat</code> → <code>order_date</code>), 수량이 없으면 <code>quantity</code>를 1로 채운다</td></tr>
<tr><td><code>add_order_key()</code></td><td><code>customer_col</code>, <code>date_col</code>, <code>output_col</code></td><td>"고객 번호_거래일"로 주문 키 <code>order_key</code>를 만든다 (원본에 주문 번호가 없어서)</td></tr>
<tr><td rowspan="2">결측치</td><td><code>missing_report()</code></td><td></td><td>컬럼별 결측 개수, 비율</td></tr>
<tr><td><code>handle_missing_values()</code></td><td><code>strategy</code>, <code>group_col</code>, <code>columns</code></td><td><code>group_median</code>, <code>group_mean</code>, <code>mean</code>, <code>median</code>, <code>mode</code>, <code>drop</code></td></tr>
<tr><td rowspan="3">이상치</td><td><code>get_iqr_bounds()</code></td><td><code>column</code>, <code>threshold=1.5</code></td><td><code>IQR</code> 정상 범위 계산</td></tr>
<tr><td><code>detect_outliers()</code></td><td><code>column</code>, <code>threshold=1.5</code></td><td>범위 밖 행 반환</td></tr>
<tr><td><code>handle_outliers()</code></td><td><code>column</code>, <code>threshold</code>, <code>method</code></td><td><code>clip</code>(경계값으로 바꾸기) 또는 <code>remove</code>(삭제)</td></tr>
<tr><td rowspan="3">이미지</td><td><code>load_images()</code></td><td><code>image_path</code></td><td><code>npz</code> 파일에서 상품 번호 배열과 <code>(N, 32, 32, 3)</code> 이미지 배열을 읽어 객체에 저장한다</td></tr>
<tr><td><code>add_image_features()</code></td><td></td><td><code>image_brightness_mean</code>, <code>image_brightness_std</code>. 상품별로 한 번 계산해 거래 행에 붙인다. <code>load_images()</code>를 먼저 실행해야 한다</td></tr>
<tr><td><code>add_image_advanced_features()</code></td><td><code>n_brightness_bins</code></td><td>밝기 히스토그램 <code>image_brightness_bin_*_ratio</code>, 엣지 강도 <code>image_edge_strength</code> (보너스 1)</td></tr>
<tr><td>텍스트</td><td><code>add_text_features()</code></td><td><code>text_col</code></td><td><code>name_word_count</code>, <code>name_length</code></td></tr>
<tr><td>파생</td><td><code>add_amount()</code></td><td><code>price_col</code>, <code>quantity_col</code>, <code>output_col</code></td><td><code>amount = price x quantity</code></td></tr>
<tr><td rowspan="2">통계</td><td><code>summary_statistics()</code></td><td><code>columns</code></td><td>평균, 중앙값, 표준편차, 최소, <code>Q1</code>, <code>Q3</code>, 최대, <code>IQR</code></td></tr>
<tr><td><code>correlation()</code></td><td><code>columns</code>, <code>method</code></td><td>상관계수 행렬</td></tr>
<tr><td rowspan="3">RFM</td><td><code>calculate_rfm()</code></td><td><code>customer_col</code>, <code>date_col</code>, <code>amount_col</code>, <code>reference_date</code>, <code>order_col</code>, <code>n_score_levels</code></td><td><code>R</code>, <code>F</code>, <code>M</code> 값과 1~5점. <code>order_col</code>을 주면 그 컬럼의 고유 개수로 주문 횟수를 센다</td></tr>
<tr><td><code>segment_customers()</code></td><td></td><td>점수 규칙으로 6개 세그먼트</td></tr>
<tr><td><code>segment_summary()</code></td><td></td><td>세그먼트별 고객 수, 평균 <code>R</code>/<code>F</code>/<code>M</code>, 매출 비중</td></tr>
<tr><td>시계열</td><td><code>monthly_sales()</code></td><td><code>date_col</code>, <code>amount_col</code></td><td>월별 매출</td></tr>
<tr><td>코호트</td><td><code>cohort_retention()</code></td><td><code>customer_col</code>, <code>date_col</code></td><td>첫 구매 월별 재구매율 (보너스 2)</td></tr>
</tbody>
</table>

- 이 외에 `_require_images()`(이미지를 읽었는지 확인)와 `_attach_article_features()`(상품별 피처를 거래 행에 붙임)가 내부에서 쓰인다. 이미지가 없는 상품의 행에는 `NaN`이 붙는다.

<br>
<br>

# 2. 사용 예시

`DataAnalyzer`를 처음부터 끝까지 호출하는 사용 예시 코드. 샘플 3개 파일이 `data/`에 있어야 한다. ([실행 가이드](run_guide.md#1-데이터-준비))

```python
# 분석 클래스와 샘플 파일 경로를 가져온다
from src.constants import ARTICLES_PATH, IMAGES_PATH, TRANSACTIONS_PATH
from src.pipeline import DataAnalyzer

# 거래, 상품, 이미지 파일 경로를 넣어 객체를 만든다
analyzer = DataAnalyzer(TRANSACTIONS_PATH, product_path=ARTICLES_PATH, image_path=IMAGES_PATH)

# 두 CSV를 읽고 합친다 (거래일은 날짜형으로 변환)
analyzer.load_data(date_cols=["order_date"])

# 구매 금액(amount = 가격 x 수량) 컬럼을 만든다
analyzer.add_amount("price", "quantity")

# 이미지 파일을 읽고 밝기 평균, 표준편차 컬럼을 만든다
analyzer.load_images()
analyzer.add_image_features()

# 상품명에서 단어 수, 글자 수 컬럼을 만든다
analyzer.add_text_features("product_name")

# 이미지가 없는 상품의 빈 값을 상품 종류별 중앙값으로 채운다
analyzer.handle_missing_values(
    strategy="group_median",
    group_col="product_type_name",
    columns=["image_brightness_mean", "image_brightness_std"],
)

# 구매 금액의 이상치 행을 찾는다
outliers = analyzer.detect_outliers("amount", threshold=1.5)

# 같은 고객이 같은 날 산 행을 주문 1건으로 묶는 주문 키를 만든다
analyzer.add_order_key("customer_id", "order_date")

# 고객별 R, F, M 값과 1~5점을 계산한다
analyzer.calculate_rfm("customer_id", "order_date", "amount", order_col="order_key")

# 점수 규칙으로 고객을 6개 세그먼트로 나눈다
analyzer.segment_customers()

# 세그먼트별 요약표를 출력한다
print(analyzer.segment_summary())
```
