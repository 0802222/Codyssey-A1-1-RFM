# DataAnalyzer 메서드 상세

[README 3.3 분석 클래스 `DataAnalyzer` 구성](../README.md#33-분석-클래스-dataanalyzer-구성)의 상세 내용이다. 메서드별 주요 인자와 사용 예시를 정리했다. 코드는 [src/pipeline.py](../src/pipeline.py)에 있다.

<br>
<br>

# 1. 메서드 목록

`DataAnalyzer`의 메서드를 단계별로 정리한 메서드 목록표:

<table>
<thead>
<tr><th>단계</th><th>메서드</th><th>주요 인자</th><th>하는 일</th></tr>
</thead>
<tbody>
<tr><td>로드</td><td><code>load_data()</code></td><td><code>date_cols</code>, <code>merge_key</code></td><td>주문 CSV와 상품 CSV를 읽고 합친다</td></tr>
<tr><td rowspan="2">결측치</td><td><code>missing_report()</code></td><td></td><td>컬럼별 결측 개수, 비율</td></tr>
<tr><td><code>handle_missing_values()</code></td><td><code>strategy</code>, <code>group_col</code>, <code>columns</code></td><td><code>group_median</code>, <code>group_mean</code>, <code>mean</code>, <code>median</code>, <code>mode</code>, <code>drop</code></td></tr>
<tr><td rowspan="3">이상치</td><td><code>get_iqr_bounds()</code></td><td><code>column</code>, <code>threshold=1.5</code></td><td><code>IQR</code> 정상 범위 계산</td></tr>
<tr><td><code>detect_outliers()</code></td><td><code>column</code>, <code>threshold=1.5</code></td><td>범위 밖 행 반환</td></tr>
<tr><td><code>handle_outliers()</code></td><td><code>column</code>, <code>threshold</code>, <code>method</code></td><td><code>clip</code>(경계값으로 바꾸기) 또는 <code>remove</code>(삭제)</td></tr>
<tr><td rowspan="3">이미지</td><td><code>parse_images()</code></td><td><code>image_shape</code>, <code>downsample_step</code></td><td>문자열 → <code>(N, H, W, C)</code> 배열, <code>arr[:, ::step, ::step]</code> 다운샘플링</td></tr>
<tr><td><code>add_image_features()</code></td><td><code>image_shape</code>, <code>downsample_step</code></td><td><code>image_brightness_mean</code>, <code>image_brightness_std</code></td></tr>
<tr><td><code>add_image_advanced_features()</code></td><td><code>n_brightness_bins</code></td><td>밝기 히스토그램 <code>image_brightness_bin_*_ratio</code>, 엣지 강도 <code>image_edge_strength</code> (보너스 1)</td></tr>
<tr><td>텍스트</td><td><code>add_text_features()</code></td><td><code>text_col</code></td><td><code>name_word_count</code>, <code>name_length</code></td></tr>
<tr><td>파생</td><td><code>add_amount()</code></td><td><code>price_col</code>, <code>quantity_col</code></td><td><code>amount = price x quantity</code></td></tr>
<tr><td rowspan="2">통계</td><td><code>summary_statistics()</code></td><td><code>columns</code></td><td>평균, 중앙값, 표준편차, 최소, <code>Q1</code>, <code>Q3</code>, 최대, <code>IQR</code></td></tr>
<tr><td><code>correlation()</code></td><td><code>columns</code>, <code>method</code></td><td>상관계수 행렬</td></tr>
<tr><td rowspan="3">RFM</td><td><code>calculate_rfm()</code></td><td><code>customer_col</code>, <code>date_col</code>, <code>amount_col</code>, <code>reference_date</code>, <code>order_col</code>, <code>n_score_levels</code></td><td><code>R</code>, <code>F</code>, <code>M</code> 값과 1~5점</td></tr>
<tr><td><code>segment_customers()</code></td><td></td><td>점수 규칙으로 6개 세그먼트</td></tr>
<tr><td><code>segment_summary()</code></td><td></td><td>세그먼트별 고객 수, 평균 <code>R</code>/<code>F</code>/<code>M</code>, 매출 비중</td></tr>
<tr><td>시계열</td><td><code>monthly_sales()</code></td><td><code>date_col</code>, <code>amount_col</code></td><td>월별 매출</td></tr>
<tr><td>코호트</td><td><code>cohort_retention()</code></td><td><code>customer_col</code>, <code>date_col</code></td><td>첫 구매 월별 재구매율 (보너스 2)</td></tr>
</tbody>
</table>

<br>
<br>

# 2. 사용 예시

`DataAnalyzer`를 처음부터 끝까지 호출하는 사용 예시 코드:

```python
from src.pipeline import DataAnalyzer  # 분석 클래스를 가져온다

analyzer = DataAnalyzer("data/orders.csv", product_path="data/products.csv")  # 주문, 상품 파일 경로를 넣어 객체를 만든다
analyzer.load_data(date_cols=["order_date", "registered_date"])  # 두 파일을 읽고 합친다 (날짜 컬럼은 날짜형으로 변환)
analyzer.handle_missing_values(strategy="group_median", group_col="category", columns=["price", "rating"])  # 빈 값을 카테고리별 중앙값으로 채운다
analyzer.add_amount("price", "quantity")  # 주문 금액(amount = 가격 x 수량) 컬럼을 만든다
analyzer.add_image_features("image_pixels", image_shape=(16, 16, 3))  # 이미지 문자열에서 밝기 평균, 표준편차 컬럼을 만든다
analyzer.add_text_features("product_name")  # 상품명에서 단어 수, 글자 수 컬럼을 만든다
outliers = analyzer.detect_outliers("amount", threshold=1.5)  # 주문 금액의 이상치 행을 찾는다
analyzer.calculate_rfm("customer_id", "order_date", "amount", order_col="order_id")  # 고객별 R, F, M 값과 1~5점을 계산한다
analyzer.segment_customers()  # 점수 규칙으로 고객을 6개 세그먼트로 나눈다
print(analyzer.segment_summary())  # 세그먼트별 요약표를 출력한다
```
