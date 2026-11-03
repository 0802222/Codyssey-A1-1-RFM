# 분석 데이터 설명

[README 3.2 분석에 사용한 데이터](../README.md#32-분석에-사용한-데이터)의 상세 내용이다. 컬럼 설명과 합성 데이터를 만든 규칙을 정리했다.

<br>
<br>

# 1. 데이터 항목(컬럼) 설명

주문 파일과 상품 파일을 합친 표의 항목 13개를 유형별로 묶은 설명표:

<table>
<thead>
<tr><th>유형</th><th>컬럼</th><th>출처 파일</th><th>설명</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">식별자</td><td><code>order_id</code></td><td>주문</td><td>주문 번호</td></tr>
<tr><td><code>customer_id</code></td><td>주문</td><td>고객 번호 (<code>RFM</code> 기준)</td></tr>
<tr><td><code>product_id</code></td><td>주문, 상품</td><td>상품 번호 (두 파일을 합치는 키)</td></tr>
<tr><td rowspan="2">날짜형</td><td><code>order_date</code></td><td>주문</td><td>주문일 (<code>Recency</code> 기준)</td></tr>
<tr><td><code>registered_date</code></td><td>상품</td><td>상품 등록일</td></tr>
<tr><td rowspan="3">수치형</td><td><code>quantity</code></td><td>주문</td><td>주문 수량</td></tr>
<tr><td><code>price</code></td><td>상품</td><td>상품 가격(원). 결측치 7.2%</td></tr>
<tr><td><code>rating</code></td><td>상품</td><td>상품 평점(1~5). 결측치 11.0%</td></tr>
<tr><td rowspan="3">범주형</td><td><code>region</code></td><td>주문</td><td>고객 지역 (7개)</td></tr>
<tr><td><code>payment_method</code></td><td>주문</td><td>결제 수단 (4개)</td></tr>
<tr><td><code>category</code></td><td>상품</td><td>상품 카테고리 (6개)</td></tr>
<tr><td>텍스트</td><td><code>product_name</code></td><td>상품</td><td>상품명</td></tr>
<tr><td>이미지 배열</td><td><code>image_pixels</code></td><td>상품</td><td>16x16x3 = 768개 픽셀 값을 공백으로 이은 문자열</td></tr>
</tbody>
</table>

- 주문 금액 `amount`는 파이프라인에서 `price x quantity`로 만든다.
- `Recency` 기준일은 데이터의 마지막 거래일인 2025-12-31이다.

<br>
<br>

# 2. 합성 데이터를 만든 규칙

- 고객은 5가지 행동 유형(VIP형, 충성형, 일반형, 신규형, 이탈형)으로 만들고, 유형마다 활동 기간과 주문 횟수를 다르게 줬다.
- 11\~12월에는 주문 확률을 1.8배로 높였다.
- 전체 주문의 약 1.5%(`Food`, `Books`는 추가 2%)를 10\~30개 대량 주문으로 만들어 이상치를 넣었다.
- 이미지는 카테고리별 기본색 + 그라데이션 + 노이즈로 만들고, 비싼 상품일수록 어둡게 했다.
- 분석 결과의 패턴은 이 규칙에서 나온 것이다. 파이프라인이 숨겨진 패턴을 제대로 찾아내는지 확인하는 용도이며, 실제 데이터에 적용할 때는 다시 검증해야 한다.
