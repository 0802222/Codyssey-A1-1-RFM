"""README 개념 설명용 그림 5장을 만든다.

- coupon_problem.png: 같은 쿠폰을 VIP와 Churned에게 보낼 때의 문제.
- iqr_definition.png: 구매 금액 15건으로 보는 IQR의 정의.
- iqr_normal.png: 정규분포와 이 과제 구매 금액 분포에서 IQR 이상치 범위를 비교한다.
- vector_concept.png: 이미지 평균을 구하는 for 반복문과 벡터 연산의 방식 차이.
- numpy_vs_for.png: for 반복문과 벡터 연산의 계산 시간을 비교한다.

실행: python -m src.concept_figures (샘플 데이터 파일이 있어야 한다)
"""

# 파일 경로를 다루는 기본 모듈
import os

# 데이터 묶음 클래스를 쉽게 만드는 도구
from dataclasses import dataclass

# 이름 있는 묶음을 만드는 도구
from typing import NamedTuple

# 그림을 그리는 라이브러리
import matplotlib.pyplot as plt

# 배열 계산 라이브러리
import numpy as np

# 표 처리 라이브러리
import pandas as pd

# 둥근 상자를 그리는 도구
from matplotlib.patches import FancyBboxPatch, Rectangle

# 공통 상수: 데이터 파일 경로와 프로젝트 폴더
from src.constants import (
    ARTICLES_PATH,
    IMAGES_PATH,
    ROOT_DIR,
    TRANSACTIONS_PATH,
)

# 분석 클래스
from src.pipeline import IQR_THRESHOLD, DataAnalyzer

# 그림을 저장할 폴더
FIGURE_DIR = os.path.join(ROOT_DIR, "docs", "figures")

# 한글이 깨지지 않게 쓰는 글꼴 (맥 기본 글꼴)
FONT_NAME = "Apple SD Gothic Neo"

# 정규분포 예시로 뽑는 값의 개수
NORMAL_SAMPLE_COUNT = 100_000

# 막대 그래프의 구간 수
HISTOGRAM_BIN_COUNT = 60

# 난수 시드 (실행할 때마다 같은 그림이 나오게 고정)
RANDOM_SEED = 42

# 노트북 2장에서 잰 시간(ms): 이미지 13,717장의 밝기 평균 계산
FOR_LOOP_MILLISECONDS = 91.9
VECTOR_MILLISECONDS = 13.2

# 벡터 연산 그림에 보여줄 실제 이미지 개수
VECTOR_EXAMPLE_COUNT = 3

# 벡터 연산 그림에 쓸 상품 종류와, 그중 살펴볼 후보 이미지 수
EXAMPLE_PRODUCT_TYPE = "Trousers"
EXAMPLE_CANDIDATE_COUNT = 50

# 그림 속 이미지 한 장의 가로, 세로 길이(그림판 좌표)
THUMBNAIL_SIZE = 1.1

# 상자의 아래 위치와 높이(그림판 좌표)
BOX_BOTTOM = 0.7
BOX_HEIGHT = 1.9

# 상자 안쪽 색과 구분선 색
COLOR_BOX_FILL = "#eef5fb"
COLOR_DIVIDER = "#bbbbbb"

# IQR 정의 그림에 쓸 실제 구매 금액 표본: 뽑는 개수와 난수 시드
IQR_EXAMPLE_COUNT = 15
IQR_EXAMPLE_SEED = 14

# 상자그림 배치 위치 (점 기둥, 상자, 수염 끝 막대, 오른쪽 괄호)
BOXPLOT_DOT_X = 0.2
BOXPLOT_BOX_LEFT = 0.3
BOXPLOT_BOX_WIDTH = 0.2
BOXPLOT_CENTER_X = BOXPLOT_BOX_LEFT + BOXPLOT_BOX_WIDTH / 2
BOXPLOT_CAP_HALF_WIDTH = 0.05
BRACKET_X = 1.02

# 축 선 색과 맨 아래 설명 글 색
COLOR_AXIS = "#888888"
COLOR_CAPTION = "#555555"

# 쿠폰 그림용 색: 쿠폰, 이탈 고객 카드 안쪽
COLOR_COUPON = "#f2a33a"
COLOR_OUTLIER_FILL = "#fdeceb"

# 색상: 정상 범위, 가운데 50%(IQR), 이상치
COLOR_NORMAL = "#9ec5e8"
COLOR_IQR = "#2f6fb0"
COLOR_OUTLIER = "#e0524a"


# 값 목록으로 히스토그램과 가로 박스플롯을 한 칸에 그린다
def draw_iqr_panel(
    histogram_axis: plt.Axes,
    box_axis: plt.Axes,
    values: np.ndarray,
    title: str,
    value_range: tuple[float, float],
) -> None:
    # 1사분위수(Q1)와 3사분위수(Q3)를 구한다
    quartile_1, quartile_3 = np.percentile(values, [25, 75])

    # IQR = Q3 - Q1
    iqr = quartile_3 - quartile_1

    # 정상 범위의 하한과 상한
    lower_bound = quartile_1 - IQR_THRESHOLD * iqr
    upper_bound = quartile_3 + IQR_THRESHOLD * iqr

    # 정상 범위를 벗어난 값의 비율(%)
    outlier_percent = float(np.mean((values < lower_bound) | (values > upper_bound)))
    outlier_percent *= 100

    # 구간별 개수를 센다
    counts, edges = np.histogram(values, bins=HISTOGRAM_BIN_COUNT, range=value_range)

    # 각 막대의 가운데 위치
    centers = (edges[:-1] + edges[1:]) / 2

    # 막대 색: 이상치는 빨강, 가운데 50%는 진한 파랑, 나머지는 연한 파랑
    colors = np.where(
        # 정상 범위 밖이면 빨강
        (centers < lower_bound) | (centers > upper_bound),
        COLOR_OUTLIER,
        # 정상 범위 안이면 Q1~Q3 사이는 진한 파랑, 아니면 연한 파랑
        np.where(
            (centers >= quartile_1) & (centers <= quartile_3), COLOR_IQR, COLOR_NORMAL
        ),
    )

    # 막대를 그린다
    histogram_axis.bar(centers, counts, width=edges[1] - edges[0], color=colors)

    # 정상 범위 경계선을 점선으로 긋는다
    for bound in (lower_bound, upper_bound):
        # 경계선 한 개
        histogram_axis.axvline(
            # 선 위치
            bound,
            # 선 색
            color=COLOR_OUTLIER,
            # 점선
            linestyle="--",
            # 선 두께
            linewidth=1.5,
        )

    # 제목에 이상치 비율을 같이 쓴다
    histogram_axis.set_title(f"{title}\n이상치 {outlier_percent:.1f}%", fontsize=13)

    # y축 눈금은 읽을 필요가 없어서 숨긴다
    histogram_axis.set_yticks([])

    # x축 눈금은 아래 박스플롯 칸에서 보여준다
    histogram_axis.set_xticklabels([])

    # 박스플롯: 상자 = 가운데 50%(IQR), 수염 끝 = 정상 범위 끝
    box_axis.boxplot(
        # 그릴 값 목록
        values,
        # 가로 방향으로 그린다
        orientation="horizontal",
        # 수염 길이는 IQR의 1.5배까지
        whis=IQR_THRESHOLD,
        # 상자 두께
        widths=0.6,
        # 이상치 점은 작고 흐린 빨강 원
        flierprops={
            # 점 모양
            "marker": "o",
            # 점 크기
            "markersize": 2,
            # 점 안쪽 색
            "markerfacecolor": COLOR_OUTLIER,
            # 점 테두리 색
            "markeredgecolor": COLOR_OUTLIER,
            # 투명도
            "alpha": 0.3,
        },
        # 상자 테두리
        boxprops={"color": COLOR_IQR, "linewidth": 2},
        # 중앙값 선
        medianprops={"color": COLOR_IQR, "linewidth": 2},
    )

    # 박스플롯 칸의 x축 범위를 위 히스토그램과 맞춘다
    box_axis.set_xlim(value_range)
    histogram_axis.set_xlim(value_range)

    # y축 눈금은 숨긴다
    box_axis.set_yticks([])

    # 하한, 상한 숫자를 박스플롯 아래에 쓴다
    box_axis.set_xlabel(f"정상 범위 {lower_bound:.4g} ~ {upper_bound:.4g}")


# 그림판 위에 놓을 이름표 하나의 정보
class MarkerLabel(NamedTuple):
    """그림판 위 이름표 한 개의 정보."""

    # 이름표 가로 위치(금액)
    position: float
    # 이름표 글
    text: str
    # 글자 색
    color: str
    # 굵게 할지
    is_bold: bool


# 실제 구매 금액 15건으로 IQR의 정의를 상자그림(박스플롯) 모양으로 보여준다
def make_iqr_definition_figure(amounts: np.ndarray) -> None:
    # 난수 생성기(Random Number Generator)로 구매 금액 15건을 뽑는다
    rng = np.random.default_rng(IQR_EXAMPLE_SEED)
    sample = np.sort(rng.choice(amounts, IQR_EXAMPLE_COUNT, replace=False))

    # 점을 흩뿌릴 가로 위치 (겹치지 않게 살짝 흔든다)
    dot_x_positions = BOXPLOT_DOT_X + rng.uniform(-0.05, 0.05, IQR_EXAMPLE_COUNT)

    # 1사분위수(Q1), 중앙값, 3사분위수(Q3)
    quartile_1, median, quartile_3 = np.percentile(sample, [25, 50, 75])

    # IQR = Q3 - Q1
    iqr = quartile_3 - quartile_1

    # 정상 범위의 하한과 상한 (울타리)
    lower_fence = quartile_1 - IQR_THRESHOLD * iqr
    upper_fence = quartile_3 + IQR_THRESHOLD * iqr

    # 울타리 안에 있는 값 중 가장 작은 값과 가장 큰 값 (수염 끝)
    inside_values = sample[(sample >= lower_fence) & (sample <= upper_fence)]
    whisker_low = float(inside_values.min())
    whisker_high = float(inside_values.max())

    # 울타리 밖의 값 (이상치)
    outlier_values = sample[(sample < lower_fence) | (sample > upper_fence)]

    # 그림 위아래 끝 (울타리와 값 전체를 담고 여유를 둔다)
    lowest_value = min(lower_fence, float(sample.min()))
    highest_value = max(upper_fence, float(sample.max()))
    value_span = highest_value - lowest_value
    axis_bottom = lowest_value - 0.08 * value_span
    axis_top = highest_value + 0.08 * value_span

    # 세로로 긴 그림판을 만들고 눈금은 숨긴다
    figure, canvas = plt.subplots(figsize=(5.5, 8))
    canvas.axis("off")

    # 그림판 좌표: 가로는 그림 배치용, 세로는 실제 구매 금액
    canvas.set_xlim(-0.05, 1.45)
    canvas.set_ylim(axis_bottom, axis_top)

    # 가운데 상자: Q1부터 Q3까지 (상자 높이 = IQR)
    canvas.add_patch(
        Rectangle(
            # 상자 왼쪽 아래 모서리
            (BOXPLOT_BOX_LEFT, quartile_1),
            # 상자 가로 길이
            BOXPLOT_BOX_WIDTH,
            # 상자 세로 길이
            iqr,
            # 상자 안쪽 색
            facecolor=COLOR_BOX_FILL,
            # 상자 테두리 색
            edgecolor=COLOR_IQR,
            # 테두리 두께
            linewidth=2,
            # 상자가 점보다 앞에 보이게 한다
            zorder=2,
        )
    )

    # 상자 안의 중앙값 선
    canvas.plot(
        [BOXPLOT_BOX_LEFT, BOXPLOT_BOX_LEFT + BOXPLOT_BOX_WIDTH],
        [median, median],
        color=COLOR_IQR,
        linewidth=2.5,
        zorder=3,
    )

    # 상자 위(Q3)에서 위 수염 끝까지 세로선을 긋는다
    canvas.plot(
        [BOXPLOT_CENTER_X, BOXPLOT_CENTER_X],
        [quartile_3, whisker_high],
        color=COLOR_IQR,
        linewidth=1.5,
    )

    # 상자 아래(Q1)에서 아래 수염 끝까지 세로선을 긋는다
    canvas.plot(
        [BOXPLOT_CENTER_X, BOXPLOT_CENTER_X],
        [quartile_1, whisker_low],
        color=COLOR_IQR,
        linewidth=1.5,
    )

    # 수염 끝의 가로 막대 (위, 아래)
    for cap_value in (whisker_low, whisker_high):
        # 수염 끝 막대 한 개
        canvas.plot(
            [
                BOXPLOT_CENTER_X - BOXPLOT_CAP_HALF_WIDTH,
                BOXPLOT_CENTER_X + BOXPLOT_CAP_HALF_WIDTH,
            ],
            [cap_value, cap_value],
            color=COLOR_IQR,
            linewidth=1.5,
        )

    # 울타리 두 개를 점선으로 긋는다 (상한은 빨강, 하한은 파랑)
    for fence_value, fence_color in (
        (upper_fence, COLOR_OUTLIER),
        (lower_fence, COLOR_IQR),
    ):
        # 울타리 점선 한 개
        canvas.plot(
            [0.0, 1.0],
            [fence_value, fence_value],
            color=fence_color,
            linestyle="--",
            linewidth=1.2,
        )

    # 울타리 밖 이상치는 빨간 원으로 찍는다
    canvas.scatter(
        [BOXPLOT_CENTER_X] * len(outlier_values),
        outlier_values,
        s=90,
        color=COLOR_OUTLIER,
        zorder=4,
    )

    # 울타리 안의 값은 검은 점으로 찍는다
    canvas.scatter(
        dot_x_positions[(sample >= lower_fence) & (sample <= upper_fence)],
        inside_values,
        s=60,
        color="black",
        zorder=4,
    )

    # 이상치 옆에 값을 쓴다
    for outlier_value in outlier_values:
        # 이상치 글 한 개
        draw_label(
            canvas,
            BOXPLOT_CENTER_X + 0.06,
            float(outlier_value),
            f"이상치 {outlier_value:.4f}",
            11,
            COLOR_OUTLIER,
            True,
            "left",
        )

    # 울타리 이름표 (선 바로 위 왼쪽에 쓴다)
    draw_label(
        canvas,
        -0.03,
        upper_fence + 0.03 * value_span,
        f"Q3 + 1.5 x IQR = {upper_fence:.4f}",
        11,
        COLOR_OUTLIER,
        True,
    )

    # 하한 울타리 이름표 (선 바로 아래 왼쪽에 쓴다)
    draw_label(
        canvas,
        -0.03,
        lower_fence - 0.03 * value_span,
        f"Q1 - 1.5 x IQR = {lower_fence:.4f}",
        11,
        COLOR_IQR,
        True,
    )

    # 오른쪽 가로선 이름표: Q3, 중앙값, Q1
    for label_value, label_text, label_bold in (
        (quartile_3, "Q3 (75% 지점)", True),
        (median, "중앙값", True),
        (quartile_1, "Q1 (25% 지점)", True),
    ):
        # 이름표 한 개 (값은 바로 아래 줄에 쓴다)
        draw_label(
            canvas,
            BOXPLOT_BOX_LEFT + BOXPLOT_BOX_WIDTH + 0.03,
            label_value,
            f"{label_text} {label_value:.4f}",
            10,
            COLOR_IQR,
            label_bold,
            "left",
        )

    # 오른쪽 괄호: IQR (Q1~Q3), 1.5배 폭 (Q3~상한, 하한~Q1)
    for bracket_low, bracket_high, bracket_text in (
        (quartile_1, quartile_3, "IQR"),
        (quartile_3, upper_fence, "1.5 x IQR"),
        (lower_fence, quartile_1, "1.5 x IQR"),
    ):
        # 괄호 모양 선 (위아래 끝에 짧은 가로선이 달린 세로선)
        canvas.plot(
            [BRACKET_X - 0.02, BRACKET_X, BRACKET_X, BRACKET_X - 0.02],
            [bracket_low, bracket_low, bracket_high, bracket_high],
            color="#666666",
            linewidth=1,
        )

        # 괄호 가운데에 이름을 쓴다
        draw_label(
            canvas,
            BRACKET_X + 0.03,
            (bracket_low + bracket_high) / 2,
            bracket_text,
            11,
            "#444444",
            False,
            "left",
        )

    # 그림 맨 아래 설명
    caption = f"구매 금액 {IQR_EXAMPLE_COUNT}건의 상자그림 (정상 범위 밖이 이상치)"
    draw_label(
        canvas,
        0.5,
        axis_bottom - 0.03 * value_span,
        caption,
        11,
        COLOR_CAPTION,
        False,
        "center",
    )

    # 파일로 저장한다
    figure.savefig(
        # 저장 경로
        os.path.join(FIGURE_DIR, "iqr_definition.png"),
        # 해상도
        dpi=110,
        # 여백은 최소로
        bbox_inches="tight",
    )

    # 메모리를 돌려준다
    plt.close(figure)


# 정규분포와 실제 구매 금액을 나란히 비교하는 그림을 만든다
def make_iqr_figure(amounts: np.ndarray) -> None:
    # 난수 생성기(Random Number Generator)
    rng = np.random.default_rng(RANDOM_SEED)

    # 평균 0, 표준편차 1인 정규분포에서 값을 뽑는다
    normal_values = rng.normal(0, 1, NORMAL_SAMPLE_COUNT)

    # 왼쪽 오른쪽 두 칸, 각 칸 아래에 박스플롯 칸을 둔다
    figure, axes = plt.subplots(
        # 2행
        2,
        # 2열
        2,
        # 그림 크기(가로, 세로 인치)
        figsize=(12, 5.2),
        # 윗줄(히스토그램)을 아랫줄(박스플롯)보다 4배 높게, 줄 사이 간격은 좁게
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.12},
    )

    # 왼쪽: 정규분포
    draw_iqr_panel(
        # 히스토그램 칸
        axes[0][0],
        # 박스플롯 칸
        axes[1][0],
        # 그릴 값
        normal_values,
        # 제목
        "정규분포 (좌우 대칭)",
        # x축 범위
        (-4, 4),
    )

    # 오른쪽: 이 과제의 구매 금액
    draw_iqr_panel(
        # 히스토그램 칸
        axes[0][1],
        # 박스플롯 칸
        axes[1][1],
        # 그릴 값
        amounts,
        # 제목
        "이 과제의 구매 금액 (오른쪽 꼬리가 김)",
        # x축 범위는 0부터 최대 금액까지
        (0, float(amounts.max())),
    )

    # 범례 대신 색 설명을 맨 위에 쓴다
    figure.text(
        # 가로 위치(그림 가운데)
        0.5,
        # 세로 위치(그림 위쪽 바깥)
        1.02,
        # 색 설명 글
        "진한 파랑 = 가운데 50% (IQR)    연한 파랑 = 정상 범위    빨강 = 이상치",
        # 가운데 정렬
        ha="center",
        # 글자 크기
        fontsize=12,
    )

    # 파일로 저장한다
    figure.savefig(
        # 저장 경로
        os.path.join(FIGURE_DIR, "iqr_normal.png"),
        # 해상도
        dpi=110,
        # 여백은 최소로
        bbox_inches="tight",
    )

    # 메모리를 돌려준다
    plt.close(figure)


# 이미지 한 장을 그림판 위 정해진 자리에 그린다
def draw_thumbnail(
    canvas: plt.Axes, image: np.ndarray, left: float, bottom: float
) -> None:
    # 이미지를 정해진 네모 칸(extent)에 맞춰 그린다
    canvas.imshow(
        # 32x32x3 이미지 배열
        image,
        # 칸의 왼쪽, 오른쪽, 아래, 위 위치
        extent=(left, left + THUMBNAIL_SIZE, bottom, bottom + THUMBNAIL_SIZE),
        # 그림 위에 보이게 한다
        zorder=2,
    )


# 화살표를 그린다 (arrow_style을 "<->"로 하면 양쪽 끝에 화살촉이 생긴다)
def draw_arrow(
    canvas: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    arrow_style: str = "-|>",
) -> None:
    # 빈 글자에 화살표만 붙여 그린다
    canvas.annotate(
        # 글자는 쓰지 않는다
        "",
        # 화살표 끝
        xy=end,
        # 화살표 시작
        xytext=start,
        # 화살표 모양
        arrowprops={"arrowstyle": arrow_style, "color": color, "linewidth": 2},
    )


# 정한 상품 종류에서 어두운 편, 중간, 밝은 편 이미지 3장을 고른다
def pick_example_images(
    article_ids: np.ndarray, images: np.ndarray, trousers_article_ids: np.ndarray
) -> np.ndarray:
    # 정한 상품 종류의 이미지만 앞에서부터 후보로 모은다
    is_candidate = np.isin(article_ids, trousers_article_ids)
    candidates = images[is_candidate][:EXAMPLE_CANDIDATE_COUNT]

    # 후보의 밝기 평균을 어두운 순서로 줄 세운다
    sorted_positions = np.argsort(candidates.mean(axis=(1, 2, 3)))

    # 하위 20%, 가운데, 상위 20% 지점의 위치 (극단값은 알아보기 어려워서 피한다)
    candidate_count = len(sorted_positions)
    picked_positions = sorted_positions[
        [candidate_count // 5, candidate_count // 2, candidate_count * 4 // 5]
    ]

    # 고른 이미지를 돌려준다
    return candidates[picked_positions]


# 글자를 그림판에 쓴다
def draw_label(
    canvas: plt.Axes,
    x_position: float,
    y_position: float,
    text: str,
    fontsize: int = 12,
    color: str = "black",
    is_bold: bool = False,
    align: str = "left",
    font_family: str = FONT_NAME,
) -> None:
    # 글자를 정해진 위치에 쓴다
    canvas.text(
        # 가로 위치
        x_position,
        # 세로 위치
        y_position,
        # 쓸 글
        text,
        # 글자 크기
        fontsize=fontsize,
        # 글자 색
        color=color,
        # 굵게 할지
        fontweight="bold" if is_bold else "normal",
        # 가로 정렬
        ha=align,
        # 세로 가운데 정렬
        va="center",
        # 글꼴 종류
        fontfamily=font_family,
    )


# 둥근 상자를 그린다 (따로 정하지 않으면 파란 상자)
def draw_box(
    canvas: plt.Axes,
    left: float,
    width: float,
    bottom: float = BOX_BOTTOM,
    height: float = BOX_HEIGHT,
    edge_color: str = COLOR_IQR,
    fill_color: str = COLOR_BOX_FILL,
) -> None:
    # 둥근 상자를 그림판에 올린다
    canvas.add_patch(
        FancyBboxPatch(
            # 상자 왼쪽 아래
            (left, bottom),
            # 상자 가로, 세로
            width,
            height,
            # 둥근 모서리
            boxstyle="round,pad=0.05,rounding_size=0.15",
            # 테두리 색
            edgecolor=edge_color,
            # 안쪽 색
            facecolor=fill_color,
            # 테두리 두께
            linewidth=2,
        )
    )


# 세그먼트 카드 한 장에 쓸 요약 값
@dataclass
class SegmentCard:
    """세그먼트 카드 한 장에 쓸 요약 값."""

    # 세그먼트 이름
    name: str
    # 평균 마지막 구매 후 일 수
    recency_days: float
    # 평균 주문 횟수
    order_count: float
    # 고객 1명당 평균 구매액
    spending_per_customer: float
    # 전체 매출 중 이 세그먼트의 비중(%)
    revenue_share_percent: float


# 세그먼트 카드를 그리고 카드 아래 설명 상자도 함께 그린다
def draw_segment_card(
    canvas: plt.Axes,
    left: float,
    card: SegmentCard,
    edge_color: str,
    fill_color: str,
    verdict_lines: tuple[str, str],
) -> None:
    # 카드 가운데 가로 위치
    center = left + 1.95

    # 카드 상자
    draw_box(canvas, left, 3.9, 2.4, 3.6, edge_color, fill_color)

    # 카드 제목(세그먼트 이름)
    draw_label(canvas, center, 5.45, card.name, 20, edge_color, True, "center")

    # 카드 안 지표 4줄
    draw_label(
        canvas, left + 0.4, 4.75, f"마지막 구매  {card.recency_days:.0f}일 전", 14
    )
    draw_label(canvas, left + 0.4, 4.15, f"평균 주문  {card.order_count:.1f}회", 14)
    draw_label(
        canvas, left + 0.4, 3.55, f"1인당 구매액  {card.spending_per_customer:.3f}", 14
    )
    draw_label(
        canvas, left + 0.4, 2.95, f"매출 비중  {card.revenue_share_percent:.1f}%", 14
    )

    # 카드에서 설명 상자로 가는 화살표
    draw_arrow(canvas, (center, 2.3), (center, 1.85), edge_color)

    # 설명 상자
    draw_box(canvas, left, 3.9, 0.4, 1.4, edge_color, "white")

    # 설명 두 줄
    draw_label(canvas, center, 1.2, verdict_lines[0], 12, "black", False, "center")
    draw_label(canvas, center, 0.75, verdict_lines[1], 13, edge_color, True, "center")


# 같은 쿠폰을 VIP와 Churned에게 보낼 때의 문제를 그림으로 보여준다
def make_coupon_figure(vip: SegmentCard, churned: SegmentCard) -> None:
    # 그림판 한 장을 만들고 눈금은 숨긴다
    figure, canvas = plt.subplots(figsize=(12, 5.4))
    canvas.axis("off")

    # 그림판 좌표 범위: 가로 0~12, 세로 0~6.4
    canvas.set_xlim(0, 12)
    canvas.set_ylim(0, 6.4)

    # 왼쪽 카드: VIP
    draw_segment_card(
        canvas,
        0.2,
        vip,
        COLOR_IQR,
        COLOR_BOX_FILL,
        ("쿠폰 없이도 이미 사는 고객", "할인 비용만 나간다"),
    )

    # 오른쪽 카드: Churned
    draw_segment_card(
        canvas,
        7.9,
        churned,
        COLOR_OUTLIER,
        COLOR_OUTLIER_FILL,
        ("오래 안 온 고객 (복귀 가치 낮음)", "쿠폰 비용만 나간다"),
    )

    # 가운데 쿠폰 상자
    draw_box(canvas, 4.7, 2.6, 3.2, 1.4, COLOR_COUPON, "#fff4e0")
    draw_label(canvas, 6.0, 3.9, "같은 쿠폰", 17, COLOR_COUPON, True, "center")

    # 쿠폰에서 두 카드로 가는 화살표
    draw_arrow(canvas, (4.7, 3.9), (4.2, 3.9), COLOR_COUPON)
    draw_arrow(canvas, (7.3, 3.9), (7.8, 3.9), COLOR_COUPON)

    # 가운데 아래: 구매액이 몇 배 차이 나는지
    spending_ratio = vip.spending_per_customer / churned.spending_per_customer
    draw_label(
        canvas,
        6.0,
        1.9,
        f"1인당 구매액 약 {spending_ratio:.0f}배 차이",
        14,
        "black",
        True,
        "center",
    )
    draw_label(
        canvas, 6.0, 1.4, "그룹을 나눠 다르게 대해야 한다", 13, "black", False, "center"
    )

    # 파일로 저장한다
    figure.savefig(
        # 저장 경로
        os.path.join(FIGURE_DIR, "coupon_problem.png"),
        # 해상도
        dpi=110,
        # 여백은 최소로
        bbox_inches="tight",
    )

    # 메모리를 돌려준다
    plt.close(figure)


# 세그먼트별 평균 지표를 카드 값으로 만든다
def build_segment_card(rfm: pd.DataFrame, segment_name: str) -> SegmentCard:
    # 이 세그먼트의 고객만 고른다
    members = rfm[rfm["segment"] == segment_name]

    # 평균 지표와 매출 비중을 계산해 카드로 돌려준다
    return SegmentCard(
        # 세그먼트 이름
        name=segment_name,
        # 평균 마지막 구매 후 일 수
        recency_days=float(members["recency"].mean()),
        # 평균 주문 횟수
        order_count=float(members["frequency"].mean()),
        # 고객 1명당 평균 구매액
        spending_per_customer=float(members["monetary"].mean()),
        # 전체 매출 중 이 세그먼트가 차지하는 비중(%)
        revenue_share_percent=float(
            members["monetary"].sum() / rfm["monetary"].sum() * 100
        ),
    )


# 이미지 평균을 for 반복문과 벡터 연산으로 구하는 방식을 그림으로 보여준다
def make_vector_concept_figure(examples: np.ndarray) -> None:
    # 그 이미지들의 밝기 평균(실제 계산 결과)
    brightness_means = examples.mean(axis=(1, 2, 3))

    # 그림판 한 장을 만들고 눈금은 숨긴다
    figure, canvas = plt.subplots(figsize=(12, 5.6))
    canvas.axis("off")

    # 그림판 좌표 범위: 가로 0~12, 세로 0~7
    canvas.set_xlim(0, 12)
    canvas.set_ylim(0, 7)

    # 윗줄 제목: for 반복문
    draw_label(
        canvas,
        0.1,
        6.7,
        "for 반복문: 이미지를 한 장씩 꺼내 계산",
        14,
        COLOR_OUTLIER,
        True,
    )

    # 윗줄: 이미지 -> 값 쌍을 그린다
    for example_number in range(VECTOR_EXAMPLE_COUNT):
        # 이 쌍이 시작하는 가로 위치
        left = 0.3 + 3.3 * example_number

        # 몇 번째 계산인지 쓴다
        draw_label(canvas, left, 5.95, f"{example_number + 1}번째", 11)

        # 이미지를 그린다
        draw_thumbnail(canvas, examples[example_number], left, 4.5)

        # 이미지에서 값으로 가는 화살표
        draw_arrow(canvas, (left + 1.25, 5.05), (left + 1.9, 5.05), COLOR_OUTLIER)

        # 계산된 밝기 평균
        draw_label(
            canvas, left + 2.0, 5.05, f"{brightness_means[example_number]:.1f}", 13
        )

    # 윗줄 끝: 계속 반복한다는 표시
    draw_label(canvas, 10.4, 5.05, "... x 13,717번", 13, COLOR_OUTLIER, True)

    # 두 줄 사이에 점선을 긋는다
    canvas.plot([0.1, 11.9], [3.7, 3.7], color=COLOR_DIVIDER, linestyle=":")

    # 아랫줄 제목: 벡터 연산
    draw_label(
        canvas, 0.1, 3.2, "벡터 연산: 배열 전체를 한 번에 계산", 14, COLOR_IQR, True
    )

    # 이미지 묶음 전체를 감싸는 상자
    draw_box(canvas, 0.2, 4.3)

    # 상자 안에 이미지를 가로로 늘어놓는다
    for example_number in range(VECTOR_EXAMPLE_COUNT):
        # 이미지 한 장
        draw_thumbnail(
            canvas, examples[example_number], 0.5 + 1.2 * example_number, 1.35
        )

    # 이미지가 더 있다는 표시
    draw_label(canvas, 4.2, 1.9, "...", 18, align="center")

    # 상자 설명: 배열 모양
    draw_label(canvas, 0.5, 0.95, "images 배열 모양 (13717, 32, 32, 3)", 11)

    # 상자에서 결과로 가는 화살표
    draw_arrow(canvas, (4.8, 1.65), (8.0, 1.65), COLOR_IQR)

    # 화살표 위 설명: 코드 한 줄
    draw_label(
        canvas,
        6.4,
        2.0,
        "images.mean(axis=(1, 2, 3))",
        10,
        COLOR_IQR,
        False,
        "center",
        "monospace",
    )

    # 화살표 아래 설명
    draw_label(canvas, 6.4, 1.15, "한 번에", 12, COLOR_IQR, True, "center")

    # 결과 상자
    draw_box(canvas, 8.3, 3.6)

    # 결과 상자 안에 쓸 값 목록 글
    result_text = ", ".join(f"{value:.1f}" for value in brightness_means)

    # 결과 상자 안: 이미지마다 값 1개가 한꺼번에 나온다
    draw_label(canvas, 10.1, 1.95, f"[{result_text}, ...]", 12, align="center")
    draw_label(canvas, 10.1, 1.25, "값 13,717개", 11, align="center")

    # 파일로 저장한다
    figure.savefig(
        # 저장 경로
        os.path.join(FIGURE_DIR, "vector_concept.png"),
        # 해상도
        dpi=110,
        # 여백은 최소로
        bbox_inches="tight",
    )

    # 메모리를 돌려준다
    plt.close(figure)


# for 반복문과 벡터 연산의 계산 시간을 막대로 비교한다
def make_speed_figure() -> None:
    # 그림 한 장을 만든다
    figure, axis = plt.subplots(figsize=(8, 2.8))

    # 막대에 쓸 이름과 시간과 색
    labels = [
        # 위 막대 이름
        "for 반복문\n(이미지를 한 장씩 13,717번)",
        # 아래 막대 이름
        "벡터 연산\n(배열 전체를 한 번에)",
    ]

    # 막대 길이(ms)
    milliseconds = [FOR_LOOP_MILLISECONDS, VECTOR_MILLISECONDS]

    # 막대 색
    bar_colors = [COLOR_OUTLIER, COLOR_IQR]

    # 가로 막대를 그린다
    bars = axis.barh(labels, milliseconds, color=bar_colors, height=0.55)

    # 막대 끝에 시간을 쓴다
    axis.bar_label(bars, labels=[f"{value} ms" for value in milliseconds], padding=6)

    # 몇 배 차이인지 쓴다
    speed_ratio = FOR_LOOP_MILLISECONDS / VECTOR_MILLISECONDS
    axis.set_title(
        # 제목 글
        f"밝기 평균 계산 시간: 벡터 연산이 약 {speed_ratio:.0f}배 빠르다",
        # 글자 크기
        fontsize=13,
    )

    # 막대 이름이 위에서부터 for, 벡터 순서가 되게 뒤집는다
    axis.invert_yaxis()

    # 오른쪽 여백을 두어 글자가 잘리지 않게 한다
    axis.set_xlim(0, FOR_LOOP_MILLISECONDS * 1.25)

    # x축 이름
    axis.set_xlabel("계산 시간 (ms, 짧을수록 좋다)")

    # 위쪽, 오른쪽 테두리를 지운다
    axis.spines[["top", "right"]].set_visible(False)

    # 파일로 저장한다
    figure.savefig(
        # 저장 경로
        os.path.join(FIGURE_DIR, "numpy_vs_for.png"),
        # 해상도
        dpi=110,
        # 여백은 최소로
        bbox_inches="tight",
    )

    # 메모리를 돌려준다
    plt.close(figure)


# 그림 5장을 모두 만든다
def main() -> None:
    # 한글 글꼴을 쓴다
    plt.rcParams["font.family"] = FONT_NAME

    # 마이너스 기호가 깨지지 않게 한다
    plt.rcParams["axes.unicode_minus"] = False

    # 저장 폴더가 없으면 만든다
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 분석 클래스로 구매 금액을 만든다
    analyzer = DataAnalyzer(TRANSACTIONS_PATH, ARTICLES_PATH, IMAGES_PATH)
    analyzer.load_data(date_cols=["order_date"])

    # 구매 금액 컬럼을 만든다
    analyzer.add_amount()

    # 주문 키를 만들고 RFM 점수와 세그먼트를 계산한다
    analyzer.add_order_key()
    analyzer.calculate_rfm("customer_id", "order_date", "amount", order_col="order_key")
    rfm = analyzer.segment_customers()

    # 이미지 배열을 읽는다
    article_ids, images = analyzer.load_images()

    # 상품 종류가 정한 종류(바지)인 상품 번호를 모은다
    is_example_type = analyzer.df["product_type_name"] == EXAMPLE_PRODUCT_TYPE
    example_article_ids = analyzer.df.loc[is_example_type, "article_id"].astype(str)

    # 벡터 연산 그림에 쓸 이미지 3장을 고른다
    example_images = pick_example_images(
        # 상품 번호 배열
        article_ids,
        # 이미지 배열
        images,
        # 정한 종류의 상품 번호
        example_article_ids.unique(),
    )

    # 그림 5장을 만든다
    make_iqr_definition_figure(analyzer.df["amount"].to_numpy())
    make_coupon_figure(
        build_segment_card(rfm, "VIP"), build_segment_card(rfm, "Churned")
    )
    make_iqr_figure(analyzer.df["amount"].to_numpy())
    make_vector_concept_figure(example_images)
    make_speed_figure()


# 파일을 직접 실행했을 때만 main을 부른다
if __name__ == "__main__":
    main()
