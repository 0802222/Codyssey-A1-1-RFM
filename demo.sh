#!/usr/bin/env bash
# 시연 스크립트: 실행만 하면 과제 결과를 순서대로 출력한다.
# 사용법: ./demo.sh            (전체)
#         ./demo.sh --no-nb    (노트북 재실행 생략)

# 에러가 나면 바로 멈추고, 없는 변수를 쓰면 에러로 본다
set -euo pipefail

# 이 스크립트가 있는 폴더(프로젝트 최상위)로 이동한다
cd "$(dirname "$0")"

# .venv가 있으면 그 Python을, 없으면 python3를 쓴다
# 가상환경의 Python이 실행 가능하면
if [ -x ".venv/bin/python" ]; then
    # 가상환경의 Python을 쓴다
    PY=".venv/bin/python"
# 없으면
else
    # 시스템 python3를 쓴다
    PY="python3"
fi

# 단계 제목을 보기 좋게 출력하는 함수
step() {
    # 첫 번째 인자를 제목으로 출력한다
    printf '\n========== %s ==========\n' "$1"
}

# 0단계 제목 출력
step "0. 실행 환경"

# Python과 라이브러리 버전을 출력한다
"$PY" -c "import sys, numpy, pandas, matplotlib, seaborn; print('Python', sys.version.split()[0]); print('NumPy', numpy.__version__, '/ Pandas', pandas.__version__, '/ Matplotlib', matplotlib.__version__, '/ Seaborn', seaborn.__version__)"

# 1단계 제목 출력
step "1. 금지 라이브러리 사용 여부 검사"

# 코드에서 금지 라이브러리 import를 찾는다
if grep -rnE "(^|\")\s*(import|from)\s+(cv2|PIL|sklearn|nltk|ydata_profiling|pandas_profiling|sweetviz)" src data tests notebooks --include=*.py --include=*.ipynb; then
    # 찾았으면 알려주고
    echo "금지 라이브러리가 발견됐다"

    # 실패로 끝낸다
    exit 1
# 못 찾았으면
else
    # 통과했다고 알려준다
    echo "OpenCV, Pillow, Scikit-learn, NLTK, 자동 EDA 라이브러리 사용 없음"
fi

# 2단계 제목 출력
step "2. 단위 테스트"

# 테스트를 모두 돌리고 마지막 40줄만 보여준다
"$PY" -m unittest discover -s tests -v 2>&1 | tail -n 40

# 3단계 제목 출력
step "3. 데이터 생성 (고정 시드 42)"

# 합성 데이터 CSV를 만든다
"$PY" -m data.generate_data

# 4단계 제목 출력
step "4. 분석 파이프라인 실행 (src/pipeline.py)"

# 전체 분석 흐름을 실행하고 결과를 출력한다
"$PY" -m src.pipeline

# --no-nb 옵션이 없으면 노트북도 실행한다
if [ "${1:-}" != "--no-nb" ]; then
    # 5단계 제목 출력
    step "5. 분석 노트북 재실행"

    # 노트북 실행 도구가 설치돼 있으면
    if "$PY" -c "import nbconvert" 2>/dev/null; then
        # notebooks 폴더에서 노트북을 처음부터 끝까지 실행해 결과를 저장한다
        (cd notebooks && "../$PY" -m nbconvert --to notebook --execute --inplace analysis_report.ipynb)

        # 끝났다고 알려준다
        echo "notebooks/analysis_report.ipynb 실행 완료, 차트는 docs/figures/ 에 저장됨"

        # 저장된 차트 파일 목록을 보여준다
        ls docs/figures
    # 도구가 없으면
    else
        # 생략 이유를 알려준다
        echo "nbconvert가 없어 생략한다. pip install -r requirements.txt 후 다시 실행"
    fi
fi

# 마지막 제목 출력
step "완료"
