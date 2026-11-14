# 실행 가이드

[README 1.4 실행 방법](../README.md#14-실행-방법)에서 다루지 않은 상세 내용이다. 시연 스크립트의 단계별 동작과, 한 단계씩 직접 실행하는 방법을 정리했다.

<br>
<br>

# 1. 시연 스크립트 `demo.sh`의 단계

`demo.sh`가 순서대로 하는 일을 정리한 표:

| 단계 | 하는 일 |
|---|---|
| 0 | 실행 환경과 라이브러리 버전 출력 |
| 1 | 금지 라이브러리 `import` 여부 검사 (`grep`) |
| 2 | 단위 테스트 43개 실행 |
| 3 | 합성 데이터 생성 (시드 42) |
| 4 | `src/pipeline.py` 실행, 결과를 터미널에 출력 |
| 5 | 노트북 재실행, 차트를 `docs/figures/`에 저장 (`--no-nb`면 생략) |

- `./demo.sh --no-nb`는 5단계(노트북 재실행)만 생략한다.

<br>
<br>

# 2. 단계별 직접 실행

`demo.sh`를 쓰지 않고 한 단계씩 직접 실행하는 명령어:

```bash
.venv/bin/python -m data.generate_data                 # 데이터 생성 (data/orders.csv, data/products.csv)
.venv/bin/python -m unittest discover -s tests -v      # 단위 테스트
.venv/bin/python -m src.pipeline                       # 파이프라인 실행, 결과를 터미널에 출력
cd notebooks && ../.venv/bin/python -m nbconvert --to notebook --execute --inplace analysis_report.ipynb
```

<br>
<br>

# 3. 코딩 컨벤션 검사

개발 환경(다른 PC 포함)에서 코드를 고친 뒤 실행하는 검사 도구 설치와 실행 명령어. 대상은 `src`, `tests`, `data/generate_data.py`이다.

```bash
# 검사 도구 설치 (처음 한 번)
.venv/bin/pip install -r requirements-dev.txt

# 코드 모양 검사 (--check를 빼면 자동 수정)
.venv/bin/black --check src tests data/generate_data.py

# import 순서 검사
.venv/bin/isort --check src tests data/generate_data.py

# PEP8, 미사용 코드, docstring 검사
.venv/bin/ruff check src tests data/generate_data.py

# 타입 힌트 검사
.venv/bin/mypy src tests data/generate_data.py
```

<br>
<br>

# 4. 환경 참고

- `Python 3.10` 이상에서 동작한다. (개발 환경: `Python 3.11.4`, macOS)
- `Python 3.8`은 수명 종료(EOL)라 VS Code Jupyter 확장이 지원하지 않는다. 3.8로 만든 `.venv`는 지우고 다시 만든다.
- 터미널, VS Code, 노트북이 서로 다른 `Python`을 쓰면 `No module named ...` 오류가 난다. 하나로 맞춘다.
