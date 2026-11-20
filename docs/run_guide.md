# 실행 가이드

[README 1.4 실행 방법](../README.md#14-실행-방법)에서 다루지 않은 상세 내용이다. 샘플 데이터 준비, 시연 스크립트의 단계별 동작, 한 단계씩 직접 실행하는 방법을 정리했다.

<br>
<br>

# 1. 데이터 준비

대회 규칙상 원본과 샘플을 저장소에 올리지 않았으므로, 분석을 실행하려면 샘플 3개 파일(`data/transactions_sample.csv`, `data/articles_sample.csv`, `data/images_sample.npz`)을 먼저 만든다. ([데이터 설명](data_description.md#1-데이터-출처와-사용-조건))

샘플을 만드는 두 가지 방법

| 방법 | 순서 | 비고 |
|---|---|---|
| 내 컴퓨터에서 만들기 | 1. Kaggle 계정으로 대회 규칙에 동의한다<br>2. 원본을 `data/raw/`에 내려받아 푼다 (아래 명령어)<br>3. `python -m data.make_sample --raw-dir data/raw`를 실행한다 | 원본이 약 34GB라 디스크 여유와 시간이 필요하다. 샘플 생성은 거래 파일을 조각으로 두 번 읽는다 |
| Kaggle 노트북에서 만들기 | 1. 새 Kaggle 노트북에 대회 데이터를 입력(Input)으로 추가한다<br>2. 코드(`src/`, `data/make_sample.py`, `data/__init__.py`)를 공개 Dataset으로 올려 입력에 추가한다 (아래 2단계)<br>3. 첫 셀로 코드를 작업 폴더에 복사하고 `data.make_sample`을 실행한다 (아래 코드) | 원본을 내려받지 않아도 된다. 만들어진 샘플로 같은 노트북 안에서 분석까지 실행한다 |

```bash
# 원본 내려받기 (kaggle CLI 설치와 API 토큰 설정, 대회 규칙 동의가 먼저 필요하다)
.venv/bin/kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw

# 압축 풀기 (transactions_train.csv, articles.csv, images/ 폴더가 나온다)
unzip data/raw/h-and-m-personalized-fashion-recommendations.zip -d data/raw

# 샘플 만들기 (결과: data/ 아래 파일 3개, 개수를 터미널에 출력한다)
.venv/bin/python -m data.make_sample --raw-dir data/raw
```

Kaggle 노트북에서 만들 때의 코드 준비와 첫 셀

```bash
# 1. 내 컴퓨터에서 코드만 묶는다 (프로젝트 최상위 폴더에서 실행)
zip -r hm-src.zip src data/make_sample.py data/__init__.py -x "*/__pycache__/*"

# 2. Kaggle의 Datasets에서 New Dataset으로 hm-src.zip을 올리고 공개(Public)로 설정한다
#    비공개면 다른 사람이 노트북을 다시 실행할 수 없다
```

```python
# 코드를 작업 폴더로 복사하고 샘플을 만드는 첫 셀 (Dataset 경로는 `/kaggle/input` 아래 `datasets/<사용자>/<Dataset 이름>` 형태이다. 본인 계정과 이름으로 바꾼다)
import os
import shutil

# 작업 폴더, 코드 Dataset 폴더, 대회 데이터 폴더
WORK = "/kaggle/working"
CODE = "/kaggle/input/datasets/chod5ng/hm-src-zip"
RAW = "/kaggle/input/competitions/h-and-m-personalized-fashion-recommendations"

# 입력 폴더는 읽기 전용이라 src와 data를 작업 폴더로 복사한다
shutil.copytree(f"{CODE}/src", f"{WORK}/src", dirs_exist_ok=True)
shutil.copytree(f"{CODE}/data", f"{WORK}/data", dirs_exist_ok=True)

# 노트북의 ROOT(= 한 단계 위 폴더)가 작업 폴더가 되도록 notebooks 폴더로 이동한다
os.makedirs(f"{WORK}/notebooks", exist_ok=True)
os.chdir(f"{WORK}/notebooks")

# 샘플 3개 파일을 만든다 (결과: /kaggle/working/data)
os.system(f"cd {WORK} && python -m data.make_sample --raw-dir {RAW}")
```

- `kaggle` 명령은 `requirements.txt`에 없다. Kaggle 공식 CLI 패키지를 따로 설치해야 하고, API 토큰 파일은 저장소에 넣지 않는다.
- 위 첫 셀과 노트북 전체 실행(Run All)을 Kaggle에서 확인했다. 샘플 생성 출력이 로컬과 같고 에러 없이 끝났으며, 노트북은 공개로 저장했다.
- 샘플 생성 결과는 고객 1,000명, 거래 23,422건, 상품 13,752개(이미지 13,717장)이다. 시드가 고정이라 같은 원본이면 항상 같은 결과가 나온다.

<br>
<br>

# 2. 시연 스크립트 `demo.sh`의 단계

`demo.sh`가 순서대로 하는 일을 정리한 표. 샘플 3개 파일이 있어야 3단계 이후가 동작한다.

| 단계 | 하는 일 |
|---|---|
| 0 | 실행 환경과 라이브러리 버전 출력 |
| 1 | 금지 라이브러리 `import` 여부 검사 (`grep`) |
| 2 | 단위 테스트 실행 |
| 3 | `src/pipeline.py` 실행, 결과를 터미널에 출력 |
| 4 | 노트북 재실행, 차트를 `docs/figures/`에 저장 (`--no-nb`면 생략) |

- `./demo.sh --no-nb`는 4단계(노트북 재실행)만 생략한다.

<br>
<br>

# 3. 단계별 직접 실행

`demo.sh`를 쓰지 않고 한 단계씩 직접 실행하는 명령어:

```bash
.venv/bin/python -m data.make_sample --raw-dir data/raw   # 샘플 만들기 (data/*_sample.*)
.venv/bin/python -m unittest discover -s tests -v          # 단위 테스트
.venv/bin/python -m src.pipeline                           # 파이프라인 실행, 결과를 터미널에 출력
cd notebooks && ../.venv/bin/python -m nbconvert --to notebook --execute --inplace analysis_report.ipynb
```

<br>
<br>

# 4. 코딩 컨벤션 검사

개발 환경(다른 PC 포함)에서 코드를 고친 뒤 실행하는 검사 도구 설치와 실행 명령어. 대상은 `src`, `tests`, `data/make_sample.py`이다.

```bash
# 검사 도구 설치 (처음 한 번)
.venv/bin/pip install -r requirements-dev.txt

# 코드 모양 검사 (--check를 빼면 자동 수정)
.venv/bin/black --check src tests data/make_sample.py

# import 순서 검사
.venv/bin/isort --check src tests data/make_sample.py

# PEP8, 미사용 코드, docstring 검사
.venv/bin/ruff check src tests data/make_sample.py

# 타입 힌트 검사
.venv/bin/mypy src tests data/make_sample.py
```

<br>
<br>

# 5. 환경 참고

- `Python 3.10` 이상에서 동작한다. (개발 환경: `Python 3.11.4`, macOS)
- `Python 3.8`은 수명 종료(EOL)라 VS Code Jupyter 확장이 지원하지 않는다. 3.8로 만든 `.venv`는 지우고 다시 만든다.
- 터미널, VS Code, 노트북이 서로 다른 `Python`을 쓰면 `No module named ...` 오류가 난다. 하나로 맞춘다.
