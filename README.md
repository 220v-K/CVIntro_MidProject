# CIFAR-10 KNN 실험 프로젝트

## 개요
이 프로젝트는 CIFAR-10 이미지 분류 문제를 대상으로 K-최근접 이웃(K-Nearest Neighbors, KNN) 분류기의 여러 평가 전략을 비교 분석합니다. `train/test` 단일 분할, `train/valid/test` 3-way 분할, `5-fold` 교차 검증 세 가지 실험 규칙을 동일한 하이퍼파라미터 후보 집합(k ∈ {1,3,5,7,9}, 거리 지표 L1/L2)으로 수행하고, 성능 지표(정확도·정밀도·재현율·F1)를 수집해 시각화 및 로그로 남깁니다.

## 폴더 구조
```text
CVIntro_MidProject/
├─ cifar-10-batches-py/          # CIFAR-10 원본 배치 파일 (python 버전)
├─ main.py                       # 실험 진입점, 공용 유틸 및 데이터 로더
├─ train_test_split_only.py      # train/test 단일 분할 KNN 실험
├─ train_valid_test_split.py     # train/valid/test 분할 + 최적 하이퍼파라미터로 test 평가
├─ five_fold_cross_validation.py # 5-fold Stratified KFold 기반 실험
├─ plot.py                       # 성능 지표 시각화 및 결과 디렉터리 관리
├─ results/                      # 실행 시 생성되는 결과 로그 및 그래프
└─ README.md
```

## 요구 사항
- Python 3.10 이상 권장
- 필수 패키지: `numpy`, `scikit-learn`, `pandas`, `tqdm`, `matplotlib`
  - 가상환경 사용 시 예: `pip install numpy scikit-learn pandas tqdm matplotlib`
- `cifar-10-batches-py/` 디렉터리에 CIFAR-10 Python 버전 배치 파일이 존재해야 합니다. (현재 리포지토리에 포함되어 있다면 추가 다운로드가 필요 없습니다.)

## 실행 방법
1. (선택) 가상환경을 준비하고 의존성을 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy scikit-learn pandas tqdm matplotlib
   ```
2. 원하는 실험 규칙을 `--classifier` 인자로 지정하여 실행합니다.
   ```bash
   python main.py --classifier train_test       # train/test 단일 분할
   python main.py --classifier train_valid_test # train/valid/test 분할
   python main.py --classifier 5-fold           # 5-fold 교차 검증
   ```
3. 실행 시 각 실험은 모든 k(1,3,5,7,9)와 거리 지표(L1, L2)에 대해 모델을 학습하고 평가합니다. 진행 상황은 `tqdm` 프로그레스바로 표시됩니다.

## 실험 모드 요약
| 인자 값 | 설명 | 검증 전략 | 최적 하이퍼파라미터로 테스트 평가 |
|---------|------|-----------|-------------------------------------|
| `train_test` | 학습 데이터 전체로 학습 후 테스트 데이터로 바로 평가 | 단일 분할 | 즉시 테스트 평가 |
| `train_valid_test` | 학습 데이터를 90% 학습·10% 검증으로 분리 | 고정 검증 세트 | 검증 최고 조합을 다시 학습 후 테스트 평가 |
| `5-fold` | StratifiedKFold(5)으로 5회 교차 검증 후 평균 성능 계산 | 교차 검증 | 검증 최고 조합을 전체 학습 데이터에 재학습 후 테스트 평가 |

## 주요 스크립트 및 함수
- `main.py`
  - `load_data()` / `load_data_train_valid_test()` : CIFAR-10 배치를 불러와 학습·검증·테스트 세트를 구성합니다.
  - `run_knn(case)` : 실험 모드 선택에 따라 각 분할 전략을 호출하고, 결과 저장 및 시각화를 담당합니다.
  - `save_results_to_txt()` : 성능 지표를 CSV 형태 텍스트와 최고 성능 요약, 테스트 결과를 함께 기록합니다.
- `train_test_split_only.py`
  - `knn_classifier_train_test_split_only()` : 지정된 k, 거리 지표에 대한 KNN을 학습·예측하고 지표를 반환합니다.
- `train_valid_test_split.py`
  - `knn_classifier_train_valid_test_split()` : 검증 세트를 이용해 하이퍼파라미터를 탐색합니다.
  - `evaluate_with_best_k_and_distance_metric()` : 최고 성능 조합을 다시 학습해 테스트 세트로 평가합니다.
- `five_fold_cross_validation.py`
  - `knn_classifier_5_fold_cross_validation()` : `StratifiedKFold`로 각 fold 성능을 평균 냅니다.
- `plot.py`
  - `plot_results()` : k 값에 따른 정확도, 정밀도, 재현율, F1 점수를 각 거리 지표별로 그래프로 저장합니다.
  - `ensure_results_dir()` : 결과 저장용 디렉터리 자동 생성.

## 결과 확인
- 각 실행 모드별 결과는 `results/<실험_이름>/` 하위에 저장됩니다.
  - `knn_results.txt` : k와 거리 지표별 정량 지표, 최고 성능 요약, 테스트 세트 평가(해당되는 경우).
  - `knn_l1_distance.png`, `knn_l2_distance.png` : 거리 지표별 성능 곡선 그래프.
- 동일한 디렉터리를 여러 번 실행하면 파일이 덮어쓰기 되므로, 실험 버전을 유지하려면 결과 폴더를 별도로 보관하세요.

이 README는 GPT-5-Codex를 통해 작성했습니다.