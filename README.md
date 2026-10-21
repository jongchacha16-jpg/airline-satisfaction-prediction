# 항공사 승객 만족도 예측 — 0.0005를 진짜라고 판정하는 법

> **Kaggle Playground Series S6E10** · 정형 데이터 이진 분류 · ROC-AUC
> **OOF AUC 0.95866 → 0.95994** (+0.00128) · 699,635 train / 299,844 test · 5-Fold

SeSac AI 엔지니어링 과정 중간평가 과제입니다.

---

## 이 프로젝트에서 실제로 어려웠던 것

이미 OOF AUC **0.95866**인 LightGBM 베이스라인이 있었고, 목표는 **0.9600** 돌파였습니다.
남은 여지는 0.0013.

그런데 측정해보니 **Fold별 AUC의 표준편차가 0.00058**이었습니다.
**잡음과 목표 개선폭의 자릿수가 같았습니다.**

그래서 이 프로젝트의 어려운 부분은 모델을 더 좋게 만드는 일이 아니라,
**"지금 올라간 0.0005가 진짜 개선인가, 운인가"를 판정하는 일**이었습니다.
작업의 상당 부분을 모델링이 아니라 **실험 프로토콜 설계와 검증**에 썼습니다.

그 결과 설계한 피처 **4개 중 3개를 기각**했고, 기대했던 **CatBoost 도입도 비용 대비
효과가 없다고 결론**냈습니다. 이 프로젝트에서 가장 많이 한 일은 추가가 아니라 **기각**입니다.

| 단계 | OOF AUC | 기여 | 비용 |
|---|---|---|---|
| 베이스라인 (LightGBM) | 0.95866 | — | 40s |
| Step 1 · 피처 엔지니어링 | 0.95934 | +0.00068 | 8분 |
| Step 2 · Optuna + 시드 앙상블 | 0.95986 | +0.00052 | 90분 |
| Step 3·4 · 모델 다변화 + 스태킹 | **0.95994** | +0.00008 | 27분 |

**목표 0.9600에 0.00006 미달했습니다.** 사유는 아래 「한계」에 적었습니다.

> **상세 기술 문서는 [`chacha/README.md`](./chacha/README.md)** 에 있습니다 —
> 실험 프로토콜, 단계별 측정값, 기각 근거, 전이 안전성 검증까지.

---

## 핵심 판단 5가지

**1. 성능을 재기 전에 "무엇을 개선이라 부를지"부터 정했습니다**

```
Pooled OOF 개선 且 5개 Fold 중 4개 이상에서 개선일 때만 채택한다.
```

Fold 분할을 모든 노트북이 문자 그대로 공유하게 고정하고(`SEED=42`), 각 노트북 첫 셀에서
기존 산출물과 행별 일치를 `assert`합니다. 이 계약이 깨지면 스태킹 전체가 무효입니다.

**2. 런타임을 잘못 읽은 것 하나가 전략을 뒤집었습니다**

노트북 **전체** 실행 시간 1,205초를 LightGBM **학습** 시간으로 읽고 있었습니다.
실측은 **40초**. 이 정정으로 *"Optuna 50 trial은 16시간이라 불가능"* 이라는 판단이
무너졌고, 우회 없이 **full 5-Fold 직접 최적화**가 가능해졌습니다 — 결과적으로 가장 큰
기여를 한 단계가 열렸습니다.

**3. 가장 유망하다고 본 피처가 기각됐습니다**

EDA에서 0점 역전이 뚜렷했습니다(`Inflight wifi service` 0점의 만족률 88%, 1~5점 구간의
단조 증가와 정반대). 그런데 플래그를 추가해도 효과가 없었습니다 —
**LightGBM이 `<= 0.5` 분기로 0점을 이미 자력 분리**하고 있었기 때문입니다.

> EDA에서 현상이 보인다는 것과, 그 현상을 피처로 만들면 모델이 좋아진다는 것은 별개입니다.

**4. 대조군을 동등하게 대우하지 않았다면 이득이 17% 부풀려졌습니다**

Optuna 결과를 시드 3개로 평균 내면서 **챔피언도 동일한 시드 3개로 재측정**했습니다.
한쪽에만 분산 감소 혜택이 가면 같은 조건의 비교가 아니기 때문입니다.
챔피언을 1-seed로 뒀다면 튜닝 이득이 +0.00034 대신 +0.00040으로 보였을 것입니다.

**5. Pearson 상관은 앙상블 판단에 쓸 수 없습니다**

전 모델 쌍의 Pearson은 0.9933~0.9990으로 "전부 동일"하게 보이지만,
Spearman은 0.9267~0.9836으로 벌어집니다. **ROC-AUC는 예측값의 순위만 사용**하므로
Spearman이 올바른 척도입니다.

---

## 한계

- **목표 0.9600에 0.00006 미달.** 피처(4개 중 3개 기각)와 앙상블(스태킹 +0.00008)
  양쪽에서 여지가 소진됐고, **세 단계 중 튜닝만이 확실히 유효**했습니다.
- **OOF 추정치에는 상반된 두 편향이 섞여 있습니다.** Early stopping이 채점 대상 Fold로
  `best_iteration`을 고르므로 낙관 쪽으로, OOF는 단일 모델 예측인데 test는 5개 모델
  평균이라 비관 쪽으로 작용합니다. 그래서 OOF를 절대 수치가 아니라 **모델 간 비교
  기준으로만** 쓰고, 모든 판정을 "직전 채택본 대비 Δ"로 내렸습니다.
- **최종 +0.00008 중 일부는 MLP의 낙관 편향에 기댄 것일 수 있습니다.** 사전에 설계한
  가드가 발동했고 대안(rank-average)을 검토했지만, 대안이 채택 기준에 미달해 스태커를
  유지했습니다.
- **Public LB는 확인하지 않았습니다.** 로컬 OOF 과적합을 피하기 위한 의도적 선택입니다.

---

## 구성

```
.
├── baseline[Lab].ipynb                      LightGBM 베이스라인 (출발점)
├── mlp_pytorch_baseline[Lab].ipynb          MLP 베이스라인 (스태킹 입력 중 하나)
└── chacha/                                  ← 이번 작업
    ├── 00_lgbm_baseline[Lab].ipynb          기준선 재현 + OOF 복원
    ├── 01_feature_engineering[Lab].ipynb    피처 4그룹 + 절제 실험
    ├── 02_hyperparameter_tuning[Lab].ipynb  Optuna + 다중 시드 검증
    ├── 03_model_diversification[Lab].ipynb  CatBoost / XGBoost + 다양성 분석
    ├── 04_stacking[Lab].ipynb               메타 모델 + 제출 파일
    ├── README.md                            상세 기술 문서
    ├── prd.md                               계획서 · 코드 감사 · 전체 실험 로그
    ├── run_nb.py                            노트북 배치 실행 래퍼
    └── outputs/                             단계별 요약 (단계별 측정값·최적 파라미터)
```

## 재현

**데이터와 대용량 예측 산출물은 저장소에 포함하지 않습니다** — 재현 가능하고 합계 약
310MB이기 때문입니다. 아래 순서로 실행하면 전부 재생성됩니다.

```bash
# 1. 데이터 준비 — Kaggle 대회 페이지에서 받아 data/ 에 배치
#    https://www.kaggle.com/competitions/playground-series-s6e10
#    data/train.csv, data/test.csv, data/sample_submission.csv

# 2. 의존성
uv sync      # optuna, lightgbm, xgboost, catboost 포함

# 3. 순서대로 실행 (전체 약 2시간, CPU)
python run_nb.py '00_lgbm_baseline[Lab].ipynb'
python run_nb.py '01_feature_engineering[Lab].ipynb'
python run_nb.py '02_hyperparameter_tuning[Lab].ipynb'
python run_nb.py '03_model_diversification[Lab].ipynb'
python run_nb.py '04_stacking[Lab].ipynb'
```

각 노트북이 직전 단계의 산출물을 읽고 **Fold 계약을 검증**하므로 순서를 지켜야 합니다.
파일명의 `[Lab]`이 셸에서 글로브 문자 클래스로 해석되기 때문에 `jupyter nbconvert` CLI
대신 `run_nb.py`(nbclient 래퍼)를 사용합니다.

## 기술 스택

Python 3.13 · LightGBM 4.7 · XGBoost 3.4 · CatBoost 1.2 · Optuna 5.0 ·
scikit-learn 1.9 · pandas 3.0 · PyTorch 2.14
