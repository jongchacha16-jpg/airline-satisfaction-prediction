# PRD — 항공사 승객 만족도 예측 (중간평가)

> **전제 (Assumption)**: "해당 폴더의 data"는 `중간평가/data/` (Kaggle Playground S6E10 항공사 만족도 데이터)를 의미한다.
> 본 문서는 `baseline[Lab].ipynb`와 **동일한 형식·방법론**으로 후속 분석 노트북을 만들기 위한 계획서다.
>
> **작업 폴더 (Working Directory)**: 신규 분석은 전부 `중간평가/chacha/` 안에서 진행한다.
> - 노트북: `chacha/<주제>[Lab].ipynb`
> - 예측 산출물: `chacha/outputs/predictions/` (모델 예측) · `chacha/outputs/reports/` (실험 기록)
> - 데이터: 복사하지 않고 `../data/` 를 그대로 참조 (`find_data_dir()` 가 `data` / `../data` 를 모두 탐색하므로 수정 불필요)
> - 기존 `중간평가/outputs/oof_mlp.csv`·`test_mlp.csv` 는 **`chacha/outputs/predictions/` 로 복사**해 스태킹 입력을 한 폴더에 모은다.
> - 루트 `중간평가/baseline[Lab].ipynb`, `submission.csv` 등 **기존 파일은 수정하지 않는다** (읽기 전용 참조).

---

## 1. 현황 파악 (Current State)

### 1-1. 폴더 구성

| 경로 | 내용 |
|---|---|
| `data/train.csv` | 699,635 × 23 (id + 21 features + `satisfaction`) |
| `data/test.csv` | 299,844 × 22 |
| `data/sample_submission.csv` | 299,844 × 2 (`id`, `satisfaction` = 확률값 0.4436…) |
| `baseline[Lab].ipynb` | EDA + LogReg 벤치마크 + LightGBM 5-Fold (43 cells, 실행 완료) |
| `mlp_pytorch_baseline[Lab].ipynb` | PyTorch MLP 5-Fold (18 cells, 실행 완료) |
| `outputs/oof_mlp.csv`, `outputs/test_mlp.csv` | MLP의 OOF/Test 예측 (스태킹 입력) — 현재는 `chacha/outputs/predictions/` 로 복사됨 |
| `submission.csv` | 현재 제출본 = LightGBM 5-Fold 앙상블 |

### 1-2. 데이터 스키마

- **타깃**: `satisfaction` (bool) → `True` 44.36% / `False` 55.64% (불균형 없음 → 계층화 분할만 적용)
- **범주형 4개** (`CAT_COLS`): `Gender`, `Customer Type`, `Type of Travel`, `Class`
- **숫자형 17개** (`NUM_COLS`), 그중 **평점 13개** (`RATING_COLS`, 0–5점, 0 = 미이용/해당없음)
- **연속형 4개**: `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- **결측**: `Arrival Delay in Minutes` 단독 — train 292행 / test 130행 (< 0.05%)

### 1-3. 기존 EDA에서 확정된 사실 (재검증 불필요, 피처 설계의 근거)

1. **평점 0점 역전**: `Online boarding` 0점 → 만족률 ~62%, `Inflight wifi service` 0점 → ~88%. 1→5 구간은 단조 증가. **U-shape 비선형** → 트리 모델 우위 근거.
2. **지연 Zero-inflated**: 출발 지연 0분이 90.8%. 구간별 만족률 0분 0.448 → 60분+ 0.314.
3. **핵심 범주형**: `Type of Travel`, `Class` 편차 큼 / `Gender` 거의 무의미 (feature importance ≈ 0).
4. **Covariate Shift 없음**: train/test 범주 비율 차 ≤ 0.2%p → adversarial validation 불필요.
5. **상관 vs 중요도 괴리**: `Inflight wifi service` 선형 상관 7위 → LGBM gain 2위 (비선형 신호).


### 1-4. 성능 기준선 (Acceptance Bar)

| 모델 | 검증 방식 | AUC | 런타임 |
|---|---|---|---|
| Logistic Regression | holdout 20% | 0.92046 | 수십 초 |
| **LightGBM** | **5-fold OOF** | **0.95866** | **40s** (Step 0 실측) |
| MLP (PyTorch) | 5-fold OOF | 0.95730 | 89s (CPU) |

> ⚠️ **런타임 정정**: baseline 노트북이 출력한 `1,205초` 는 EDA·LogReg를 포함한 **노트북 전체** 실행 시간이다.
> Step 0에서 LightGBM 5-fold 루프만 분리 측정한 결과 **데이터 로드 포함 40초** (fold당 ≈7초).
> 초기 PRD의 예산 가정(§3-5)은 이 수치를 LGBM 학습 시간으로 오인한 것이며 아래와 같이 전면 완화한다.

> **채택 기준값 추이**: `0.95866`(기준선) → `0.95934`(Step 1) → `0.95986`(Step 2) → **`0.95994`(Step 4, 최종)**.
> 최종 fold 벡터 `[0.96022, 0.95890, 0.96075, 0.95994, 0.96000]`.

---

### 1-5. 기준선 코드 감사 (Baseline Audit)

`baseline[Lab].ipynb` 의 **코드와 원본 데이터를 직접 검증**한 결과. (§1-3은 노트북의 서술·출력에 근거, 본 절은 코드 자체 검증)

#### ✅ 이상 없음으로 확인된 항목

| 검사                                 | 결과                                         |
| ---------------------------------- | ------------------------------------------ |
| train 내 중복 피처 행                    | **0건** (라벨 충돌도 0)                          |
| train/test `id` 중복                 | **0건**                                     |
| train/test 피처 행 중복 (누수)            | **0건**                                     |
| test의 미등록 범주 수준                    | **없음** (4개 범주형 전부)                         |
| `RATING_COLS` 탐지 규칙 (`max() <= 5`) | train 기준·test 기준 결과 **동일한 13개** → 규칙 안정    |
| OOF 커버리지                           | `assert (oof_count == 1).all()` 로 이미 보증됨   |
| 순 지연 음수 7.67%                      | **정상** (비행 중 지연 회복). 이상치 아니며 F3이 포착할 신호 구조 |

**추가 드리프트 검증** (기준선 cell 25는 범주형 4개만 검사 — 평점·연속형 미검증 공백을 메움):
- 평점 13종 값별 비율 train vs test 최대 절대차 **≤ 0.0021** (최대: `Cleanliness`)
- **0점 비율** 차이 ≤ 0.0005 (`Inflight wifi` 1.873%→1.895%, `Online boarding` 1.307%→1.270%) → **F1 0점 플래그의 전제가 test에서도 성립**
- 연속형 분위수(1/25/50/75/99%) 차이: `Age` 0, 지연 0, `Flight Distance` 최대 5 (99% 지점)
- `Arrival Delay` 결측률 train 0.042% vs test 0.043%
→ **드리프트 없음. adversarial validation 불필요 확정.**

#### ⚠️ 예측·계획에 영향을 주는 결함

**A. 조기 종료의 낙관 편향이 모델 간 비대칭 (Differential optimism)**
- LGBM: `early_stopping(100)` 이 **채점 대상인 검증 fold**로 `best_iteration` 을 고른다.
- MLP: `best["va_pred"]` 를 **val AUC가 최대인 에폭**에서 저장 — 약 15 에폭에 걸친 잡음 지표의 argmax.
- 부스팅 곡선은 매끄럽고 자기상관이 높아 선택 편향이 작지만, 에폭 argmax는 더 크다.
  → **MLP의 0.95730이 LGBM의 0.95866보다 더 부풀려져 있다.** 두 수치는 엄밀히 동일 선상 비교가 아니다.
- **대응**: nested early stopping은 예산(4h) 초과이므로 재설계하지 않는다.
  §4 Step 4에서 **메타 모델이 MLP에 단독 OOF 대비 과도한 가중치를 주면 단순 평균을 채택**한다.

**B. 판정 기준이 Pooled OOF 단일값이라 잡음에 취약**
- 기준선 Fold별 AUC: 0.95877 / 0.95759 / 0.95930 / 0.95870 / 0.95902 → **std ≈ 0.0006**
- 1차 목표 개선폭 +0.0013이 fold 산포와 같은 자릿수다.
- **대응 (§3-4에 반영)**: Fold 계약이 분할을 고정하므로 비교는 대응표본(paired)이라 민감도가 높다.
  그럼에도 **5개 fold 중 4개 이상에서 개선**되어야 유효한 변경으로 인정하고, §7에 fold별 AUC 5개를 모두 기록한다.

#### 📝 서술·표현상의 결함 (발표 평가 대상이므로 수정)

| 위치 | 문제 | 수정 |
|---|---|---|
| cell 37 | Fold별 트리 수가 256~549로 2배 이상 차이나는데 **원시 gain을 그대로 평균** → fold 1이 중요도 차트를 과대 지배 | fold별 gain을 합=1로 정규화 후 평균 (Step 0 노트북에 적용 완료) |
| cell 0 | "14개 기내 서비스" 서술 vs 코드가 찾는 **13개** | 13개로 정정 |
| cell 34 | LogReg(holdout 20%)와 LGBM(5-fold OOF)을 **같은 표에 다른 프로토콜로** 병치 | 프로토콜 차이 명시. 또한 U-shape를 문서화해놓고 평점을 선형 연속값으로 투입해 "하한선"이 과소평가됨을 부기 |

**C. 런타임 추정치 오인 (Step 0에서 발견)**
- baseline이 출력한 `1,205초` 는 노트북 전체 시간이며 LGBM 학습 시간이 아니다. 실측 **40초**.
- 초기 PRD는 이를 근거로 "Optuna 16시간 → 불가"라 판단하고 단일 fold 우회 전략을 세웠으나, **근거가 무효**다.
- **대응**: §3-5 예산 전면 개정, §4 Step 2를 full 5-fold 직접 탐색으로 수정.

#### 결론

데이터 측 결함은 **없다** (누수·중복·드리프트 전부 음성). 방법론 결함 A·B는 기준선을 무효화하지 않으며,
본 PRD의 구조를 바꾸지 않고 **§3-4 판정 규칙과 §4 Step 4 가드에 반영**하는 선에서 처리한다.

## 2. 목표 (Objectives)

| 구분 | 내용 |
|---|---|
| **1차 목표 (희망)** | OOF ROC-AUC **≥ 0.9600**. **도달 보장 없음** — 아래 주석 참조 |
| **실질 판정 기준** | 매 단계 **직전 채택본을 4/5 fold 이상에서 초과**. **최종 달성 = 0.95994** (Step 4 `stacker`) |
| **2차 목표** | 다중 모델 OOF 스태킹으로 **≥ 0.9610** |
| **산출물** | 평가 제출용 `submission.csv` + 수업 발표용 노트북 2~3종 |
| **비목표** | Public LB 점수 추종, 외부 데이터 사용, 거대 모델(TabNet/FT-Transformer) 탐색 |

> ⚠️ **목표 도달 가능성에 대한 솔직한 평가 (Step 1 이후 갱신)**
> F1·F2·F3가 전부 기각된 것은 **GBDT 관점에서 피처 공간이 거의 소진**되었다는 증거다.
> *(Step 2 완료 후 갱신: 0.95986 달성으로 목표까지 **0.00014** 남음. 아래 비관 전망보다 진척이 좋았다 —
> Optuna 이득 +0.00034 는 예상 범위(+0.0002~0.0005) 내였고, 시드 앙상블 +0.00016 이 추가로 기여했다.)*
>
> 남은 +0.00066 를 Step 2·3이 메운다는 보장은 없다 — 이미 합리적인 LGBM에 대한 Optuna 이득은
> 통상 +0.0002~0.0005 수준이고, 스태킹 이득은 베이스 모델 간 상관에 의해 상한이 정해진다(§1-5 A).
> **`0.9600` 은 희망 목표이며, 실제 의사결정을 지배하는 것은 "직전 채택본 초과" 규칙이다.**
> 미달 시에도 §7 실험 로그가 왜 미달했는지를 설명하면 완료 조건(§8)을 만족한다.

---

## 3. 하드 제약 (Hard Constraints) — 반드시 준수

### 3-1. Fold 계약 (Fold Contract) ★ 최우선

모든 노트북은 아래 코드를 **문자 그대로 동일하게** 사용한다. 서로 다른 fold 분할로 생성된 OOF는 스태킹 시 누수를 일으키므로 변경 금지.

```python
SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(train, y))   # train 전체 인덱스 기준, 정렬/셔플 금지
```

### 3-2. 예측 산출물 계약 (Artifact Contract)

`mlp_pytorch_baseline[Lab].ipynb`의 `save_predictions()` 규약을 그대로 재사용한다.

- `chacha/outputs/predictions/oof_<name>.csv` → 컬럼 `id, fold, pred` (train 699,635행)
- `chacha/outputs/predictions/test_<name>.csv` → 컬럼 `id, pred` (test 299,844행)
- `chacha/outputs/reports/` → 단계별 요약 · Optuna 탐색 로그 (모델 입력이 아닌 분석 기록)

```
outputs/
├── predictions/   oof_* · test_* × 7 모델 = 14 파일   ← 스태킹 입력, Fold 계약 대상
└── reports/       folds_lgbm · step1~4 요약 = 7 파일  ← 실험 기록
```
- `<name>` ∈ `lgbm`, `mlp`, `cat`, `xgb`, `lgbm_fe`, `stack`
- 저장 전 `np.isfinite(...).all()` assert 유지

### 3-3. 제출 파일 계약 (Submission Contract)

`baseline[Lab].ipynb` cell 40의 assert 블록을 **필수**로 복사한다.

```python
assert pred_df["id"].is_unique
assert set(pred_df["id"]) == set(sample["id"])
submission = sample[["id"]].merge(pred_df, on="id", how="left", validate="one_to_one")
assert len(submission) == len(sample)                      # 299,844
assert (submission["id"].values == sample["id"].values).all()
assert submission[TARGET].notna().all()
assert submission[TARGET].between(0, 1).all()
```

> **덮어쓰기 정책**: 신규 모델은 `chacha/outputs/predictions/` 에만 쓴다. 루트 `중간평가/submission.csv` 는 건드리지 않고,
> 최종 제출본은 `chacha/submission_<모델>_<OOF점수>.csv` 로 저장한다. **루트 `submission.csv` 는 끝까지 불가침**으로 운용했다 (§8 참조).

### 3-4. 실험 프로토콜 (baseline 9장 "실험 원칙" 계승)

- **단일 변경 원칙**: 실험 1건당 변경점 1개. 피처 추가와 하이퍼파라미터 변경을 동시에 하지 않는다.
  **누적 실험 시 주의(§7 Step 1에서 적발)**: 직전 그룹이 기각되면 그 피처를 **제거한 구성 위에서** 다음 그룹을 측정한다.
  기각된 피처를 깔고 측정하면 다음 그룹의 기여가 왜곡된다 (F4: 교란 시 +0.00045 → 제거 시 +0.00068).
- **판정 기준은 OOF AUC 단독**. Public LB는 참고값으로만 기록.
- **4/5 Fold 규칙 (§1-5 B)**: Fold별 AUC std ≈ 0.0006 으로 목표 개선폭과 자릿수가 같다.
  Pooled OOF 개선만으로는 부족하며, **5개 fold 중 4개 이상에서 개선**되어야 채택한다.
  §7 실험 로그에 fold별 AUC 5개를 모두 기록한다.
- 모든 실험은 §7 실험 로그 표에 1행으로 기록.

### 3-5. 런타임 예산 (Runtime Budget) — Step 0 실측 기준 개정

LGBM 5-fold full = **40s** (fold당 ≈7초, 데이터 로드 포함). 초기 가정(20분)보다 **30배 빠르다**.

| 작업 | 개정 예산 | 초기 가정 |
|---|---|---|
| LGBM 5-fold 1회 | 40s | 1,205s |
| 피처 그룹 실험 4회 (Step 1) | ~3분 | 80분 |
| **Optuna 50 trial × full 5-fold** | **~35분** | 16시간 (불가 판정) |
| CatBoost 5-fold | ~5~10분 (추정) | 30분 |
| 전체 계획 | **≤ 1.5시간** | 4시간 |

**영향**: 초기 PRD의 "단일 fold 탐색 + lr 상향" 우회 전략은 **불필요**하다.
Optuna를 **full 5-fold, `learning_rate=0.05` 원본 설정 그대로** 돌릴 수 있으므로,
탐색 조건과 확정 조건이 일치해 **탐색-확정 간 괴리(search-refit gap)가 사라진다.** → §4 Step 2 수정.

## 4. 작업 계획 (Work Plan)

### Step 0 — LGBM 기준선 OOF 재생성 (`chacha/00_lgbm_baseline[Lab].ipynb`) ✅ 완료

**재생성 대상은 모델이 아니라 "파일"이다.** `baseline[Lab].ipynb` 는 5-Fold LightGBM을 학습하고
fold별 검증 예측(`oof_pred`)과 test 앙상블 예측(`test_pred`)을 **메모리 변수로만** 갖고 있다가,
cell 40에서 `test_pred` 만 `submission.csv` 로 저장하고 끝난다. **`oof_pred` 는 디스크에 남지 않는다.**
→ `outputs/` 에 `oof_mlp.csv` 는 있는데 `oof_lgbm.csv` 가 없는 이유이며, 이 상태로는 §4 Step 4 스태킹이 불가능하다.

따라서 Step 0에서 재생성하는 것은 다음 2개 파일이다.

| 생성 파일 | 내용 | 용도 |
|---|---|---|
| `chacha/outputs/predictions/oof_lgbm.csv` | `id, fold, pred` × 699,635행 — 각 행이 **검증 fold에 속했을 때의 예측값** | 스태킹 메타 모델의 입력 컬럼. 이게 없으면 Step 4 자체가 성립 불가 |
| `chacha/outputs/predictions/test_lgbm.csv` | `id, pred` × 299,844행 — 5개 fold 모델 예측의 평균 | 스태킹 결과를 test에 적용할 때 필요. `submission.csv` 와 수치는 같지만 컬럼명 규약(`pred`)이 다름 |

**왜 OOF가 반드시 파일이어야 하는가**: 스태킹 메타 모델은 "베이스 모델이 *학습에 쓰지 않은* 데이터에 대해 낸 예측"으로만
학습해야 한다. fold 1의 검증 예측은 fold 2~5로 학습한 모델이 낸 값이므로 누수가 없다.
train 전체를 다시 예측하면(= in-fold 예측) 자기 학습 데이터를 맞히는 값이라 과대평가되고 스태커가 망가진다.
그래서 "다시 돌려서 예측만 뽑기"가 아니라 **fold 루프를 동일하게 재실행**해야 한다.

**수행 방법**
1. `baseline[Lab].ipynb` 의 cell 2·3·5·8·28·33 (import / `find_data_dir` / 로드 / 컬럼 정의 / 전처리 / LGBM 5-fold 루프)을 그대로 가져온다.
2. `LGB_PARAMS`, `SEED=42`, `N_FOLDS=5`, `early_stopping(100)`, `num_boost_round=5000` — **파라미터 일절 변경 금지**.
3. §3-1 fold 계약 코드로 `FOLDS` 를 만들고, `fold_id` 배열을 함께 구성한다.
4. 루프 종료 후 `save_predictions("lgbm", oof_pred, test_pred)` 호출.

**검증 (이 단계의 진짜 목적)**
- 재현 OOF AUC 가 **0.95866 ± 0.0002** 이내인가 → 벗어나면 fold 분할이나 전처리가 baseline과 달라졌다는 뜻이므로 원인 규명 전까지 다음 단계 진행 금지.
- `oof_lgbm.csv` 의 `fold` 컬럼이 `중간평가/outputs/oof_mlp.csv` 의 `fold` 컬럼과 **행별로 완전히 동일**한가:
  ```python
  a = pd.read_csv("outputs/predictions/oof_lgbm.csv")[["id", "fold"]]
  b = pd.read_csv("outputs/predictions/oof_mlp.csv")[["id", "fold"]]
  m = a.merge(b, on="id", suffixes=("_lgbm", "_mlp"), validate="one_to_one")
  assert len(m) == 699_635 and (m.fold_lgbm == m.fold_mlp).all()
  ```
  이 assert가 Step 0의 핵심 산출물이다 — 두 노트북이 같은 fold 계약을 쓰고 있음을 증명하며,
  이후 모든 모델이 이 계약에 묶인다.
- `test_lgbm.csv["pred"]` 와 기존 `중간평가/submission.csv["satisfaction"]` 의 최대 절대 오차 < 1e-9 → baseline 재현 확인.

**예산**: 실측 **40초** (초기 추정 20분은 §1-5 C 참조). 결과는 §7 실험 로그 1행에 기록.

### Step 1 — 피처 엔지니어링 (`chacha/01_feature_engineering[Lab].ipynb`) ✅ 완료

EDA 근거 기반으로만 설계. 아래 4개 그룹을 **그룹 단위로 순차 추가**하며 각각 OOF 측정.

| # | 피처 그룹 | 근거 | 생성 규칙 |
|---|---|---|---|
| F1 | **0점 플래그 13개** `<col>_is_zero` | §1-3 (1) 0점 역전 | `(df[c] == 0).astype(int)` for c in RATING_COLS |
| F2 | **평점 집계** `rating_mean/std/sum/zero_count` | 서비스 전반 만족 수준 요약 | 13개 평점 대상 행단위 통계 (0 포함본 / 0 제외본 2종 비교) |
| F3 | **순 지연** `net_delay`, `has_delay` | §1-3 (2) zero-inflated | `Arrival - Departure`, `(Departure > 0).astype(int)` |
| F4 | **빈도 인코딩** `Age_freq`, `Flight Distance_freq` | MLP 노트북에서 이미 사용 중 | fold별 train에서만 fit (`add_freq_features` 재사용) |

**결측 전파 규칙**: `Arrival Delay in Minutes`에 NaN이 있으므로 `net_delay`도 NaN을 상속한다.
→ **LGBM/CatBoost/XGB: NaN 유지** (분기 자동 학습), **LogReg/MLP: median imputation**. 기존 컨벤션과 동일.
**누수 방지**: F1~F3은 행 단위 변환이므로 전체 적용 가능. F4는 **반드시 fold별 train에서만 fit**.

- 채택 기준: 그룹 추가 후 OOF AUC > 직전 최고치 **그리고 4/5 fold 개선**.
- 예산: 실측 468초 (7개 구성). **→ 완료, 결과는 §7 참조.**

### Step 2 — 하이퍼파라미터 탐색 (`chacha/02_hyperparameter_tuning[Lab].ipynb`) ✅ 완료

- 도구: Optuna TPE, **full 5-fold OOF AUC를 목적함수로** 직접 최적화 (§3-5 개정 예산으로 가능).
- `learning_rate=0.05`, `num_boost_round=5000`, `early_stopping(100)` 는 기준선과 동일 유지 — 탐색/확정 조건 일치.
- 탐색 공간: `num_leaves` (31–255), `min_child_samples` (10–200), `feature_fraction` (0.6–1.0),
  `bagging_fraction` (0.6–1.0) + `bagging_freq`, `lambda_l1/l2` (1e-3–10, log).
- **비교 기준**: Step 1 채택본 `lgbm_fe` (23 feat, OOF 0.95934). `BASELINE_AUC = 0.95934` 로 고정하고
  `oof_lgbm_fe.csv` 의 fold 벡터를 대응 비교 대상으로 로드한다.
- **과적합 방지**: 50 trial의 최댓값은 OOF에 대해서도 낙관 편향을 가지므로(§1-5 A와 같은 성격),
  최적 파라미터는 **seed 3개로 재학습해 평균 OOF**를 채택 기준으로 삼는다.
  **동일 처리를 양쪽에 적용**: Step 1 챔피언도 같은 seed 3개로 재측정해 비교한다 —
  3-seed 평균(분산 감소)과 1-seed 단일값을 비교하면 튜닝 모델에 부당한 이점이 생긴다. (+6분)
- 예산: 약 40분 → **실측 90분** (§7 Step 2 「4」 참조).

### Step 3 — 모델 다변화 (`chacha/03_model_diversification[Lab].ipynb`) ✅ 완료

| 모델 | 추가 이유 | 예산 |
|---|---|---|
| **CatBoost** | 범주형 native 처리(Ordered TS)가 LGBM과 달라 상관 낮은 오차 기대 | ~30분 |
| **XGBoost** (선택) | 예산 여유 시에만. CatBoost 대비 다양성 기여 적으면 생략 | ~25분 |
| MLP | 이미 보유 (`outputs/*_mlp.csv`) | 0 |

각 모델은 Step 1 채택 피처 + §3-1 fold 계약으로 학습하고 `save_predictions()`로 `chacha/outputs/predictions/` 에 저장.
**추가 판단 기준**: 신규 모델 OOF가 단독으로 0.95 미만이거나 LGBM OOF와 피어슨 상관 > 0.99면 다양성 기여 없음 → 생략.

### Step 4 — 스태킹 (`chacha/04_stacking[Lab].ipynb`) ✅ 완료

- 입력: `chacha/outputs/predictions/oof_*.csv` 를 `id` 기준 merge
- **입력 비대칭 주의**: `lgbm_tuned` 는 3-seed 예측 평균이라 이미 분산이 낮다. 단일 모델 `mlp` 와 동등하지 않다. → 피처 행렬 (모델 수만큼의 컬럼).
- 메타 모델: **Logistic Regression** 1순위 (해석 용이, 가중치를 발표에 활용), Ridge / shallow LGBM 비교.
- 메타 모델의 CV는 **동일한 `fold` 컬럼**으로 수행 (새 분할 생성 금지).
- 비교 대상: 단순 평균(rank-average 포함) vs 학습된 스태커. 단순 평균이 이기면 단순 평균 채택.
- **낙관 편향 가드 (§1-5 A)**: MLP의 OOF는 에폭 argmax 선택으로 LGBM보다 더 부풀려져 있다.
  메타 모델이 MLP에 단독 OOF 순위에 비해 과도한 가중치를 부여하면 **학습된 스태커를 버리고 단순 평균을 채택**한다.
- 예산: 10분.

### Step 5 — 제출 및 문서화

- ✅ `chacha/submission_stacker_0.95994.csv` 생성 완료 (§3-3 assert 7종 통과).
- ✅ 루트 승격 **하지 않음**(확정). 루트 `submission.csv` 는 `baseline[Lab].ipynb` 의 산출물로 보존하고,
  제출에는 `chacha/submission_stacker_0.95994.csv` 를 사용한다. 파일명에 OOF 점수를 포함해 구성 추적이 가능하다.
- 실험 로그 표(§7) 완성, baseline 9장 형식의 "향후 개선 과제" 갱신.

---

## 5. 노트북 작성 규격 (Presentation Requirements) ★ 평가 대상

중간평가 제출물이므로 **AUC만큼 서술 품질이 중요**하다. 신규 노트북은 `baseline[Lab].ipynb`의 체재를 그대로 따른다.

1. **파일명**: `<주제>[Lab].ipynb` (예: `fe_lgbm[Lab].ipynb`, `catboost[Lab].ipynb`, `stacking[Lab].ipynb`)
2. **섹션 번호 + 이중 언어 제목**: `## 4. 데이터 전처리 파이프라인 (Data Preprocessing Pipeline)`
3. **코드 셀 앞에 근거 마크다운**: 무엇을·왜 하는지 불릿으로 선서술 (baseline cell 29, 32 톤 유지)
4. **차트/표 뒤에 해석 블록**: `- **~ 결과**:` 형태로 수치 인용하며 해석 (baseline cell 18, 38 톤)
5. **차트 스타일 통일**: `COLOR_MAIN = "#2a78d6"`, `COLOR_ALT = "#eb6834"`, top/right spine 제거, 가벼운 그리드
6. **환경 독립성**: `find_data_dir()` / `OUT_DIR` (Kaggle `/kaggle/working` 분기) 블록 복사 — 로컬·Kaggle 양쪽 실행 가능
7. **수식은 LaTeX**, 전문 용어는 `한국어 (English)` 병기
8. 마지막 셀에 `전체 실행 시간` 출력
9. **커널 고정**: `sesac-ai` 커널(`.venv/bin/python` (프로젝트 루트의 uv 가상환경))을 사용한다.
   시스템 기본 `python3`(anaconda)에는 lightgbm이 없고, `sesac-ai` 가 baseline과 동일한 버전 스택
   (lightgbm 4.7.0 / pandas 3.0.5 / scikit-learn 1.9.1)을 제공해 재현성이 보장된다.
10. **배치 실행 주의**: 파일명의 `[Lab]` 이 쉘·nbconvert에서 글로브 문자 클래스로 해석되므로,
   CLI 실행은 `jupyter nbconvert` 대신 `nbclient` 스크립트(`chacha/run_nb.py`)를 사용한다.

---

## 6. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| ~~LGBM 1회 20분~~ → **해소** (실측 40초, §1-5 C) | — | 제약 아님. full 5-fold 자유 반복 가능 |
| 피처 추가해도 개선 미미 (baseline 이미 0.9587로 포화권) | 1차 목표 미달 | 목표를 +0.0013으로 보수적 설정. 미달 시 스태킹(Step 4)에 비중 이동 |
| F4 빈도 인코딩 누수 | OOF 과대평가 | fold train에서만 fit, `add_freq_features` 재사용 + 검증 코드 |
| 스태킹에 다른 fold 분할 혼입 | 심각한 누수 | 모든 `oof_*.csv`의 `fold` 컬럼 일치 여부 assert |
| 루트 `submission.csv` 성급한 덮어쓰기 | baseline 노트북 재현성 파괴 | **루트 불가침 — 최종까지 미변경.** `chacha/` 내 점수 포함 파일명으로만 저장 |

---

## 7. 실험 로그 (작업 중 갱신)

| # | 노트북 | 변경점 (단일) | OOF AUC | Δ vs 0.95866 | 런타임 | 채택 |
|---|---|---|---|---|---|---|
| 0 | baseline[Lab] | — (기준선) | 0.95866 | — | 1205s(전체 노트북) | ✅ |
| 0b | mlp_pytorch_baseline[Lab] | MLP | 0.95730 | −0.00136 | 89s | 스태킹용 보유 |
| 1 | chacha/00_lgbm_baseline[Lab] | Step 0 LGBM OOF 재생성 (파라미터 무변경) | **0.95866** | ±0.00000 | 40s | ✅ 재현 완료 |

**Step 0 Fold별 상세** (이후 실험의 4/5 Fold 비교 기준선):

| fold | AUC | trees | n_valid |
|---|---|---|---|
| 1 | 0.95877 | 549 | 139,927 |
| 2 | 0.95759 | 293 | 139,927 |
| 3 | 0.95930 | 286 | 139,927 |
| 4 | 0.95870 | 280 | 139,927 |
| 5 | 0.95902 | 256 | 139,927 |
| **pooled** | **0.95866** | — | 699,635 |

mean 0.95868 / std 0.00058

| 2 | chacha/01_feature_engineering[Lab] | +F1 0점 플래그 13개 | 0.95865 | −0.00001 | 42s | ❌ 2/5 fold |
| 3 | 〃 | +F2a 평점 집계(0 포함) | 0.95851 | −0.00015 | 37s | ❌ 1/5 fold |
| 3b | 〃 | +F2b 평점 집계(0 제외) | 0.95846 | −0.00020 | 39s | ❌ 0/5 fold |
| 4 | 〃 | +F3 순 지연 | 0.95848 | −0.00018 | 36s | ❌ 0/5 fold |
| 5 | 〃 | +F4 빈도 인코딩 (누적 = F1+F2a+F3 포함, **교란됨**) | 0.95911 | +0.00045 | 47s | ⚠️ 교란 |
| 5a | 〃 | **F4 단독 (BASE + 빈도 인코딩, 23 feat)** | **0.95934** | **+0.00068** | 59s | **✅ 5/5 fold** |
| 5b | 〃 | F1 + F4 (36 feat) | 0.95932 | +0.00065 | 55s | ✅ 5/5 (F4 단독보다 복잡, 미채택) |

**Step 1 결론**: 설계한 4개 그룹 중 **F4(빈도 인코딩)만 유효**. 채택 구성 `lgbm_fe` = BASE 21 + `Age_freq`, `Flight Distance_freq` = **23 feature, OOF 0.95934**.
Fold별 AUC: 0.95947 / 0.95839 / 0.96013 / 0.95930 / 0.95946 (5/5 개선)

| 6 | chacha/02_hyperparameter_tuning[Lab] | Optuna 23 trial (timeout 2700s 중단) | 0.95974 | +0.00040 | 2790s | 탐색 최고값 |
| 6a | 〃 | ├ 챔피언 3-seed 평균 (대조군) | 0.95936 | +0.00002 | 2318s | 비교 기준 |
| 6b | 〃 | ├ **튜닝 3-seed 평균** | **0.95970** | **+0.00034** | 259s | **✅ 5/5 fold** |
| 6c | 〃 | └ **3-seed 예측 평균 → `lgbm_tuned`** | **0.95986** | **+0.00120** | — | **✅ 최종 채택** |

**Step 2 채택 파라미터**: `num_leaves=60`, `min_child_samples=21`, `feature_fraction=0.800`,
`bagging_fraction=0.894`, `bagging_freq=6`, `lambda_l1=0.00205`, `lambda_l2=8.183`
(`learning_rate=0.05`, `num_boost_round=5000`, `early_stopping(100)` 고정)

**3-seed 평균 fold 벡터** (Step 3 이후 대응 비교 대상): `[0.95993, 0.95851, 0.96050, 0.95981, 0.95984]`

| 7 | chacha/03_model_diversification[Lab] | **CatBoost** 5-fold (튜닝 없음) | 0.95935 | −0.00051 | 1397s | 다양성 입력 ✅ |
| 7b | 〃 | **XGBoost** 5-fold (튜닝 없음) | 0.95929 | −0.00057 | 187s | 다양성 입력 ✅ |
| 7c | 〃 | 동일가중 rank-average 최고 조합 (tuned+fe+cat) | 0.95989 | +0.00003 | — | 참고 (15중 2개만 champion 초과) |
| 8 | chacha/04_stacking[Lab] | prob + LogReg | 0.95980 | −0.00006 | 13s | ❌ |
| 8b | 〃 | rank + LogReg | 0.95968 | −0.00018 | 〃 | ❌ |
| 8c | 〃 | **logit + LogReg** | **0.95994** | **+0.00008** | 〃 | **✅ 4/5 fold — 최종 채택** |

**최종 산출물**: `chacha/submission_stacker_0.95994.csv` (299,844행, §3-3 assert 7종 전부 통과)
`outputs/predictions/oof_stack.csv` / `test_stack.csv` 동시 저장.

**최종 누적**: 기준선 `0.95866` → **`0.95994`** (**+0.00128**). 1차 목표 `0.9600` 대비 **−0.00006 미달**.

| 9 | — | **Public LB 제출 — 스태킹** | LB **0.95935** | OOF 대비 −0.00059 | — | 최종 제출 |
| 9b | — | Public LB 제출 — `lgbm_tuned` 단독 (대조군) | LB 0.95922 | OOF 대비 −0.00064 | — | 비교용 |

**LB 검증 결론 (2026-10-22)**: 두 구성의 OOF→LB 격차가 −0.00059 / −0.00064 로 **거의 동일**하다.
OOF는 절대 수치로 약 0.0006 낙관적이나 **편향이 모델 간 일정**하므로 상대 비교 도구로 유효하다
— §3-4의 "직전 채택본 대비 Δ" 판정 원칙이 실증되었다.
스태킹 이득은 로컬 +0.00008 → LB **+0.00013** 으로 **축소되지 않고 전이**되었다.
§7 Step 3·4 「5」에서 제기한 "MLP 편향 의존" 우려는 **기각**된다 (단, 단일 공개 표본이므로 크기는 잡음 범위).


---


#### Step 3·4에서 확인된 사실

**1. Pearson 상관은 앙상블 판단에 쓸 수 없다**
전 모델 쌍의 Pearson은 0.9933~0.9990 으로 "모두 동일"하게 보이지만, Spearman은 0.9267~0.9836 으로 벌어진다.
ROC-AUC는 예측값의 **순위만** 사용하므로 **Spearman이 올바른 척도**다.
(`mlp`: Pearson 0.9947 / Spearman 0.9357 — 가장 극단적인 사례)

| | lgbm_tuned | lgbm_fe | mlp | cat | xgb |
|---|---|---|---|---|---|
| **lgbm_tuned** | 1.000 | 0.984 | **0.936** | 0.980 | 0.981 |
| **cat** | 0.980 | 0.968 | 0.937 | 1.000 | 0.979 |
| **xgb** | 0.981 | 0.970 | 0.927 | 0.979 | 1.000 |

**2. CatBoost의 비용 대비 효과가 나빴다**
CatBoost 1,397초 (fold당 2,470~4,672 iter) vs XGBoost 187초 — **7.5배 비싼데 단독 성능 차이는 +0.00006**.
다양성(Spearman 0.980 vs 0.981)도 사실상 동일했다.
→ Oblivious Tree + Ordered TS 가 LightGBM과 근본적으로 다른 오차 구조를 만들 것이라는 §4 Step 3의
  가설은 **이 데이터에서 성립하지 않았다**. 재현 시 XGBoost만으로 충분하다.

**3. 동일 가중 평균은 강한 모델을 끌어내린다**
`lgbm_tuned` 포함 15개 조합 중 champion 초과는 **2개뿐이며 그마저 +0.00003**.
`lgbm_tuned` 가 이미 3-seed 앙상블이라 분산이 낮은 반면 나머지는 모두 더 약하기(0.9573~0.9594) 때문이다.
→ §4 Step 4의 "단순 평균이 이기면 단순 평균" 규칙은 **발동하지 않았다** (rank-average 1/5 fold, −0.00005).

**4. 메타 피처 표현이 결과를 가른다**
동일한 LogReg 메타 모델인데 표현에 따라 OOF가 0.95968~0.95994 로 갈렸다.

| 표현 | OOF | 해석 |
|---|---|---|
| `logit` | **0.95994** | 0/1 근처에서 포화된 확률을 선형 영역으로 펼쳐 선형 메타 모델에 적합 |
| `prob` | 0.95980 | 포화 구간에서 선형 결합의 표현력이 떨어짐 |
| `rank` | 0.95968 | 분포 강건성은 얻지만 **확률 크기 정보를 버려** 손해가 더 컸음 |

→ OOF↔Test 분포 차이를 우려해 도입한 `rank` 변환은 **이 데이터에서 역효과**였다. 정규화 강도(C=1 vs 0.1)는 무관.

**5. 낙관 편향 가드가 실제로 발동했다 (§1-5 A 검증)**
메타 모델이 `mlp` 에 부여한 가중치는 **4위**인데 단독 성능은 **6위** — 순위 괴리 2로 가드 조건을 충족했다.
예측대로 **에폭 argmax로 부풀려진 OOF가 메타 모델을 오도**한 것이다.
다만 대안인 rank-average가 채택 기준(4/5 fold)을 통과하지 못해(1/5) **스태커를 유지**했다.
→ 가드는 "스태커를 버려라"가 아니라 "대안을 검토하라"로 작동했고, 대안이 더 나빠 기각되었다.
  **이 때문에 최종 +0.00008 중 일부는 `mlp` 의 편향에 기댄 것일 수 있으며, Public LB에서 축소될 수 있다.**

**6. 음의 계수는 오류가 아니다**
`lgbm`(구버전, 0.95866)이 계수 **−0.198** 을 받았다. 메타 모델이 `lgbm_tuned` 와의
**공통 오차를 상쇄**하는 용도로 사용한 것으로, 선형 스태킹에서 정상적인 동작이다.

**7. 스태킹의 실질 이득은 +0.00008 로 매우 작았다**
베이스 모델 5개를 추가 학습(약 27분)해 얻은 것이 champion 대비 +0.00008 이다.
Step 1의 교훈과 같은 결론 — **이 데이터셋은 GBDT 관점에서 거의 소진된 상태**이며,
Step 2의 하이퍼파라미터 튜닝(+0.00052)이 가장 비용 대비 효율이 높은 단계였다.

#### Step 2에서 확인된 사실

**1. OOF에 대한 선택 편향이 거의 없었다 (+0.00004)**
23회 시행의 최댓값 `0.95974` 와 동일 파라미터 3-seed 평균 `0.95970` 의 차이가 **+0.00004** 에 불과하다.
→ Optuna가 OOF 잡음을 긁어모은 것이 아니라 **실제로 더 나은 파라미터를 찾았다**는 증거이며,
699,635행 pooled OOF는 이 규모의 탐색에 대해 충분히 안정적인 목적함수다.

**2. 대조군을 동등하게 처리한 것이 실제로 필요했다**
챔피언의 3-seed 평균 `0.95936` 은 단일 seed `0.95934` 보다 `+0.00002` 높다.
결론을 바꿀 크기는 아니지만, 챔피언을 1-seed로 두고 비교했다면 튜닝 이득이 `+0.00040` 으로
**17% 부풀려졌을 것**이다. seed 간 std는 양쪽 모두 0.00002~0.00004로 매우 작다.

**3. 이득의 약 1/3은 튜닝이 아니라 시드 앙상블에서 나왔다**
- 튜닝 순수 기여 (3-seed 평균끼리 비교): **+0.00034**
- 시드 3개 예측을 평균한 추가 이득: **+0.00016**
- 합계 `0.95986` — Step 1 대비 +0.00052, 기준선 대비 **+0.00120**
→ `lgbm_tuned` 는 **단일 모델이 아니라 3-seed 예측 평균**이다. Step 4 스태킹에서 단일 모델 `mlp` 와
  동등한 입력으로 취급하면 안 된다 (이미 분산이 낮춰진 입력).

**4. 런타임 재추정 실패 (2회째)**
trial당 60초 가정 → 실측 70~190초. `num_leaves` 증가 시 비용이 선형 이상으로 늘어난다.
40 trial 목표가 **timeout 2700초에서 23 trial로 중단**, 단계 전체는 40분 예상 대비 **90분**.
best는 trial 15에서 나왔고 이후 7회 갱신 없음(trial 13~22 범위 0.95949~0.95974로 분산 축소).
→ **추가 탐색의 기대 이득이 작다**고 판단해 재시도하지 않는다.

#### Step 1에서 확인된 사실 (설계 가정의 수정)

**1. EDA에서 현상이 관측되는 것과, 그 현상을 피처로 만들면 모델이 개선되는 것은 별개다.**
F1(0점 플래그)은 PRD가 "가장 가치 높은 항목"으로 지목했으나 **2/5 fold, −0.00001 로 기각**되었다.
0점 역전 현상 자체는 실재하지만(§1-3), LightGBM은 평점을 수치형으로 분할하며 `<= 0.5` 분기로
**0점을 이미 자력 분리**한다. 명시적 플래그는 정보를 추가하지 않고 탐색 공간만 넓혀 약한 음의 효과를 낸다.
F2(집계)·F3(차분)도 동일한 이유 — **트리가 분할 조합으로 근사 가능한 변환은 GBDT에 기여하지 않는다.**

**2. 유효했던 것은 트리가 스스로 만들 수 없는 정보였다.**
`Age_freq` / `Flight Distance_freq` 는 "같은 값을 갖는 행이 몇 개인가"라는 **열 전체의 분포 정보**로,
행 단위 분할로는 도달할 수 없다. 이것만이 5/5 fold에서 개선(+0.00068)을 냈다.
→ **향후 피처 설계 원칙**: 행 내부 변환이 아니라 **집계·분포 기반 정보**를 우선한다.

**3. 채택 피처의 Test 전이 안전성 검증 (빈도 인코딩 고유 리스크)**
빈도 인코딩은 **값의 출현 횟수**를 쓰므로, test에만 있는 값은 `fillna(0)` 으로 **학습 중 본 적 없는 `freq=0`** 이 되어
트리가 임의 분기를 탄다. 유일한 채택 피처가 여기 기대고 있으므로 직접 측정했다.

| 피처 | nunique (tr/te) | fold-train 미등장 비율 — **test** | 동일 — **validation** |
|---|---|---|---|
| `Age` | 75 / 75 | **0.0000%** (전 fold) | 0.0000% |
| `Flight Distance` | 3,474 / 3,317 | 0.025~0.031% | 0.021~0.037% |

→ `Age` 는 완전 피복. `Flight Distance` 는 최악 **0.037%** 로 무시 가능하며,
무엇보다 **test의 미등장 비율(0.025~0.031%)과 validation의 미등장 비율(0.021~0.037%)이 동일 수준**이다.
즉 OOF 평가 시 겪는 조건과 test 추론 시 겪는 조건이 같으므로, **OOF 개선분이 test로 전이된다고 볼 수 있다.**
(`Flight Distance` 는 count 중앙값 52, 최솟값 1로 희소 구간이 존재하나 전체 영향은 상기 비율로 제한된다.)

**4. 누적 실험 설계의 구조적 결함 (절제 실험으로 적발)**
`F4_freq`(누적, 42 feat)는 기각된 F1·F2·F3 위에서 측정되어 **+0.00045로 과소평가**되었다.
BASE 위에 F4만 올리자 **0.95934 (+0.00068)** 로 개선폭이 1.5배가 되었다.
→ **§3-4 보강**: 누적 설계는 직전 그룹이 *채택되었을 때만* 단일 변경 원칙을 만족한다.
그룹이 기각되면 **기각 피처를 제거한 구성에서 다음 그룹을 재측정**해야 한다.

---

## 8. 완료 정의 (Definition of Done)

- [x] `chacha/outputs/predictions/` 에 모든 모델의 `oof_*.csv` / `test_*.csv` 존재, fold 컬럼 전원 일치
      (`lgbm`, `lgbm_fe`, `lgbm_tuned`, `cat`, `xgb`, `mlp`, `stack` — Step 4 첫 셀에서 assert 통과)
- [x] 최종 OOF AUC **0.95994**. 1차 목표 `0.9600` 에 **−0.00006 미달** — 사유는 §7 Step 3·4 「7」:
      피처(Step 1 3/4 기각)·앙상블(스태킹 +0.00008) 양쪽에서 여지가 소진됨. 튜닝만이 유효했다(+0.00052).
- [x] §3-3 assert 전체 통과한 제출 파일 `chacha/submission_stacker_0.95994.csv` (299,844행)
- [x] 신규 노트북 5종이 §5 서술 규격 충족, `run_nb.py` 로 전체 재실행 검증 완료
- [x] §7 실험 로그 전 행 기입
- [x] 루트 `중간평가/submission.csv` 는 **승격하지 않고 원본 유지**로 결정 (2026-10-21)
      → 해당 파일은 `baseline[Lab].ipynb` cell 40의 산출물이므로, 덮어쓰면 기준선 노트북의
        Restart & Run All 결과와 디스크 내용이 불일치하고 Step 0의 비트 단위 재현 검증도 무효가 된다.
        제출은 `chacha/submission_stacker_0.95994.csv` 를 직접 사용한다.
