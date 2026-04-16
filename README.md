# Credit Scoring — Give Me Some Credit
### Analytical Solution: Top 12% on Kaggle (Private AUC = 0.86720)

> 📌 **Note:** The Russian version of this README is available below.

---

## 📚 Notebooks & Resources

| Language | Link |
|----------|------|
| 🇬🇧 English |[credit_scoring_final_en.ipynb](./credit_scoring_final_en.ipynb)) |
| 🇷🇺 Russian | [credit_scoring_final_ru.ipynb](./credit_scoring_final_ru.ipynb) |
| 🏆 Kaggle | *(link to your Kaggle notebook)* |

---

## 🇬🇧 English

### Project Overview

This project solves the Kaggle competition **Give Me Some Credit**. The objective is to predict the probability that a borrower will experience **serious financial distress** — defined as 90+ days past due — within two years, using 10 demographic and behavioural features derived from credit bureau data. The metric is **ROC-AUC**, which measures the model's ability to rank borrowers by risk regardless of the classification threshold.

The final ensemble achieves **Private AUC = 0.86720**, placing in the **top 12%**. The solution emphasises **analytical rigour**, **interpretability**, and **banking applicability**, choosing RF + LR over gradient boosting specifically to meet Basel II explainability requirements.

---

### Analytical Approach

#### 1. Exploratory Data Analysis (EDA) — Key Insights

- **Class imbalance** is severe: only 6.7% of borrowers defaulted (14:1 ratio). `class_weight='balanced'` was applied to both models — without it, a model predicting "no default" for everyone achieves 93% accuracy but zero recall on actual defaults.
- **Missing values are not random**: `MonthlyIncome` is missing for ~20% of clients; the missing group has a **+13 pp higher default rate** than average — the fact of missingness is itself a risk signal. A binary flag `income_is_missing` was created *before* any filling.
- **DPD features are zero-inflated**: 75%+ of delinquency counts are zero. Standard `pd.qcut` fails on these, returning IV ≈ 0. After manual 0/1/2/3+ binning, IV jumps to 0.60–0.88.
- **RevolvingUtilisation > 1.0 is a real signal**: clipping to 1.0 (the logical business ceiling) *reduced* AUC by 0.005. Clients with utilisation > 1.0 have a **37% default rate** vs 6% for those below — indicating overdraft or penalty accrual. The p99 threshold (1.09) was used instead.
- **Age is non-linear**: Pearson correlation = 0.12, but IV = 0.26 — a 2× gap revealing non-linear structure. Young borrowers (<25) carry nearly double the average default rate, yet the relationship is not U-shaped; binning captures this structure better than a continuous feature.

#### 2. Feature Engineering — Business-Driven

| Hypothesis | Feature | Logic |
|------------|---------|-------|
| Severe delinquency outweighs mild | `total_past_due_weighted` | Weights 45/75/120 = midpoint of each DPD interval |
| Intensity matters more than raw count | `dpd_per_loan` | "Problem days" normalised by number of credit lines |
| Systemic violators are a distinct risk tier | `is_chronic_delinquent` | Delinquencies in ALL three DPD categories simultaneously |
| High utilisation + delinquency = compounding risk | `util_x_past_due` | Interaction of the two strongest predictors |
| Portfolio complexity amplifies delinquency | `complexity_risk` | Total past due × number of credit lines |

`total_past_due_weighted` achieved **IV = 1.28** — the highest in the dataset, surpassing any original feature. New features collectively contribute a meaningful share of RF feature importance, confirming the value of hypothesis-driven engineering.

All features were constructed **before** imputation — protecting the information content of missing-value flags.

#### 3. Modelling Strategy

**Why RF + LR rather than XGBoost/CatBoost?**  
An intentional choice: in regulated financial institutions, models must be *explained* to clients and regulators. LR coefficients directly quantify each factor's directional influence; RF feature importance aligns with business intuition. Gradient boosting was deliberately excluded — not for performance reasons, but for compliance reasons.

| Criterion | RandomForest | LogisticRegression | XGBoost / CatBoost |
|-----------|-------------|-------------------|--------------------|
| Interpretability | ✅ Feature importance | ✅ Signed coefficients | ⚠️ Needs SHAP |
| Stability | ✅ High | ✅ Very high | ⚠️ Overfitting risk |
| Regulatory fit (Basel II) | ✅ High | ✅ Very high | ⚠️ Medium |

**Hyperparameter tuning:** Bayesian Optimisation (Optuna, 30 trials each). TPE sampler directs search toward promising regions — 30 Optuna trials typically match or beat 300 GridSearch iterations.

**Validation:** 5-fold stratified OOF cross-validation. Each row appears in the holdout fold exactly once, giving an honest AUC estimate over 100% of training data. StandardScaler is fitted strictly inside each train-fold to prevent scale leakage.

**Ensemble:** optimal RF/LR blend weight found by scanning [0.40, 0.90]. Final weights: **RF × 0.69 + LR × 0.31**.

#### 4. Key Technical Decisions

| Decision | Alternative | Reason |
|----------|------------|--------|
| Median imputation | IterativeImputer | IterativeImputer caused up to 99% mean drift in `absolute_debt` between train and test |
| p99 winsorizing | Outlier deletion | Preserve signal, eliminate extreme scale distortion |
| DPD binning (0/1/2/3+) | Continuous counters | Monotonicity confirmed; non-linearity encoded explicitly |
| Missing-value flags built on raw data | Post-cleaning flags | Flags built after winsorizing lose information about original anomalies |
| Optuna | GridSearchCV | ~3× more efficient per iteration count |

#### 5. Model Interpretation

- **RevolvingUtilisationOfUnsecuredLines** — primary predictor. A borrower at the credit ceiling has no buffer; any income shock triggers immediate default. LR coefficient: positive.
- **total_past_due_weighted / dpd_per_loan** — weighted severity and intensity of past violations. Portfolio-normalised features outperform raw counts.
- **Age (binned)** — LR coefficient: negative. Older borrowers carry lower risk, reflecting accumulated savings and debt management experience.
- **is_chronic_delinquent** — present in only ~1–2% of borrowers, yet default rate ~85%. Disproportionately important for the tail of the risk distribution.

#### 6. Business Value

- At threshold 0.15: model catches **~72% of all defaults**, rejecting only ~15% of good applicants → suitable for high default-cost environments.
- At threshold 0.30: higher precision, lower recall → suitable when the cost of rejecting a good client approaches the cost of a default.
- Calibration curve lies close to the diagonal → predicted probabilities are reliable for **risk-based pricing**, not just binary approval decisions.
- Estimated deployment benefit: **~80% reduction** in manual underwriting time for automated pre-screening.

---

### Results Summary

| Stage | OOF AUC |
|-------|---------|
| Baseline (RF, no feature engineering) | ~0.853 |
| + Data cleaning (winsorizing, error codes, flags) | ~0.857 |
| + Feature Engineering (20+ features) | ~0.862 |
| + RF + LR Ensemble | ~0.864 |
| + Optuna tuning + median imputation | ~0.865 |
| **Kaggle Private Leaderboard** | **0.86720** |

---

### How to Reproduce

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/give-me-some-credit.git
   cd give-me-some-credit
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```
3. Download the competition data from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place `cs-training.csv` and `cs-test.csv` in the `data/` folder.
4. Run the notebook of your choice. Final predictions will be saved to `submissions/submission_final_ensemble.csv`.

### Dependencies

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn >= 1.3`, `optuna`
- Exact versions in `requirements.txt`

---

## 🇷🇺 Русский

> 📌 **English version** is available [above](#-english).

### Описание проекта

Это решение соревнования Kaggle **Give Me Some Credit**. Задача — предсказать вероятность того, что заёмщик допустит **серьёзную финансовую просрочку** (90+ дней) в течение двух лет, используя 10 демографических и поведенческих признаков из кредитного бюро. Метрика — **ROC-AUC**, измеряющая способность модели ранжировать заёмщиков по уровню риска независимо от порога классификации.

Финальный ансамбль достигает **Private AUC = 0.86720** (топ-12%). Решение акцентирует **аналитический подход**, **интерпретируемость** и **применимость в банковском деле** — RF + LR выбраны вместо градиентного бустинга именно ради соответствия требованиям объяснимости Basel II.

---

### Аналитический подход

#### 1. Исследовательский анализ (EDA) — ключевые инсайты

- **Сильный дисбаланс классов**: дефолтников всего 6.7% (соотношение 14:1). `class_weight='balanced'` применён к обеим моделям — без него модель, предсказывающая «норму» для всех, даёт 93% accuracy, но нулевой recall по дефолтам.
- **Пропуски не случайны**: `MonthlyIncome` отсутствует у ~20% клиентов; группа без дохода имеет **+13 п.п. к дефолту**. Факт пропуска — самостоятельный сигнал риска. Бинарный флаг `income_is_missing` создан *до* любого заполнения.
- **DPD-признаки zero-inflated**: 75%+ значений просрочек — нули. Стандартный `pd.qcut` даёт IV ≈ 0. После ручного биннинга 0/1/2/3+ IV вырастает до 0.60–0.88.
- **RevolvingUtilisation > 1.0 — реальный сигнал**: обрезка до 1.0 (логичный бизнес-порог) *снизила* AUC на 0.005. Клиенты с утилизацией > 1.0 имеют **37% дефолтов** против 6% у остальных. Использован порог p99 (1.09).
- **Возраст нелинеен**: корреляция Пирсона = 0.12, IV = 0.26 — разрыв вдвое, свидетельствующий о нелинейной структуре. Биннинг описывает её лучше непрерывного признака.

#### 2. Feature Engineering — бизнес-обоснование

| Гипотеза | Признак | Логика |
|----------|---------|--------|
| Тяжёлая просрочка опаснее лёгкой | `total_past_due_weighted` | Веса 45/75/120 = середина интервала в днях |
| Интенсивность важнее абсолютного числа | `dpd_per_loan` | «Проблемные дни», нормированные на число кредитов |
| Системный нарушитель — особый тип риска | `is_chronic_delinquent` | Просрочки во всех трёх DPD-категориях одновременно |
| Высокая утилизация + просрочки = двойной риск | `util_x_past_due` | Взаимодействие двух сильнейших предикторов |
| Сложность портфеля усиливает просрочки | `complexity_risk` | Сумма просрочек × число кредитных линий |

`total_past_due_weighted` достиг **IV = 1.28** — лучший показатель в датасете. Все признаки созданы **до** импутации, чтобы сохранить информационную ценность флагов пропусков.

#### 3. Стратегия моделирования

**Почему RF + LR, а не XGBoost/CatBoost?**  
Осознанный выбор: в регулируемых финансовых организациях модели должны быть объяснимы клиентам и регуляторам. LR-коэффициенты прямо показывают направление и силу влияния каждого фактора; важность признаков RF соответствует бизнес-логике. Градиентный бустинг был исключён не из-за качества, а из-за требований к интерпретируемости (Basel II).

**Настройка гиперпараметров:** Bayesian Optimisation (Optuna, по 30 итераций на каждую модель). 30 итераций Optuna, как правило, превосходят 300 итераций GridSearch.

**Валидация:** 5-fold стратифицированная OOF кросс-валидация. StandardScaler подгоняется строго внутри каждого train-фолда во избежание утечки масштаба.

**Ансамбль:** оптимальные веса найдены сканированием [0.40, 0.90]. Финал: **RF × 0.69 + LR × 0.31**.

#### 4. Ключевые технические решения

| Решение | Альтернатива | Причина |
|---------|-------------|---------|
| Медианная импутация | IterativeImputer | IterativeImputer давал drift до 99% в `absolute_debt` между train и test |
| Winsorizing p99 | Удаление выбросов | Сохраняем сигнал, устраняем экстремальный масштаб |
| Биннинг DPD (0/1/2/3+) | Непрерывные счётчики | Монотонность подтверждена; нелинейность закодирована явно |
| Флаги на сырых данных | Флаги после очистки | Флаги после winsorizing теряют информацию о первоначальных аномалиях |
| Optuna | GridSearchCV | ~3× эффективнее по числу итераций |

#### 5. Интерпретация модели

- **RevolvingUtilizationOfUnsecuredLines** — главный предиктор. Клиент «на максимуме» лимита живёт от зарплаты до зарплаты; любой финансовый шок — мгновенный дефолт. Коэффициент LR: положительный.
- **total_past_due_weighted / dpd_per_loan** — взвешенная тяжесть и интенсивность просрочек, нормированные на портфель.
- **Возраст (биннированный)** — коэффициент LR: отрицательный. Старшие заёмщики несут меньший риск — накопленные сбережения и опыт.
- **is_chronic_delinquent** — всего ~1–2% клиентов, но дефолт ~85%. Непропорционально важен для «хвоста» распределения риска.

#### 6. Бизнес-ценность

- Порог 0.15: модель выявляет **~72% всех дефолтов**, отклоняя лишь ~15% хороших клиентов — подходит при высокой стоимости дефолта.
- Порог 0.30: выше precision, ниже recall — подходит, когда цена потери хорошего клиента сравнима с дефолтом.
- Calibration Curve близка к диагонали → вероятности пригодны для **риск-ориентированного ценообразования**, а не только бинарного решения.
- Оценочный эффект от внедрения: **~80% сокращения** времени ручного андеррайтинга при автоматическом пре-скрининге.

---

### Итоговые результаты

| Этап | OOF AUC |
|------|---------|
| Бейзлайн (RF, без инженерии признаков) | ~0.853 |
| + Очистка (winsorizing, коды ошибок, флаги) | ~0.857 |
| + Feature Engineering (20+ признаков) | ~0.862 |
| + Ансамбль RF + LR | ~0.864 |
| + Optuna + медианная импутация | ~0.865 |
| **Kaggle Private Leaderboard** | **0.86720** |

---

### Как воспроизвести

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/YOUR_USERNAME/give-me-some-credit.git
   cd give-me-some-credit
   ```
2. Создать виртуальное окружение и установить зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate   # или `venv\Scripts\activate` на Windows
   pip install -r requirements.txt
   ```
3. Скачать данные соревнования с [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) и поместить `cs-training.csv` и `cs-test.csv` в папку `data/`.
4. Запустить ноутбук на выбранном языке. Финальные предсказания сохранятся в `submissions/submission_final_ensemble.csv`.

### Зависимости

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn >= 1.3`, `optuna`
- Точные версии в `requirements.txt`
