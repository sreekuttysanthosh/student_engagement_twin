# PROJECT LOG: Student Engagement Digital Twin (MSc GenAI capstone)

Replication of Kumi et al. (2025), DOI 10.1109/AIIoT65859.2025.11105257, plus dashboard.
Last updated: 2026-10-25. Project root: `D:\student_engagement_twin`.

**Status legend:** DONE = run and checked. CODE READY = code written, not yet run. PENDING = not started. RISK = may be blocked.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11.9 (venv `.venv`; system Python 3.14 deliberately not used) |
| Key libraries | SDV 2.0.0, SDMetrics 0.32.0, torch 2.14.1+cpu, XGBoost, scikit-learn, SHAP, pandas 2.3.3, matplotlib (added late) |
| Hardware | CPU only. No GPU |
| Dataset | Canvas Network Person-Course (tab-separated), 325,199 rows x 26 columns |
| Config file | `configs/base.yaml` (the spec says `default.yaml`; kept as is to preserve working code) |

---

## 2. Decision log

| # | Decision | Reason | Status |
|---|---|---|---|
| D1 | `completed_%` is a 0-1 fraction, so the threshold is 0.5 | Raw range is 0.006-1.0. A threshold of 50 would label everyone 0 | Verified |
| D2 | Cleaning keeps rows with non-missing `completed_%` | All such rows already have activity (activity minimum is 1.0), giving 26,661 rows | Verified |
| D3 | Stratified 80/20 split, seed 42, before any fitting | Spec requirement. Test set untouched until final evaluation | Verified |
| D4 | KNN imputation (k=5) fitted on train only, rounded to integers | Counts cannot be fractions | Verified |
| D5 | **Option A**: `Engagement_Level` is a categorical generator column | Paper balances classes by conditional sampling, which needs the label inside the generator. Proposal does not forbid it | **Deviation from build spec section 4, to document in PROJECT.md** |
| D6 | Option B (separate labeler) kept as a baseline | Spec-compliant alternative | Labeler built, not yet used to label synthetic rows |
| D7 | CTGAN `log_frequency=False` | Default flattened class share to 0.543 / 0.457 vs real 0.825 / 0.175 | Verified |
| D8 | Pair trends computed manually | SDV report returns NaN for this property in this version combination | Documented limitation |
| D9 | XGBoost parameters fixed in advance (200 trees, depth 4, lr 0.1, seed 42), never tuned | Prevents test-set leakage | Verified |
| D10 | Unbalanced and balanced results compared on the same imbalanced real test set | Test set must reflect real data | Verified |

---

## 3. Issues and library incompatibilities (README material)

| # | Issue | Fix |
|---|---|---|
| I1 | `py` launcher listed a dead Python 3.10 (deleted files) | Installed working Python 3.11.9 |
| I2 | SDV 2.0: `SingleTableMetadata` removed | `Metadata.detect_from_dataframes({table: df})` |
| I3 | SDV 2.0: `cuda=` argument rejected | Argument omitted (`enable_gpu` exists) |
| I4 | SDV 2.0: `fit` needs `{table: df}`; `sample` returns a dict | Dict form plus `unwrap()` helper |
| I5 | SDV 2.0: `Synthesizer.load()` removed | `sdv.utils.load_synthesizer` |
| I6 | `sdv.evaluation.single_table` removed | `from sdv.evaluation import evaluate_quality` |
| I7 | Column Pair Trends = NaN in SDV report | Manual mean of `CorrelationSimilarity` over 6 pairs |
| I8 | CTGAN default distorts class share | `log_frequency=False` |
| I9 | `matplotlib` missing from install list | `%pip install matplotlib`; add to `pyproject.toml` |

Evidence for I2-I7 is in `00_api_checks.ipynb`.

---

## 4. Measured results to date

### 4.1 Data (`data_audit.json`)
Raw 325,199 -> clean 26,661 -> train 21,328 / test 5,333 (all match the paper).
Class share 0.8246 / 0.1754 in train, 0.8247 / 0.1753 in test.
Missing % train: 17.08 / 10.53 / 0.00 / 68.11. Test: 17.10 / 10.09 / 0.00 / 67.84 (nevents / ndays_act / ncontent / nforum_posts).

### 4.2 Labeler (Option B baseline, 5-fold CV on train)
Weighted F1 0.902. Class-1 recall 0.672, F1 0.710. Under-labels class 1 by about 11%.

### 4.3 Generators (300 epochs, seed 42, CPU)
| Generator / variant | Time | Class share (0 / 1) |
|---|---|---|
| CTGAN baseline | 5.8 min | n/a |
| TVAE baseline | 3.0 min | n/a |
| CTGAN target, `log_frequency=False` | 6.0 min | 0.824 / 0.176 |
| TVAE target | 2.0 min | 0.8435 / 0.1565 |
| CTGAN from_column | 9.0 min | 0.8352 / 0.1648 |
| TVAE from_column | 3.6 min | 0.8296 / 0.1704 |

### 4.4 Fidelity against real train (`fidelity_summary.json`)
| Experiment | Shapes | Pairs | Overall |
|---|---|---|---|
| CTGAN random baseline (4 cols) | 0.8942 | 0.9394 | 0.9168 |
| TVAE random baseline (4 cols) | 0.8643 | 0.9338 | 0.8990 |
| CTGAN random + target | 0.9343 | 0.9624 | 0.9483 |
| TVAE random + target | 0.9141 | 0.9273 | 0.9207 |
| CTGAN from_column + target | 0.9433 | 0.9587 | 0.9510 |
| TVAE from_column + target | 0.9297 | 0.9567 | 0.9432 |

Mean missing-value similarity: 0.9985 (`random`, by construction) vs about 0.948 (`from_column`). Largest flag-rate gaps: CTGAN `nevents` +8.7 pts, TVAE `ndays_act` -10.2 pts.
Fake-data correlations are higher than real ones (cause untested; hypothesis: KNN imputation).

### 4.5 XGBoost on real test set (`random` strategy)
| Model | Weighted F1 | Class-1 recall | Class-1 F1 | FP | FN |
|---|---|---|---|---|---|
| Real-train reference | 0.9065 | 0.685 | 0.724 | 192 | 295 |
| CTGAN unbalanced | 0.8669 | 0.489 | 0.581 | 180 | 478 |
| TVAE unbalanced | 0.8956 | 0.626 | 0.686 | 186 | 350 |
| CTGAN balanced | 0.8043 | 0.812 | 0.567 | 984 | 176 |
| TVAE balanced | 0.8300 | 0.902 | 0.626 | 917 | 92 |

Balancing trades missed high-engagement students for false alarms. It is not a uniform improvement.
Paper reference (not a target): unbalanced F1 CTGAN 85%, TVAE 90%. Balanced CTGAN 84%, TVAE 86%.

### 4.6 SHAP (5 models, additivity gaps about 1e-5)
`ncontent` first and `nevents` second in all five models, as in the paper. Ranks 3 and 4 are not interpreted. Local explanations saved for TVAE (TP, FN, FP).

### 4.7 `from_column` generators
Both generators reproduce the direction of the real missingness-engagement link (class-1 share lower when a flag is 1). Strength: `nforum_posts` ratio real 3.2, CTGAN 2.5, TVAE 1.9. TVAE almost lost the `ndays_act` flag (rate 0.0037 vs real 0.1053).

---

## 5. Checklist against the build spec

### Phase A: data and preprocessing
- [x] Download, inspect, validate dataset (DONE)
- [x] Cleaning, target derivation, stratified split, audit JSON (DONE)
- [x] KNN imputation on train only (DONE)
- [x] Both missingness strategies implemented (DONE)
- [x] Labeler (Option B) built and cross-validated (DONE; not yet used to label synthetic rows)

### Phase B: generators and fidelity
- [x] CTGAN and TVAE, three variants each (DONE)
- [x] Balanced sets by conditional sampling (DONE)
- [x] Fidelity: shapes, pairs, overall, missing similarity (DONE)
- [ ] **RealTabFormer** (PENDING, RISK)
- [ ] Balanced `from_column` sets (PENDING)

### Phase C: prediction and explanation
- [x] XGBoost unbalanced and balanced, `random` (DONE)
- [x] SHAP global and local (DONE)
- [ ] XGBoost `from_column` unbalanced (CODE READY, not run)
- [ ] XGBoost `from_column` balanced (PENDING)
- [ ] XGBoost for RealTabFormer (PENDING, after RealTabFormer)
- [ ] SHAP for `from_column` models (PENDING, optional)
- [ ] Option B baseline: label baseline synthetic rows with the labeler and train XGBoost (PENDING, optional)

### Phase D: code quality (spec section 7)
- [ ] Move working code from notebooks into `src/engagement_twin/` modules
- [ ] `pyproject.toml` with pinned dependencies (include matplotlib)
- [ ] Move settings into `configs/base.yaml` (seed, epochs, k, XGBoost params)
- [ ] pytest: target derivation, schema validation, split isolation, missingness restoration, feature order, metrics, prediction validation
- [ ] Smoke test that skips generator training
- [ ] CLI: prepare, train-generators, train-classifier, evaluate, explain
- [ ] Ruff clean, `.gitignore` check, logging instead of print
- [ ] Tidy folders: notebooks into `notebooks/`, `check_cleaning.py` and `inspect_raw.py` into `scripts/`

### Phase E: dashboard
- [ ] Streamlit app reading saved artifacts only (Overview, Prediction, Real vs synthetic, Explainability, What-if)
- [ ] Clear "unavailable" state when artifacts are missing
- [ ] What-if page labelled exploratory and non-causal

### Phase F: documentation
- [ ] README: install, hardware, run order, library incompatibilities (section 3)
- [ ] PROJECT.md: methodology, D5 deviation, limitations, ethical safeguards
- [ ] Final report: results tables from section 4, honest comparison with the paper

---

## 6. Open risks and limitations

1. **RealTabFormer:** slowest and most fragile. CPU-only, 10 epochs, up to 6,600 steps. If it cannot run, document it as blocked. Do not substitute silently. It does not support conditional sampling (paper), so no balanced run.
2. **One seed everywhere.** Differences between generators are not claimed as winners. A multi-seed run would help if time allows.
3. **D5 deviation** (label inside the generator) conflicts with the build spec wording. Consider confirming with the supervisor.
4. **Paper full text not read.** The labeling method is inferred from the summary and abstract. State this in the report.
5. **Manual pair trends** are not guaranteed identical to SDV's, so paper comparisons are indicative.
6. TVAE `from_column` restored data has almost no `ndays_act` NaNs while the real test set has about 10%. Effect on the classifier is untested.
7. Predictions are exploratory decision support. SHAP describes model behaviour, not causes. Missing forum activity is not proof of low engagement. No privacy claim.

---

## 7. Plan and time estimate (working hours, approximate)

| Step | Task | Est. hours |
|---|---|---|
| 1 | XGBoost `from_column` unbalanced (run, interpret, results cell) | 0.5 |
| 2 | Balanced `from_column`: sample, restore, fidelity, XGBoost | 1.5 |
| 3 | RealTabFormer: API check, 10-epoch run, restore, fidelity, XGBoost | 3 to 6 (training on CPU is the uncertain part) |
| 4 | Move code into `src/engagement_twin/`, config, `pyproject.toml` | 4 to 6 |
| 5 | pytest suite and smoke test | 2 to 3 |
| 6 | CLI and ruff cleanup | 1.5 to 2 |
| 7 | Streamlit dashboard | 4 to 6 |
| 8 | README, PROJECT.md, final report text, last full check | 4 to 5 |
| | **Total remaining** | **about 21 to 30 hours (roughly 3 to 5 working days)** |

**If time is short, cut in this order** (last item first): multi-seed runs, SHAP for `from_column`, Option B baseline, balanced `from_column`, RealTabFormer (document as blocked). **Do not cut:** step 1, tests, dashboard basics, README, PROJECT.md.

---

## 8. Run order (for the README)

`01_data` -> `02_preprocessing` -> `03_labeler` -> `04_generators` -> `05_fidelity` -> `06_classification` -> `07_explainability`. `00_api_checks` is evidence only. `04_generators` is safe to re-run (it reuses saved models unless `FORCE_RETRAIN=True`).