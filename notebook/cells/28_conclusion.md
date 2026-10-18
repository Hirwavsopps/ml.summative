## 13. Reproduction notes and known limitations

### 13.1 How to reproduce every number in the report

```bash
git clone <repository-url> && cd ml-summative
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py                      # fetches the CSV and verifies its SHA-256
python src/build_notebook.py                     # executes this notebook top to bottom
python src/build_report.py                       # renders report/Final_Report.pdf
```

`src/download_data.py` checks the digest `9f71fda9…34964` and the row count (253,680) before anything
else runs, so a truncated or substituted download fails loudly rather than silently changing the results.
Every experiment is seeded from a single `SEED = 42`, and `OMP_NUM_THREADS = 1` is pinned before NumPy
loads, so the floating-point reductions happen in a fixed order. With the pinned versions in `requirements.txt` the table reproduces to four decimals; a different scikit-learn or TensorFlow release can shift the last digit, so pin before you compare.

### 13.2 Artefacts this notebook produces

| Path | Contents |
|---|---|
| `results/table1_dataset_summary.csv` | variable dictionary (report Table I) |
| `results/table2_experiments.csv` | all nine experiments (report Table II) |
| `results/table3_ablation.csv` | the A1–A9 ablation (report Table III) |
| `results/table4_learning_curves.csv` | learning-curve endpoints and diagnoses |
| `results/feature_importance_permutation.csv` | permutation importances (report Figure 10b) |
| `results/subgroup_*.csv` | recall/precision by age, health, BMI, sex, income |
| `figures/*.png` | all ten report figures at 200 dpi |
| `models/best_traditional.joblib` | best classical model + preprocessing, one-call scoring |
| `models/best_deep_learning_savedmodel/` | best network, TensorFlow SavedModel format |
| `models/best_deep_learning_config.json` + `.weights.h5` | architecture + weights, rebuilds without TF signatures |
| `models/run_metadata.json` | versions, seeds, splits, thresholds, full metric record |
| `logs/tensorboard/` | one event directory per network training run |

### 13.3 Limitations stated plainly

**The features are the ceiling, not the models.** All nine experiments land within a narrow band, and the
reason is visible in Figure 3: the strongest single predictor, self-rated general health, has a Spearman
correlation of 0.29 with the target. Questionnaire data simply does not contain the glucose measurement that
defines the label. No architecture can recover information that was never collected, which is why the
learning curves and not the model zoo are the most informative output of this study.

**Survey self-report bias.** Height and weight are self-reported, so BMI is systematically understated;
diabetes status is self-reported too, so the label itself carries error. Respondents who do not know they
are diabetic are labelled healthy. Both effects push measured performance *down*, so the reported metrics are
conservative estimates of what the same model would achieve against a laboratory-confirmed outcome.

**Detection bias.** `CholCheck`, `AnyHealthcare` and `NoDocbcCost` describe access to care, and access
determines the probability of being *diagnosed* rather than the probability of being *diseased*. A model that
learns these features is partly learning who gets tested. This is the most important ethical caveat in the
study: deploying such a model could widen the gap it appears to close.

**One year, one country.** BRFSS 2015 is a single cross-section of United States adults. Nothing in this
notebook establishes that the relationships transfer to Rwanda or to any other population with a different
age structure, diet, or diagnostic capacity. Transfer would require recalibration on local data; it is a
research programme, not an inference.

**Duplicated profiles.** 25,772 rows repeat another row's 21 answers, and 1,566 profile groups carry both
labels (Section 2.4). The A9 ablation quantifies the effect of removing them rather than assuming it away.

**Subsampled SVM.** E5 was trained on 25,000 rows for tractability and is therefore not a like-for-like
comparison with E3 and E4; it is flagged as such in every table it appears in.
