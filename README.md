# Non-Invasive Diabetes Risk Screening from Behavioural and Demographic Indicators

**A controlled comparison of classical machine learning and deep neural networks on 253,680 BRFSS survey records.**

Crispin Hebert Hirwa — African Leadership University — Introduction to Machine Learning, Summative Assignment, 2026 September Term.

---

## What this project is

Type 2 diabetes is one of the fastest-growing non-communicable diseases in low- and middle-income countries,
but diagnosing it requires a laboratory test that most primary-care facilities in those settings cannot perform.
This project asks a narrower, tractable question: **how much of the diagnostic signal is recoverable from what a
rural clinic can collect in five minutes** — a questionnaire, a weighing scale and a blood-pressure cuff?

Using 253,680 records from the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015), we predict
diabetic-or-pre-diabetic status from **twenty-one features that contain no laboratory measurement**, and we
compare two modelling paradigms under identical conditions:

| Paradigm | Models |
|---|---|
| **Classical (scikit-learn)** | Logistic regression · Random forest (default and tuned) · Histogram gradient boosting · RBF-kernel SVM |
| **Deep learning (TensorFlow/Keras)** | MLP baseline · regularised MLP with BatchNorm + Dropout · two-branch entity-embedding network (Functional API) · class-weighted network with LR decay |

That is **ten experiments**, plus a **nine-run ablation** that isolates feature engineering, standardisation,
feature selection and duplicate-profile removal.

---

## Headline results

| Experiment | Model | ROC-AUC | PR-AUC | Recall @0.5 | Recall @tuned |
|---|---|---|---|---|---|
| E4 | Hist. Gradient Boosting | 0.8295 | 0.4295 | 0.1620 | 0.5639 |
| E8 | Embeddings + MLP (Functional) | 0.8297 | 0.4260 | 0.1531 | 0.5756 |
| E9 | Best DL + class weights | 0.8294 | 0.4257 | 0.7923 | 0.5941 |
| E7 | MLP 128-64-32 + BN/Dropout | 0.8298 | 0.4239 | 0.1175 | 0.5570 |
| E6 | MLP 64-64 (Sequential) | 0.8286 | 0.4233 | 0.1645 | 0.6215 |
| E3 | Random Forest (tuned) | 0.8238 | 0.4208 | 0.1366 | 0.5905 |
| E1 | Logistic Regression | 0.8239 | 0.4050 | 0.7641 | 0.6090 |
| E2 | Random Forest (default) | 0.7963 | 0.3682 | 0.1858 | 0.6071 |
| E5 | RBF-kernel SVM | 0.8066 | 0.3598 | 0.0456 | 0.6226 |

Three findings are worth stating up front, because each one contradicts an intuition:

1. **The best classical and best deep model are separated by 0.0035 PR-AUC.** On this data, the flexibility of
   a neural network buys almost nothing over a well-specified gradient-boosted ensemble.
2. **The decision threshold matters more than the architecture.** Moving the threshold from 0.5 to the
   validation-tuned value changes recall by up to 0.58 (E5), which is an order of magnitude larger than any
   difference between model families.
3. **The engineered features do not help tree models.** They add +0.0086 PR-AUC to logistic regression and
   −0.0066 to a random forest. The trees already learn the interactions that the engineered features encode,
   so the gains are not additive — a null result that the ablation was designed to detect.

---

## Repository layout

```
.
├── README.md                        ← you are here
├── requirements.txt                 ← pinned, installable from a clean environment
├── .gitignore                       ← excludes data/ and model binaries
│
├── notebook/
│   ├── diabetes_risk_screening.ipynb  ← the graded artefact, executed top to bottom
│   └── cells/                       ← the same cells as plain files (build source)
│
├── src/
│   ├── download_data.py             ← fetches the raw CSV and verifies its SHA-256
│   ├── preprocessing.py             ← FeatureEngineer (importable, so saved models reload)
│   ├── build_notebook.py            ← assembles + executes the notebook, capturing outputs
│   └── build_report_pdf.py          ← renders the IEEE two-column report
│
├── figures/                         ← all ten report figures, 200 dpi
├── models/                          ← best_traditional.joblib, best_deep_learning_savedmodel/
├── results/                         ← every table as CSV (report Tables I–III)
├── logs/tensorboard/                ← one event directory per network training run
└── report/
    ├── content.txt                  ← the report source
    └── Final_Report.pdf             ← 3,500–5,000 word IEEE-style report
```

---

## Data

**CDC Diabetes Health Indicators** — UCI Machine Learning Repository, dataset ID 891,
DOI [10.24432/C53919](https://doi.org/10.24432/C53919). Derived from the CDC's
[BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2014.html) telephone survey.

| Property | Value |
|---|---|
| Rows | 253,680 (12.7× the assignment's 20,000 minimum) |
| Columns | 23 (1 ID, 1 target, 21 features) |
| Missing values | 0 |
| Target | `Diabetes_binary` ∈ {0, 1}, 13.93 % positive |
| Imbalance ratio | 6.19 : 1 |
| SHA-256 | `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964` |

**The raw CSV is deliberately not committed to this repository.** Fetch it with:

```bash
python src/download_data.py
```

which downloads from `https://archive.ics.uci.edu/static/public/891/data.csv`, verifies the digest and the row
count, and writes `data/raw/`. A zipped copy is distributed through the Google Drive folder linked in the report.

---

## Reproducing everything

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py          # fetch + verify the dataset
python src/build_notebook.py         # execute the notebook end to end (~18 min, CPU only)
python src/build_report_pdf.py       # render the report PDF and print the word count
```

Everything is deterministic: one seed (`42`) is propagated to Python, NumPy, scikit-learn and TensorFlow, and
`OMP_NUM_THREADS=1` is pinned before NumPy loads so floating-point reductions happen in a fixed order. With the
pinned versions in `requirements.txt`, rerunning reproduces the experiment table to four decimals. A different
scikit-learn or TensorFlow release can move the last digit, so pin before comparing.

### Filling in the submission links

The three links on the report's first page are the only values that cannot be generated. Set them in one
command, which rewrites the report source and re-renders the PDF:

```bash
python src/set_links.py \
    --code  https://github.com/USERNAME/ml-summative \
    --drive https://drive.google.com/drive/folders/XXXXXXXX \
    --video https://youtu.be/XXXXXXXX
```

### Running in Google Colab

The notebook bootstraps itself, so **Runtime -> Run all** is enough:

- if there is no `data/raw` directory it creates the output folders in the working directory;
- the provenance cell downloads the CSV from UCI and asserts the SHA-256 digest and row count;
- the feature-engineering cell writes `src/preprocessing.py` from an embedded copy before importing it.

Install only the libraries Colab does not already ship, and do **not** install `requirements.txt` there —
it would replace Colab's TensorFlow with `tensorflow-cpu` and can break the runtime:

```python
!pip install -q scikit-learn==1.9.1 matplotlib seaborn
```

Colab's preinstalled scikit-learn and TensorFlow are different releases from the pins, so expect the last
digit of a few metrics to move. The report's figures were produced with the pinned versions.

### Hardware notes

The reference run used **CPU only** — 16 cores, no GPU. `tensorflow-cpu` is pinned in `requirements.txt` to keep
the install at a few hundred megabytes instead of pulling a full CUDA stack. If you have a supported GPU, replace
`tensorflow-cpu==2.21.0` with `tensorflow==2.21.0`; the Keras code is unchanged. Note that TensorFlow ≥ 2.11 has
no native-Windows GPU support — use WSL2 in that case.

---

## Notebook structure

The notebook is organised as a single top-to-bottom experiment log. Every section opens with a statement of what
it does and why, and closes with an interpretation of the output.

| § | Contents |
|---|---|
| 1 | Environment, seeds, thread pinning, path resolution |
| 2 | Data provenance and a four-part integrity audit, including the duplicate-profile analysis |
| 3 | Exploratory analysis: Spearman structure and mutual information screens |
| 4 | Stratified 70/15/15 split with assertions on disjointness and prevalence |
| 5 | Feature engineering via `src/preprocessing.py`, with the leakage guard |
| 6 | Shared evaluation machinery: one metric function, one logger, two pre-declared comparison rules |
| 7 | Experiments E1–E5 (classical), each with a justification of its design change |
| 8 | Ablation A1–A9: feature set, scaling, feature selection, duplicate removal |
| 9 | Learning curves and the bias–variance diagnosis |
| 10 | `tf.data` pipeline, class weights, and the shared training routine |
| 11 | Experiments E6–E9 (deep learning) |
| 12 | Consolidated results: Table II, confusion matrices, ROC/PR, threshold analysis, error analysis |
| 13 | Model persistence with a verified save/load round trip, plus reproduction notes |

---

## Deliverables

| Deliverable | Location |
|---|---|
| Written report (3,500–5,000 words, IEEE style) | `report/Final_Report.pdf` |
| Notebook | `notebook/diabetes_risk_screening.ipynb` |
| Best traditional model | `models/best_traditional.joblib` (estimator + preprocessing in one bundle) |
| Best deep-learning model | `models/best_deep_learning_savedmodel/` (SavedModel) + `best_deep_learning_config.json` + `best_deep_learning.weights.h5` |
| Experiment table | `results/table2_experiments.csv` |
| Run metadata (versions, seeds, thresholds) | `models/run_metadata.json` |
| TensorBoard logs | `logs/tensorboard/` |

---

## Using the saved traditional model

The bundle includes the fitted `FeatureEngineer` and `StandardScaler`, so scoring accepts **raw survey rows**:

```python
import joblib, pandas as pd

bundle = joblib.load("models/best_traditional.joblib")
row = pd.DataFrame([{...}])                     # the 21 raw BRFSS columns

X = bundle["feature_engineer"].transform(row[bundle["raw_feature_order"]])
# the scaler is applied only for the models that were trained standardised
if bundle["uses_scaling"]:
    X = bundle["scaler"].transform(X[bundle["feature_order"]])
risk = bundle["estimator"].predict_proba(X)[:, 1]
flagged = risk >= bundle["decision_threshold"]
```

`bundle["decision_threshold"]` is the F1-optimal threshold selected on the **validation** partition, not the
default 0.5 — using 0.5 on a 6.2:1 imbalanced problem under-flags patients, which is the wrong error to make in
screening.

---

## Licence and attribution

The dataset is distributed by the UCI Machine Learning Repository under its terms of use and derives from CDC
public-domain survey data. Cite it as:

> CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository.
> https://doi.org/10.24432/C53919
