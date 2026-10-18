# Non-Invasive Diabetes Risk Screening from Behavioural and Demographic Indicators

### A comparative study of classical machine learning and deep neural networks on 253,680 BRFSS records

| | |
|---|---|
| **Student** | Crispin Hebert Hirwa |
| **Course** | Introduction to Machine Learning — Summative Assignment (Final Project) |
| **Institution** | African Leadership University |
| **Term** | 2026 September Term |
| **Report** | `report/Final_Report.pdf` |
| **Dataset** | CDC Diabetes Health Indicators (UCI ML Repository ID 891), DOI [10.24432/C53919](https://doi.org/10.24432/C53919) |

---

## What this notebook does

This notebook builds and evaluates a complete classification workflow that predicts whether an adult respondent
is diabetic or pre-diabetic (`Diabetes_binary = 1`) using **only questionnaire and examination variables** —
no blood test, no laboratory result. That constraint is the entire point of the project: it is what makes
the model deployable in a community clinic that owns a weighing scale and a tape measure but not a laboratory.

The notebook is organised as a single top-to-bottom experiment log. Each section opens with a short
statement of what it does and why, and closes with an interpretation of what the output means.

| Section | Contents |
|---|---|
| 1 | Environment, reproducibility controls, and project paths |
| 2 | Data provenance and integrity audit |
| 3 | Exploratory analysis: imbalance, correlation structure, univariate association |
| 4 | Stratified splitting and leakage-safe preprocessing |
| 5 | Feature engineering and its ablation |
| 6 | Classical experiments E1–E5 (Logistic Regression, Random Forest ×2, Histogram Gradient Boosting, SVM) |
| 7 | Learning curves and bias–variance diagnosis |
| 8 | Deep learning experiments E6–E9 (``tf.data`` pipelines, Sequential and Functional APIs, embeddings, class weighting) |
| 9 | Consolidated evaluation: experiment table, confusion matrices, ROC/PR, error analysis |
| 10 | Model persistence and reproduction instructions |

**All nine experiments are executed on a fixed, stratified 70/15/15 split.** The validation split is used for
early stopping and threshold selection; the test split is touched exactly once, in Section 9.
