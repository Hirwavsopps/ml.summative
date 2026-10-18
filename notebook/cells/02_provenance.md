## 2. Data provenance and integrity audit

### 2.1 Where the data comes from

The dataset is **CDC Diabetes Health Indicators**, distributed by the UCI Machine Learning Repository as
dataset ID 891 (DOI [10.24432/C53919](https://doi.org/10.24432/C53919)). It is a curated extract of the
**Behavioural Risk Factor Surveillance System (BRFSS) 2015** telephone survey, published by the United
States Centers for Disease Control and Prevention. Each row is one adult respondent; the features are their
answers to the survey plus self-reported height and weight, from which BMI is derived.

It satisfies every constraint of the assignment brief:

| Requirement | Status |
|---|---|
| Original or from an open repository | UCI ML Repository, DOI-minted, publicly downloadable |
| Not a scikit-learn/Keras built-in or toy dataset | Downloaded as a raw CSV from the UCI static host |
| Not synthetic or generated | Real 2015 BRFSS responses, 253,680 respondents |
| ≥ 20,000 raw rows before preprocessing | **253,680 rows** — 12.7× the threshold |
| Categorical target, ≥ 2 classes | `Diabetes_binary` ∈ {0, 1} |

### 2.2 Why this dataset and not a laboratory dataset

The decisive property is that **none of the 21 features is a laboratory measurement.** There is no fasting
glucose, no HbA1c, no oral glucose tolerance test. The strongest predictors available are high blood pressure,
high cholesterol, general health rating and BMI — all obtainable by interview, cuff and scale.

That is not a limitation of the dataset; it is the argument for the project. Glucose-based diagnosis is
out of reach for most primary-care facilities in low- and middle-income countries, where the WHO reports
that laboratory capacity, not clinical intent, is the binding constraint. A model restricted to
questionnaire-and-examination inputs answers a real question: *given only what a rural clinic can collect
in five minutes, how much of the diabetes signal can be recovered, and which modelling paradigm recovers
most of it?*

### 2.3 The integrity audit

`src/download_data.py` fetches the CSV from
`https://archive.ics.uci.edu/static/public/891/data.csv`, verifies its SHA-256 digest and writes it to
`data/raw/`. The cell below does the same thing when the file is absent -- which is the normal case in
Colab -- and then asserts the digest and the row count, so a substituted download stops the run rather
than quietly changing every number downstream. The raw file is **never committed to git** (see `.gitignore`); it is distributed through the
Google Drive folder linked in the report. The cell below re-verifies the digest and prints the audit.
