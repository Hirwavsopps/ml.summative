# ----------------------------------------------------------------------------------------------
# Integrity audit: missingness, duplication, constant columns, and domain validity.
# ----------------------------------------------------------------------------------------------
TARGET = "Diabetes_binary"
ID_COL = "ID"

# Coding scheme taken from the UCI data dictionary for dataset 891.
VARIABLE_SPEC = {
    "HighBP":              ("binary",  "High blood pressure, self-reported",            {0, 1}),
    "HighChol":            ("binary",  "High cholesterol, self-reported",                {0, 1}),
    "CholCheck":           ("binary",  "Cholesterol check in past 5 years",              {0, 1}),
    "BMI":                 ("integer", "Body mass index (kg/m^2)",                       None),
    "Smoker":              ("binary",  "Smoked >=100 cigarettes in lifetime",            {0, 1}),
    "Stroke":              ("binary",  "Ever told had a stroke",                         {0, 1}),
    "HeartDiseaseorAttack":("binary",  "Coronary heart disease or myocardial infarction",{0, 1}),
    "PhysActivity":        ("binary",  "Physical activity in past 30 days (not job)",    {0, 1}),
    "Fruits":              ("binary",  "Consumes fruit >=1 time per day",                {0, 1}),
    "Veggies":             ("binary",  "Consumes vegetables >=1 time per day",           {0, 1}),
    "HvyAlcoholConsump":   ("binary",  "Heavy alcohol consumption",                      {0, 1}),
    "AnyHealthcare":       ("binary",  "Has any health coverage",                        {0, 1}),
    "NoDocbcCost":         ("binary",  "Could not see doctor in past 12 months (cost)",  {0, 1}),
    "GenHlth":             ("ordinal", "General health, 1=excellent .. 5=poor",          {1, 2, 3, 4, 5}),
    "MentHlth":            ("integer", "Days mental health not good (0-30)",             None),
    "PhysHlth":            ("integer", "Days physical health not good (0-30)",           None),
    "DiffWalk":            ("binary",  "Difficulty walking or climbing stairs",          {0, 1}),
    "Sex":                 ("binary",  "Sex, 0=female 1=male",                           {0, 1}),
    "Age":                 ("ordinal", "13-level age category, 1=18-24 .. 13=80+",       set(range(1, 14))),
    "Education":           ("ordinal", "1=never attended .. 6=college graduate",         set(range(1, 7))),
    "Income":              ("ordinal", "1=<$10k .. 8=>=$75k",                            set(range(1, 9))),
}

FEATURES = [c for c in raw.columns if c not in (ID_COL, TARGET)]

# ---- check 1: missing values ---------------------------------------------------------------
missing = raw.isna().sum()
n_missing_total = int(missing.sum())

# ---- check 2: exact duplicate rows ---------------------------------------------------------
n_dup_rows = int(raw.duplicated().sum())
n_dup_features = int(raw.drop(columns=[ID_COL]).duplicated().sum())

# ---- check 3: constant columns -------------------------------------------------------------
constant_cols = [c for c in raw.columns if raw[c].nunique() <= 1]

# ---- check 4: domain validity --------------------------------------------------------------
domain_violations = {}
for col, (_, _, allowed) in VARIABLE_SPEC.items():
    if allowed is None:
        continue
    bad = set(raw[col].unique()) - allowed
    if bad:
        domain_violations[col] = sorted(bad)

print("=" * 78)
print("INTEGRITY AUDIT")
print("=" * 78)
print(f"Rows                                  : {len(raw):,}")
print(f"Columns (incl. ID and target)         : {raw.shape[1]}")
print(f"Missing cells (total)                 : {n_missing_total}")
print(f"Rows with >=1 missing cell            : {int(raw.isna().any(axis=1).sum()):,}")
print(f"Exact duplicate rows                  : {n_dup_rows:,}")
print(f"Duplicate rows ignoring ID            : {n_dup_features:,}")
print(f"Constant columns                      : {constant_cols if constant_cols else 'none'}")
print(f"Domain violations                     : {domain_violations if domain_violations else 'none'}")
print(f"Unique IDs / rows                     : {raw[ID_COL].nunique():,} / {len(raw):,}")
print()
print("Physical range of every feature:")
rng = pd.DataFrame({
    "min": raw[FEATURES].min(),
    "max": raw[FEATURES].max(),
    "n_unique": raw[FEATURES].nunique(),
}).T
print(rng.to_string())

# The audit is an assertion, not a narrative: if any of these fail the notebook stops here.
assert n_missing_total == 0, "Unexpected missing values"
assert n_dup_rows == 0, "Unexpected duplicate rows"
assert not constant_cols, "Constant column present"
assert not domain_violations, "Out-of-range coded value present"
print("\nAll integrity assertions passed.")

# ---- check 5: duplicated *response profiles* and whether they agree on the label ----------------
# Two respondents who answer all 21 questions identically are not an error -- they are two different
# people who happen to give the same answers.  But if such a profile carries both labels, no model
# using these 21 features can separate them: that is an irreducible error floor, and it is worth
# measuring rather than discovering later.
profile = raw.groupby(FEATURES, sort=False, dropna=False)["Diabetes_binary"]
group_size = profile.size()
group_label_count = profile.nunique()
conflicting_groups = group_label_count[group_label_count > 1]

n_profiles = len(group_size)
n_multi = int((group_size > 1).sum())
n_conflict_groups = len(conflicting_groups)
n_conflict_rows = int(group_size.loc[conflicting_groups.index].sum())
dedup_prevalence = float(raw.drop_duplicates(subset=FEATURES, keep="first")[TARGET].mean())

print()
print("DUPLICATE RESPONSE PROFILES")
print(f"Distinct response profiles            : {n_profiles:,}")
print(f"Rows that are repeats of an earlier row: {len(raw) - n_profiles:,} "
      f"({(len(raw)-n_profiles)/len(raw):.2%} of the cohort)")
print(f"Profiles occurring more than once     : {n_multi:,}  (largest repeats {int(group_size.max())} times)")
print(f"Profiles carrying BOTH labels         : {n_conflict_groups:,} "
      f"({n_conflict_groups/n_multi:.1%} of repeated profiles)")
print(f"Rows sitting in a label-conflicting profile: {n_conflict_rows:,} "
      f"({n_conflict_rows/len(raw):.2%} of the cohort)")
print(f"Prevalence  raw {raw[TARGET].mean():.4f}  vs  de-duplicated {dedup_prevalence:.4f}")
print()
print(f"Interpretation: at most {n_conflict_rows/len(raw):.2%} of the cohort is unlearnable from these")
print("features alone. That is the noise floor of the task, not a defect of the model --")
print("and it bounds achievable accuracy from above no matter how deep the network is.")
