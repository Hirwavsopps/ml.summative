# ----------------------------------------------------------------------------------------------
# Deterministic, leakage-safe feature engineering.
#
# The transform lives in src/preprocessing.py rather than in this notebook so that the fitted
# object can be pickled into models/best_traditional.joblib and still be unpickled by any other
# process -- a class defined in __main__ inside a notebook cannot be.  The canonical copy is in the
# repository; the string below is that same file, so a bare hosted runtime (Colab, Kaggle) can
# materialise it and import it normally instead of failing on a missing module.
# ----------------------------------------------------------------------------------------------
PREPROCESSING_SOURCE = r'''"""
Leakage-safe feature engineering for the diabetes screening study.

Kept in a module rather than in the notebook for one practical reason: the fitted
``FeatureEngineer`` is pickled into ``models/best_traditional.joblib``, and a class defined in
``__main__`` inside a notebook cannot be unpickled from any other process. Importing it from a real
module makes the saved pipeline genuinely portable.

Every engineered feature is derived only from predictor columns. The single feature that touches
dataset-wide statistics -- ``BMI_z_sex`` -- learns its reference mean and standard deviation in
``fit`` and therefore sees only whatever rows are passed to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# WHO body-mass-index cut-points (kg/m^2): underweight, normal, overweight, obese I, II, III
BMI_BINS = [-np.inf, 18.5, 25.0, 30.0, 35.0, 40.0, np.inf]
BMI_LABELS = [0, 1, 2, 3, 4, 5]

CARDIOMETABOLIC = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]
HEALTHY_BEHAVIOURS = ["PhysActivity", "Fruits", "Veggies"]
INVERSE_BEHAVIOURS = ["Smoker", "HvyAlcoholConsump"]

ENGINEERED = [
    "BMI_band", "CardioRisk", "HealthyBehaviour", "CareAccess", "PoorHealthDays",
    "FrequentPoorDays", "AgeBMI", "BMI_z_sex", "SES",
]


class FeatureEngineer:
    """Adds nine clinically motivated features.

    ``fit`` learns the sex-specific BMI reference statistics from the *training* partition only.
    ``transform`` refuses to run before ``fit``, so the class cannot be used in a way that leaks.
    """

    def __init__(self):
        self.bmi_mean_ = None
        self.bmi_std_ = None

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        by_sex = X.groupby("Sex")["BMI"]
        self.bmi_mean_ = by_sex.mean().to_dict()
        self.bmi_std_ = by_sex.std(ddof=0).replace(0, 1.0).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.bmi_mean_ is None:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        out = X.copy()

        # 1. WHO BMI band -- exposes the 25 and 30 kg/m^2 clinical thresholds as an ordinal level
        out["BMI_band"] = pd.cut(out["BMI"], bins=BMI_BINS, labels=BMI_LABELS).astype(int)

        # 2. Cumulative cardiometabolic burden
        out["CardioRisk"] = out[CARDIOMETABOLIC].sum(axis=1).astype(int)

        # 3. Composite healthy-lifestyle score (smoking and heavy drinking count as absent behaviours)
        out["HealthyBehaviour"] = (
            out[HEALTHY_BEHAVIOURS].sum(axis=1) + (1 - out[INVERSE_BEHAVIOURS]).sum(axis=1)
        ).astype(int)

        # 4. Healthcare access composite -- also the main detection-bias channel
        out["CareAccess"] = (
            out["AnyHealthcare"] + out["CholCheck"] + (1 - out["NoDocbcCost"])
        ).astype(int)

        # 5-6. Total and clinically frequent poor-health days
        out["PoorHealthDays"] = (out["MentHlth"] + out["PhysHlth"]).astype(int)
        out["FrequentPoorDays"] = (
            (out["MentHlth"] >= 14) | (out["PhysHlth"] >= 14)
        ).astype(int)

        # 7. Age x BMI interaction -- gives linear models an effect they cannot express alone
        out["AgeBMI"] = (out["Age"] * out["BMI"]).astype(int)

        # 8. Sex-standardised BMI, using training reference statistics only
        ref_mean = out["Sex"].map(self.bmi_mean_).astype(float)
        ref_std = out["Sex"].map(self.bmi_std_).astype(float)
        out["BMI_z_sex"] = (out["BMI"] - ref_mean) / ref_std

        # 9. Socioeconomic composite
        out["SES"] = (out["Income"] + out["Education"]).astype(int)

        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
'''

SRC_DIR = ROOT / "src"
SRC_DIR.mkdir(parents=True, exist_ok=True)
if not (SRC_DIR / "preprocessing.py").exists():
    (SRC_DIR / "preprocessing.py").write_text(PREPROCESSING_SOURCE, encoding="utf-8")
    print(f"Wrote {SRC_DIR / 'preprocessing.py'} (hosted-runtime bootstrap)")

sys.path.insert(0, str(SRC_DIR))

from preprocessing import (  # noqa: E402
    FeatureEngineer, ENGINEERED, BMI_BINS, BMI_LABELS,
    CARDIOMETABOLIC, HEALTHY_BEHAVIOURS, INVERSE_BEHAVIOURS,
)

engineer = FeatureEngineer().fit(X_train)

X_train_fe = engineer.transform(X_train)
X_val_fe = engineer.transform(X_val)
X_test_fe = engineer.transform(X_test)

FULL_FEATURES = FEATURES + ENGINEERED
assert list(X_train_fe.columns) == FULL_FEATURES

print(f"Raw feature count       : {len(FEATURES)}")
print(f"Engineered additions    : {len(ENGINEERED)}  -> {', '.join(ENGINEERED)}")
print(f"Full feature count      : {len(FULL_FEATURES)}")
print(f"Reference BMI means     : female {engineer.bmi_mean_[0]:.2f} kg/m^2 | male {engineer.bmi_mean_[1]:.2f} kg/m^2")
print(f"Shapes  train {X_train_fe.shape}  val {X_val_fe.shape}  test {X_test_fe.shape}")

# ---- how much univariate signal did engineering actually add? ---------------------------------
# Transform the raw cohort once (for analysis only -- it is not used for fitting anything).
_full_cohort = engineer.transform(raw[FEATURES]).assign(target=y_all)
rho_all = _full_cohort.corr(method="spearman")["target"]

eng_signal = (
    pd.DataFrame({"feature": FULL_FEATURES, "spearman_rho": [rho_all[f] for f in FULL_FEATURES]})
    .assign(abs_rho=lambda d: d["spearman_rho"].abs())
    .sort_values("abs_rho", ascending=False)
)

print("\nStrongest univariate associations after engineering (top 10):")
print(eng_signal.head(10).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\nEngineered features inside the top 10:",
      int(eng_signal.head(10)["feature"].isin(ENGINEERED).sum()), "of", len(ENGINEERED))
eng_signal.to_csv(RESULT_DIR / "feature_signal_after_engineering.csv", index=False)
