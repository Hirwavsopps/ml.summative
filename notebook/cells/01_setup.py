# ----------------------------------------------------------------------------------------------
# Environment, seeds and paths. Run this cell first; every later cell depends on it.
# ----------------------------------------------------------------------------------------------
import os
import sys

# Thread pinning. Set *before* NumPy / scikit-learn / TensorFlow are imported, because OpenMP and
# BLAS read these once at load time.
#   - one thread makes floating-point reductions bit-reproducible, so the numbers in the report
#     survive a rerun on a different machine;
#   - it also keeps every scikit-learn internal `joblib.Parallel` call on its sequential path,
#     which means nothing depends on worker processes being available.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # silence the TF info banner
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # bit-stable CPU numerics

import json
import time
import random
import hashlib
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42

# ---- resolve the repository root, creating a working layout if this is a bare runtime ----------
def resolve_project_root() -> Path:
    """Return the repo root, or build the directory layout in the current working directory.

    Running from a clone of the repository finds `data/raw` by walking up the tree. Running in a
    hosted runtime (Colab, Kaggle, a bare Jupyter kernel) has no such directory, so instead of
    failing we create the layout here and let the next cell fetch the data. That is what makes
    "Run all" work with no setup step.
    """
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate

    for sub in ("data/raw", "figures", "models", "results", "logs/tensorboard", "src"):
        (here / sub).mkdir(parents=True, exist_ok=True)
    print(f"No repository layout found; created one under {here} and will fetch the data next.")
    return here


ROOT = resolve_project_root()

DATA_RAW = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
for _d in (FIG_DIR, MODEL_DIR, RESULT_DIR, LOG_DIR / "tensorboard"):
    _d.mkdir(parents=True, exist_ok=True)

# ---- global seeding ---------------------------------------------------------------------------
def set_global_seed(seed: int = SEED) -> None:
    """Seed every source of randomness used in this notebook."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed()

# ---- plotting ---------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")           # headless: figures are written to disk, not to a window
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
})

import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, matthews_corrcoef, classification_report, brier_score_loss,
)

import tensorflow as tf
import keras
from keras import layers

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()   # reproducible TF ops on CPU
except Exception:
    pass

print(f"Python       : {platform.python_version()}")
print(f"NumPy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"TensorFlow   : {tf.__version__}")
print(f"Keras        : {keras.__version__}")
print(f"Project root : {ROOT}")
print(f"Global seed  : {SEED}")
