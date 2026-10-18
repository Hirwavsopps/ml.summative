## 1. Environment and reproducibility controls

Machine learning experiments are only meaningful if they can be repeated. Three things make that hard in
this domain and all three are handled here: unordered hash iteration, an unseeded random number generator,
and an unseeded deep-learning framework.

The cell below fixes a single `SEED` that is propagated to Python's `random`, NumPy, scikit-learn and
TensorFlow/Keras, resolves every project path relative to the repository root (so the notebook survives being
moved, and so it can be run in Colab or any other hosted runtime without a setup step -- if the
`data/raw` directory is not found, the layout is created in the working directory instead of failing), and applies one consistent plotting style so that the figures in the report are byte-identical to the
figures produced here.

Nothing is model-specific in this cell; it is pure infrastructure.
