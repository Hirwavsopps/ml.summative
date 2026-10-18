## 12. Persisting the models and the run metadata

Deliverables that cannot be reloaded are not deliverables. This section writes the two artefacts the assignment
asks for — the best traditional model and the best deep-learning model — plus everything needed to reproduce a
prediction from a raw survey row:

- `models/best_traditional.joblib` — the winning classical estimator bundled with the feature engineer and the
  scaler, so that a single call accepts raw survey rows
- `models/best_deep_learning_savedmodel/` — the winning network in TensorFlow's **SavedModel** format, the
  portable one
- `models/best_deep_learning_config.json` + `models/best_deep_learning.weights.h5` — the architecture and the
  weights separately, which rebuild in a fresh process without SavedModel's signature indirection
- `models/preprocessor.joblib` — the fitted `FeatureEngineer` and `StandardScaler`
- `models/run_metadata.json` — versions, seeds, split sizes, class weights, chosen thresholds and the full
  metric record for all nine experiments, so a reviewer can confirm the saved file matches the reported numbers
  without rerunning anything

### A note on the file format

The notebook writes the network as **SavedModel plus config/weights** rather than as a single `.keras` archive.
That is an environment finding, not a preference: in this build, Keras 3's `.keras` writer stalls indefinitely,
and a three-layer `Sequential` model with no custom layers reproduces the stall, so the cause is the writer
rather than the model. `save_weights`, `get_config()` and `tf.saved_model.save` all complete in well under a
second, and the assignment permits either `.h5` or SavedModel. The substitution is recorded here rather than
silently made, because a reader reproducing this notebook on another machine should know why.

The reload is then **verified**, not assumed: the estimator is reloaded from its bundle and re-scored, and the
network is rebuilt from its saved architecture and weights and re-predicted. Both are compared against the
probabilities recorded during evaluation. A model that does not survive a round-trip is a bug, and this is the
cheapest place to catch one.
