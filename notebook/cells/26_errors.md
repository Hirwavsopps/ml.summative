### 11.6 Error analysis: who the model gets wrong, and why

Aggregate metrics hide structure. A model at 0.80 recall overall may be at 0.90 for one subgroup and 0.55 for
another, and the aggregate number will not reveal it. This section asks three questions of the best model:

**Question 1 — which features does it actually rely on?**
Permutation importance answers this in a way that coefficient magnitude cannot. Each feature is shuffled
independently and the drop in PR-AUC is measured. It is model-agnostic and, unlike impurity-based importance,
it is not biased toward high-cardinality features — a bias that makes `BMI` look less important than it is in
a random forest.

**Question 2 — is performance uniform across population subgroups?**
Recall and precision are recomputed within age band, general-health rating, BMI band, sex and income band. Any
subgroup where recall collapses is a group the model would systematically fail to screen, and in a clinical
deployment that is an equity problem, not a statistical footnote.

**Question 3 — do the missed cases look different from the caught ones?**
Comparing the mean feature profile of false negatives against true positives shows whether the model is
missing a *type* of patient or just missing at random. If missed cases cluster at low BMI and good self-rated
health, the model is meeting the limits of questionnaire data rather than failing to learn.
