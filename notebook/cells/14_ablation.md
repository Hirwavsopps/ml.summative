## 8. Ablation: is the feature engineering actually doing anything?

Feature engineering is the part of a pipeline that is easiest to assert and hardest to justify. Adding nine
features raises the dimensionality by 43 %, slows every fit, and makes the model harder to explain — so it
must earn its place empirically. This section runs a controlled ablation in which **the only thing that
changes between paired runs is whether the nine engineered features are present.**

Three learners are tested against both feature sets, giving six paired runs. Two further single-variable
experiments are added because they answer questions the main table cannot:

- **A7 (scaling).** E1 standardised its inputs. A7 fits the identical logistic regression on raw, unscaled
  features. Comparing the two establishes whether standardisation is a genuine modelling requirement here or
  a habit, and the solver's convergence behaviour is reported as evidence.
- **A8 (feature selection).** The five features with the lowest mutual information from Figure 3 are removed
  and the gradient-boosted model refitted. If PR-AUC is unchanged, those features are carrying no information
  and can be dropped for parsimony; if it falls, they were contributing through interactions that univariate
  screening cannot see — which is precisely the failure mode of filter-based selection.
- **A9 (duplicate sensitivity).** The audit in Section 2.4 found 25,772 rows that repeat another row's 21
  answers exactly. Repeats that land in different partitions are a leakage channel: a model that has memorised
  a profile scores it again for free. A9 refits the A6 configuration on a de-duplicated cohort of 227,908 rows.
  If PR-AUC barely moves, the reported figures were not inflated by duplication; if it drops sharply, they were.

All ablation runs use **fixed, untuned** hyperparameters so that the comparison isolates the feature set. Only
the interaction of interest is varied; everything else is held constant, which is the definition of a
controlled experiment.

**Figure 10** visualises the paired differences, because a table of near-identical numbers is exactly where
small real effects hide.
