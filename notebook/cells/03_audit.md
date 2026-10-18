### 2.4 Structural audit

Before any model sees this table, four things must be established: whether values are missing, whether rows
are duplicated, whether any column is constant (and therefore useless), and whether every value falls inside
the coding scheme the data dictionary specifies. A value of `GenHlth = 7` when the scale runs 1–5 would be a
data-entry error, and an error of that kind silently poisons a distance-based model.

The checks are cheap and they are the difference between a pipeline that is defensible and one that
merely produces a number.

One outcome of the audit needs flagging before it is reached, because it shapes the interpretation of
every result that follows. The dataset contains **no missing values and no exact duplicate rows**, but it
does contain repeated *response profiles*: 25,772 rows are repeats of another row's 21 answers, and 1,566 of
those profile groups carry **both** labels. Two respondents who answer all 21 questions identically and still
differ on diagnosis are not a data-entry error — they are two different people, and no model restricted to
these 21 features can separate them. The audit quantifies that set: it is the **irreducible error floor** of
the task, and it caps achievable accuracy from above regardless of model depth.

This is measured rather than assumed, and Section 8's A9 run tests whether it matters in practice by
refitting the best classical configuration on a de-duplicated cohort.

**Table I** (written to `results/table1_dataset_summary.csv`) is the variable dictionary used in the report;
it is generated from the code rather than typed by hand so the two can never drift apart.
