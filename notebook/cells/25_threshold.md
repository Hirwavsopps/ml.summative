### 11.5 Choosing a decision threshold

A classifier outputs a probability. Turning that probability into a decision is a separate act, and it is an
act with clinical consequences that no default can make on the user's behalf.

The default 0.5 is a convention inherited from balanced problems. It encodes the assumption that a false
positive and a false negative are equally costly. In screening for a chronic disease they are not. The cost
of a missed diabetic is an untreated patient progressing toward nephropathy, retinopathy or amputation; the
cost of a false alarm is a confirmatory blood test. The asymmetry is large and it points in one direction:
**the threshold should be set below 0.5.**

The curves below make the trade explicit rather than implicit. They are drawn on the *test* partition for the
best model, but the threshold selected for deployment is the F1-optimal value computed on the **validation**
partition in Section 11.1 — selecting a threshold from the same data used to report its performance would be
a form of leakage, and would overstate the benefit.

Different deployment contexts justify different choices, and the table reports several so the reader can pick:
a triage system that can afford follow-up tests should favour recall; a system that triggers an expensive
diagnostic pathway should favour precision.
