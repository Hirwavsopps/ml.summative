### 11.3 Figure 7 — ROC and precision–recall curves

The two curves answer different questions, and the difference is the single most important thing to understand
about evaluating an imbalanced classifier.

**ROC** plots true-positive rate against false-positive rate. Because the false-positive rate has the number of
negatives in its denominator, and there are 6.2 times as many negatives as positives, a large absolute number
of false alarms barely moves the curve. ROC-AUC therefore looks flattering on imbalanced data and can stay
high even when the positive class is poorly served.

**Precision–recall** plots precision against recall, and its denominator is the number of *predicted*
positives. It is sensitive to exactly the errors ROC hides. For a screening programme, PR-AUC is the closer
match to the decision being made, which is why it was fixed as the primary criterion in Section 6.

Both are shown for the same four models so the divergence is visible rather than asserted.
