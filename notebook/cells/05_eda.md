## 3. Exploratory analysis: what actually carries signal

Exploratory analysis here has one job — to decide, on evidence rather than intuition, which features deserve
engineering effort and which are probably noise. Two questions are asked:

**Question 1 (Figure 2): which features are redundant with each other?**
Binary and ordinal variables make Pearson's *r* an unreliable summary, so Spearman's rank correlation is
used throughout. Rank correlation is invariant to any monotone rescaling, which matters because `Income`,
`Education` and `Age` are coded as ordered bands rather than measured quantities — the gap between band 1
and band 2 is not a constant number of dollars, so treating them as interval data would be an assumption
dressed up as a measurement.

**Question 2 (Figure 3): which features carry information about the target *on their own*?**
Two complementary screens are computed. **Spearman's ρ** captures monotone association, which suits ordered
predictors, but it can only detect relationships that are monotone. **Mutual information** captures any
dependence, monotone or not, which is exactly what a tree ensemble or a neural network can exploit. Where the
two disagree, non-monotone structure is present — and that is a direct argument for the engineered features
introduced in Section 5.
