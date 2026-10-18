## 10. Deep learning: the `tf.data` input pipeline

### 10.1 Why a `tf.data` pipeline rather than `model.fit(X, y)`

Passing NumPy arrays straight to `fit` works, and it is what most tutorials do. It is also the wrong choice
here, for three reasons that all show up at this dataset's scale:

1. **Batching is not free at 177,576 rows.** A `tf.data` pipeline assembles batches on a background thread,
   overlapping input preparation with the forward and backward passes. Training becomes
   compute-bound rather than input-bound.
2. **Shuffling quality matters for SGD.** `shuffle(buffer)` with an adequately large buffer gives a
   well-mixed stream; `fit(shuffle=True)` reshuffles indices per epoch but must hold the array in place.
3. **Reusability.** The same `Dataset` object is fed to every one of the four architectures below, so the
   input path is written and verified once. That is the modularity the rubric asks for, expressed in code
   rather than in a claim.

`cache()` keeps the tensors in memory after the first epoch (the whole feature matrix is under 50 MB as
float32), `prefetch(AUTOTUNE)` overlaps the next batch's preparation with the current batch's computation, and
`drop_remainder=False` on evaluation ensures the last partial batch is still scored.

### 10.2 Class weighting, and why it is applied to only one experiment

The imbalance that motivated Section 6 also affects gradient descent. With 6.2 negatives per positive, the
average gradient is dominated by the majority class, so the network minimises loss fastest by becoming
confident about the easy negatives. Two standard remedies exist: re-weighting the loss by inverse class
frequency, and resampling the training stream.

In this notebook class weighting is introduced **only in E9**. That is a deliberate experimental design, not
an oversight: E6, E7 and E8 must be comparable to E2–E4, which were trained on the unweighted distribution,
and changing the loss between experiments would confound architecture with loss. E9 then isolates the effect of
the weighting itself.
