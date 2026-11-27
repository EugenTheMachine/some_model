# Specification: Affinity-Guided MLP Instance Segmentation

## 1. Overview

### 1.1 Working name

**Affinity-Guided MLP Instance Segmentation Network (AG-MLP-Net)**

The name is provisional. This specification describes a lightweight instance-segmentation architecture for brightfield microscopy. It is intended to retain the useful local relationship modelling of affinity-based segmentation while removing recurrent pixel-level processing such as ConvGRU message passing.

The initial implementation uses a standard U-Net backbone. The backbone should remain unchanged during the first architecture experiments so that the effect of the proposed affinity-guided assignment mechanism can be isolated.

### 1.2 Central idea

The network predicts three kinds of information:

1. **Foreground probability:** whether a pixel belongs to a cell.
2. **Directional affinities:** whether a pixel and each of its selected neighbors belong to the same cell.
3. **Pixel features:** a learned feature representation used to assign pixels to instances.

A single affinity-guided neighborhood aggregation operation combines each pixel's feature vector with features from its immediate neighbors. The resulting representation is processed by a lightweight pointwise MLP (implemented with 1×1 convolutions), which predicts a distribution over a fixed set of instance slots and a background slot.

The complete model is feed-forward. It does not use:
- GRUs or ConvGRUs;
- recurrent pixel-state updates;
- iterative graph propagation;
- DBSCAN or K-means;
- watershed or connected-component extraction;
- local-maximum detection;
- crop proposals or crop classification.

Hungarian matching is used during training to associate predicted slots with ground-truth objects. It is not used at inference.

### 1.3 Research hypothesis

The main hypothesis is:

> A single affinity-weighted neighborhood aggregation step, followed by a lightweight learned assignment head, can combine local boundary/connectivity cues and U-Net features well enough to produce accurate cell instances, without the computational and memory costs of recurrent grouping.

This is a hypothesis to test, not an assumed outcome. In particular, the assignment head and its fixed slot capacity may still contribute significant cost.

---

## 2. Design goals

The initial model should:

- accept single-channel brightfield images;
- use a conventional U-Net encoder-decoder;
- preserve integer-valued instance IDs in the training masks;
- predict a complete instance label map in one forward pass;
- use four directional affinity channels;
- perform exactly one local affinity-guided aggregation operation;
- use a small MLP implemented as pointwise convolutions;
- avoid contrastive pull/push losses in the primary model;
- avoid recurrent processing and external clustering;
- be small enough to support later backbone replacement and deployment experiments.

The first experiments should focus on whether affinity-guided aggregation improves instance-level segmentation compared with a model that predicts the same heads but does not use neighbor aggregation.

---

## 3. Input and ground-truth representation

### 3.1 Input image

For a batch of grayscale microscopy images:

\[
X \in \mathbb{R}^{B\times1\times H\times W}
\]

Use the same image preprocessing during training, validation, and inference.

### 3.2 Instance-label map

For each image:

\[
Y \in \{0,1,\ldots,N\}^{H\times W}
\]

where:
- `0` is background;
- each positive integer represents one cell instance;
- all cells belong to the same semantic class;
- instance IDs are identifiers, not ordinal class labels.

The numerical value of an ID must not affect the loss. For example, IDs 1, 2, and 17 represent three distinct objects without any ordering relationship.

### 3.3 Mask handling

- Preserve integer instance IDs while loading masks.
- Use nearest-neighbor interpolation for all geometric mask transformations.
- Never apply image normalization or bilinear interpolation to instance IDs.
- Apply spatial augmentation consistently to image and mask.
- Generate foreground and affinity targets from the transformed instance map, not from stale pre-augmentation targets.
- Ignore invalid neighbor pairs at image borders or in padded regions.

---

## 4. Overall architecture

The network contains:

1. Standard U-Net backbone.
2. Foreground prediction head.
3. Four-direction affinity prediction head.
4. Pixel feature projection.
5. Single-step affinity-guided neighborhood aggregation.
6. Pointwise MLP assignment head.
7. Background-plus-instance-slot classifier.

### 4.1 Backbone

Use an original U-Net with four downsampling stages, a bottleneck, and four decoder stages.

A representative channel configuration is:

| Stage | Channels | Resolution |
|---|---:|---:|
| Input | 1 | \(H\times W\) |
| Encoder 1 | 64 | \(H\times W\) |
| Encoder 2 | 128 | \(H/2\times W/2\) |
| Encoder 3 | 256 | \(H/4\times W/4\) |
| Encoder 4 | 512 | \(H/8\times W/8\) |
| Bottleneck | 1024 | \(H/16\times W/16\) |
| Decoder 4 | 512 | \(H/8\times W/8\) |
| Decoder 3 | 256 | \(H/4\times W/4\) |
| Decoder 2 | 128 | \(H/2\times W/2\) |
| Decoder 1 | 64 | \(H\times W\) |

The exact U-Net implementation, normalization, and activation should be fixed across the first experiments.

Let the final high-resolution decoder feature map be:

\[
F\in\mathbb{R}^{B\times C_F\times H\times W}
\]

For the representative configuration, \(C_F=64\).

### 4.2 Foreground head

A small convolutional head predicts one foreground logit per pixel:

\[
Z_f\in\mathbb{R}^{B\times1\times H\times W}
\]

\[
P_f=\sigma(Z_f)
\]

The foreground target is:

\[
Y_f(p)=\mathbb{1}[Y(p)>0]
\]

This head distinguishes cells from background but does not distinguish different cells.

### 4.3 Directional affinity head

A small convolutional head predicts four affinity logits:

\[
Z_a\in\mathbb{R}^{B\times4\times H\times W}
\]

\[
P_a=\sigma(Z_a)
\]

The channels correspond to offsets:

\[
\Delta=\{(0,1),(1,0),(1,1),(1,-1)\}
\]

representing right, down, down-right, and down-left neighbors.

For a valid pixel pair \(p\) and \(q=p+\Delta_k\), define:

\[
A^*_{p,k} =
\begin{cases}
1,&Y_p=Y_q\text{ and }Y_p>0\\
0,&Y_p\ne Y_q\text{ and }Y_p>0\text{ and }Y_q>0\\
\text{ignore},&\text{otherwise}
\end{cases}
\]

Pairs involving background are ignored in the initial design. This makes the affinity objective specifically about whether two foreground pixels belong to the same instance. Pairs whose neighbor falls outside the image are also ignored.

The affinity targets are generated directly from the instance-label map after augmentation.

### 4.4 Pixel feature projection

Before neighborhood aggregation, project the U-Net features into a smaller channel dimension:

\[
V=\phi_F(F)
\]

where \(\phi_F\) is a 1×1 convolution, optionally followed by normalization and a nonlinearity.

Use an initial feature width:

\[
C_V=32
\]

A smaller value such as 16 can be tested if memory or latency is a concern. The projection reduces the cost of neighbor aggregation and the subsequent assignment head.

---

## 5. Single-step affinity-guided neighborhood aggregation

### 5.1 Purpose

This is the central operation of AG-MLP-Net.

For each pixel, the network combines its projected feature vector with the projected features of nearby pixels. The predicted affinities control how much information is taken from each neighbor.

Unlike recurrent message passing, this operation is performed exactly once. There are no recurrent hidden states, GRU gates, or repeated propagation iterations.

### 5.2 Neighborhood definition

Use the four directional offsets from the affinity head:

\[
\Delta=\{(0,1),(1,0),(1,1),(1,-1)\}
\]

For a pixel \(p\), let \(q_k=p+\Delta_k\) denote the neighbor associated with direction \(k\).

The initial implementation uses the four predicted affinity probabilities \(a_{p,k}\) to weight the corresponding neighbor features.

A reverse-direction value is required when aggregating from the opposite side of a pixel. For a directed pair \(p\rightarrow q\), use the affinity prediction associated with that pair. For a neighbor on the opposite side, use the corresponding prediction at that neighbor pointing toward \(p\). Implement this mapping explicitly and test it with synthetic affinity maps.

### 5.3 Validity mask

Define \(m_{p,k}\in\{0,1\}\) to indicate whether the neighbor exists within the image and is not padding.

Do not treat an out-of-bounds neighbor as a real zero-valued feature. Exclude it from the aggregation denominator.

### 5.4 Weighted aggregation

Let \(v_p\in\mathbb{R}^{C_V}\) be the projected feature vector at pixel \(p\). Define:

\[
r_p =
\frac{
\sum_{k=1}^{4}m_{p,k}a_{p,k}v_{q_k}
}{
\epsilon+\sum_{k=1}^{4}m_{p,k}a_{p,k}
}
\]

where \(a_{p,k}\in[0,1]\) is the predicted affinity for the pair and \(\epsilon\) is a small constant.

The denominator normalizes the aggregate so that its scale does not grow simply because a pixel has more high-affinity neighbors.

A useful initial choice is to use the raw sigmoid affinity probabilities. Do not threshold them in the primary model: hard thresholding would make the aggregation non-differentiable with respect to the thresholded decisions.

### 5.5 Self-feature and aggregate fusion

Combine the original projected feature and the aggregated neighbor feature:

\[
U_p=[v_p,r_p]
\]

Optionally include the affinity values themselves:

\[
U_p=[v_p,r_p,a_{p,1},a_{p,2},a_{p,3},a_{p,4}]
\]

The second form gives the assignment head direct access to local boundary/connectivity confidence. Use it as the primary configuration.

The resulting tensor is:

\[
U\in\mathbb{R}^{B\times(2C_V+4)\times H\times W}
\]

For \(C_V=32\), the channel count is 68.

### 5.6 Optional residual formulation

An alternative is:

\[
\tilde v_p=v_p+\gamma r_p
\]

where \(\gamma\) is a learnable scalar initialized to a small value, such as 0.1.

This residual form has fewer channels than concatenation, but may discard useful information about the distinction between a pixel's own features and its neighbors' aggregate.

Do not use both fusion formulations simultaneously in the first experiment. Start with concatenation; evaluate residual fusion as an ablation.

### 5.7 Interpretation and limitation

High predicted affinity causes a neighbor's features to contribute more strongly; low affinity suppresses that contribution. This can encourage information to remain within cell interiors rather than mixing across cell boundaries.

However, one local aggregation step does not itself construct a complete cell instance. It only enriches pixel features with local relationship information. The assignment head is still responsible for mapping pixels to complete object identities.

---

## 6. Pointwise MLP assignment head

### 6.1 Purpose

The assignment head converts the affinity-enriched feature map into per-pixel instance-slot scores.

It is implemented using 1×1 convolutions, which are equivalent to applying the same MLP independently at every spatial position.

### 6.2 Assignment input

Concatenate the affinity-enriched tensor with the foreground probability:

\[
G=\operatorname{Concat}(U,P_f)
\]

Thus:

\[
G\in\mathbb{R}^{B\times(2C_V+5)\times H\times W}
\]

With \(C_V=32\), the assignment input has 69 channels.

The original U-Net feature map can also be concatenated directly, but this increases cost. The preferred initial implementation uses the projected feature \(V\), its neighbor aggregate \(R\), the affinity channels, and foreground probability.

### 6.3 MLP structure

Use a small pointwise MLP:

\[
H_1=\operatorname{ReLU}(\operatorname{Conv}_{1\times1}^{69\rightarrow64}(G))
\]

\[
H_2=\operatorname{ReLU}(\operatorname{Conv}_{1\times1}^{64\rightarrow32}(H_1))
\]

\[
Z_{\text{slot}}=\operatorname{Conv}_{1\times1}^{32\rightarrow K}(H_2)
\]

The widths 64 and 32 are starting values. If the model is too expensive, test 32 and 16.

The background score is generated by a separate 1×1 convolution:

\[
Z_{\text{bg}}=\operatorname{Conv}_{1\times1}^{32\rightarrow1}(H_2)
\]

Concatenate background and instance-slot logits:

\[
Z_{\text{assign}}=
\operatorname{Concat}(Z_{\text{bg}},Z_{\text{slot}})
\]

\[
Z_{\text{assign}}\in
\mathbb{R}^{B\times(K+1)\times H\times W}
\]

Apply softmax over the slot dimension:

\[
P_{\text{assign}}=
\operatorname{softmax}_{k}(Z_{\text{assign}})
\]

The resulting tensor contains the predicted probability that each pixel belongs to the background or one of the \(K\) instance slots.

### 6.4 Why a pointwise MLP?

A pointwise MLP has shared weights across all pixels and does not mix spatial positions itself. Spatial context comes from:
- the U-Net receptive field;
- the affinity-guided neighborhood aggregate;
- the affinity and foreground predictions.

This makes the assignment head computationally simple and keeps its behavior easy to isolate in ablation experiments.

### 6.5 Fixed slot capacity

The initial assignment formulation uses \(K\) foreground slots plus background. Therefore, it can represent at most \(K\) cell instances in one image.

Choose \(K\) based on the training-set instance-count distribution, preferably around the 99th percentile plus headroom. Report:
- selected \(K\);
- median, 90th, 99th percentile, and maximum instance counts;
- fraction of images with more than \(K\) instances.

Do not silently truncate objects when the ground-truth count exceeds \(K\). Increase \(K\), use a documented tiling strategy, or explicitly report the limitation.

A larger \(K\) increases the output tensor and assignment computation approximately linearly with \(KHW\).

---

## 7. Training targets and Hungarian matching

### 7.1 Why matching is needed

Predicted slot numbers have no fixed correspondence to ground-truth instance IDs. Slot 3 in one image does not necessarily represent the same object identity as slot 3 in another image.

Training therefore needs a permutation-invariant matching between predicted slots and ground-truth instances.

### 7.2 Predicted slot masks

For each foreground slot \(k\), define its soft mask:

\[
P_k(p)=P_{\text{assign}}(k,p)
\]

The background channel is excluded from object matching.

### 7.3 Ground-truth object masks

For each ground-truth instance \(i\), construct:

\[
M_i(p)=\mathbb{1}[Y(p)=i]
\]

### 7.4 Matching cost

Calculate a pairwise cost between every predicted slot and every ground-truth instance:

\[
C_{k,i}=
\lambda_{\text{BCE}}C_{\text{BCE}}(P_k,M_i)
+
\lambda_{\text{Dice}}C_{\text{Dice}}(P_k,M_i)
\]

Use Hungarian bipartite matching to find the minimum-cost one-to-one assignment.

The initial matching cost can use equal weights for BCE and Dice. Matching is performed independently for each image.

### 7.5 Assignment target map

After matching:
- pixels belonging to ground-truth instance \(i\) receive the predicted slot index matched to \(i\);
- background pixels receive index 0;
- unmatched predicted slots receive no positive pixel targets.

The target is a categorical map:

\[
T_{\text{assign}}\in\{0,\ldots,K\}^{H\times W}
\]

### 7.6 Assignment classification loss

Use per-pixel cross-entropy:

\[
\mathcal L_{\text{assign}}=
\operatorname{CE}(Z_{\text{assign}},T_{\text{assign}})
\]

The implementation should use logits directly with cross-entropy rather than applying softmax before a loss function that expects logits.

If foreground/background imbalance is substantial, evaluate class weighting or a carefully designed pixel weighting scheme. Keep the unweighted formulation as the baseline.

### 7.7 Optional matched mask loss

An optional direct mask loss can be calculated for matched slot-instance pairs:

\[
\mathcal L_{\text{mask}}=
\frac{1}{N}
\sum_i
\left[
\operatorname{BCE}(P_{\pi(i)},M_i)
+
\operatorname{DiceLoss}(P_{\pi(i)},M_i)
\right]
\]

where \(\pi(i)\) is the slot matched to instance \(i\).

This overlaps with per-pixel assignment cross-entropy. Therefore, do not enable it by default in the first experiment. Evaluate it as a separate ablation.

If an image contains no instances, define the matched mask loss as zero.

---

## 8. Complete loss function

The primary model uses:

\[
\mathcal L=
\lambda_f\mathcal L_{\text{foreground}}
+\lambda_a\mathcal L_{\text{affinity}}
+\lambda_{\text{assign}}\mathcal L_{\text{assign}}
\]

An optional matched mask term can be added in a later ablation:

\[
+\lambda_m\mathcal L_{\text{mask}}
\]

There is no contrastive pull/push loss in the primary architecture.

### 8.1 Foreground loss

Use BCE-with-logits plus Dice:

\[
\mathcal L_{\text{foreground}}=
\operatorname{BCEWithLogits}(Z_f,Y_f)
+\operatorname{DiceLoss}(\sigma(Z_f),Y_f)
\]

### 8.2 Affinity loss

Use masked BCE-with-logits over valid, non-ignored neighbor pairs:

\[
\mathcal L_{\text{affinity}}=
\frac{
\sum_{p,k}M_{p,k}
\operatorname{BCEWithLogits}(Z_{a,p,k},A^*_{p,k})
}{
\sum_{p,k}M_{p,k}+\epsilon
}
\]

where \(M_{p,k}=1\) when both endpoints have non-ignored labels and 0
otherwise. A pair is positive only when both endpoints belong to the same
positive instance; foreground-background, background-background, and
different-instance pairs are negative. Out-of-image and ignored-label pairs
are excluded.

Inspect the positive/negative balance separately for each direction. If imbalance is severe, test positive weighting or focal loss as an ablation rather than introducing it immediately.

### 8.3 Initial loss weights

| Loss | Initial weight |
|---|---:|
| Foreground | 1.0 |
| Affinity | 1.0 |
| Assignment | 1.0 |
| Matched mask | 0.0 initially |

These are starting values, not guaranteed optimal weights. Log each component independently and inspect their magnitudes and gradient contributions. Tune only when the training behavior indicates a clear imbalance.

### 8.4 Gradient flow

The affinity probabilities are used in the neighborhood aggregation operation. Therefore, the assignment loss can also backpropagate through the aggregation weights into the affinity head.

This is desirable, but it creates two learning signals for the affinity head:
1. direct affinity supervision;
2. indirect feedback from assignment quality.

Monitor affinity loss and affinity metrics while training. If assignment gradients cause affinity predictions to become poorly calibrated, consider temporarily detaching affinity weights during an early warm-up, then enabling full joint gradients. Treat this as an explicit training ablation.

---

## 9. Inference

The inference procedure is a single feed-forward pass:

1. Normalize the input image.
2. Run the U-Net backbone.
3. Predict foreground logits and probabilities.
4. Predict four-direction affinity logits and probabilities.
5. Project U-Net features to \(V\).
6. Perform one affinity-weighted neighborhood aggregation to obtain \(R\).
7. Concatenate \(V\), \(R\), affinity probabilities, and foreground probability.
8. Run the pointwise MLP assignment head.
9. Apply softmax over background and instance slots.
10. Take argmax over the slot dimension.

The resulting map contains:
- 0 for background;
- positive slot indices for predicted instances.

No clustering or graph traversal is required.

### 9.1 Slot cleanup

The primary inference pipeline should not use arbitrary minimum-area filtering or hand-designed slot merging.

If empty or tiny slots occur, report them as model behavior. A calibrated slot-existence threshold or small-object filter may be evaluated as a separate post-processing variant, with thresholds selected on validation data only.

---

## 10. Training strategy

### 10.1 Incremental implementation

Implement the model in stages to isolate failures.

**Stage 0 — Foreground U-Net**
- Train the original U-Net with the foreground head.
- Record foreground Dice, precision, recall, and efficiency.

**Stage 1 — Add affinity prediction**
- Add the four-direction affinity head.
- Train foreground and affinity objectives.
- Inspect affinity precision/recall by direction and verify that within-instance and cross-instance pairs are distinguished.

**Stage 2 — Add assignment head without aggregation**
- Add the pixel feature projection and pointwise MLP.
- Initially feed the projected features, foreground probability, and affinity probabilities to the MLP, but do not aggregate neighbor features.
- Train with Hungarian matching and assignment CE.
- This is the no-aggregation control model.

**Stage 3 — Enable one-step affinity aggregation**
- Add the single affinity-weighted neighbor aggregation.
- Keep the rest of the architecture and training setup fixed.
- Compare against Stage 2.

**Stage 4 — Joint fine-tuning**
- Fine-tune all heads jointly.
- Compare joint training from initialization, if practical, with staged training.

This sequence makes it possible to determine whether the gain comes from affinity supervision, from giving the assignment head affinity channels, or from explicitly aggregating neighboring features.

### 10.2 Augmentation

Apply spatial transforms identically to image and instance map. Use nearest-neighbor interpolation for the label map. Recompute foreground and affinity targets after each spatial transformation.

Photometric transformations affect only the image.

### 10.3 Batch and edge cases

Handle safely:
- images with zero foreground instances;
- images with one instance;
- instances truncated by crop boundaries;
- images whose instance count exceeds \(K\);
- invalid neighbor pairs at image edges.

For crop-based training, an object cut by a crop boundary remains a partial object with its original instance ID within that crop. This is acceptable if the training/evaluation protocol is consistent.

---

## 11. Experimental plan and ablations

The most important comparison is between an otherwise identical assignment model with and without affinity-guided aggregation.

| ID | Variant | Question |
|---|---|---|
| E0 | U-Net foreground only | Establish foreground baseline |
| E1 | Foreground + affinity heads | Can the model learn useful local affinities? |
| E2 | Foreground + affinity + MLP assignment, no aggregation | Establish learned-assignment baseline |
| E3 | E2 + one-step affinity aggregation | Does explicit neighbor aggregation improve instance segmentation? |
| E4 | E3 without affinity input to assignment MLP | Does direct access to affinity values help? |
| E5 | E3 with residual fusion instead of concatenation | Which fusion is more efficient/effective? |
| E6 | Four vs eight neighbor directions | Does a larger neighborhood help enough to justify the cost? |
| E7 | Different projected feature widths | What is the smallest useful feature width? |
| E8 | Different MLP widths | How small can the assignment head be? |
| E9 | Different slot capacities \(K\) | What is the quality/cost trade-off? |
| E10 | Assignment CE vs CE + matched mask loss | Does direct mask supervision help? |
| E11 | Detached vs fully differentiable affinity weights | Does assignment feedback help affinity learning? |
| E12 | Staged vs joint training | Which training strategy is more stable? |

### 11.1 Suggested feature and MLP widths

Projected feature width:

\[
C_V\in\{16,32,48\}
\]

MLP hidden widths:

- 32 → 16;
- 64 → 32 (primary starting configuration);
- 64 → 64 (larger comparison).

Keep the smallest configuration that does not cause a meaningful loss in instance-level quality.

### 11.2 Four versus eight neighbors

The primary model uses four directions. An eight-neighbor variant can add the reverse directions or use a complete 3×3 neighborhood excluding the center.

If using eight affinities, define and supervise all eight directed neighbor relations consistently. Do not duplicate the same pair with inconsistent target conventions.

The eight-neighbor variant should be treated as an ablation, not a default requirement.

### 11.3 Reproducibility

Keep fixed across architecture comparisons:
- train/validation/test split;
- random seeds;
- image resolution;
- augmentation;
- optimizer and schedule;
- training budget and early stopping;
- evaluation implementation;
- inference and post-processing policy.

Repeat the most important comparisons with multiple seeds if resources permit.

---

## 12. Evaluation

### 12.1 Segmentation quality

Report:
- object-level precision, recall, and F1;
- average precision at specified IoU thresholds;
- AJI;
- PQ, where the evaluation protocol is appropriate;
- foreground Dice and IoU;
- boundary F1 or another boundary-sensitive metric.

Document the object-matching and IoU conventions for every metric.

### 12.2 Error analysis

Separate:
- merged neighboring cells;
- one cell split into multiple predictions;
- missed cells;
- false-positive instances;
- boundary leakage;
- pixels assigned to a neighboring slot;
- unused or duplicate slots;
- images exceeding slot capacity.

Stratify errors by cell size, density, contrast, touching boundaries, and image location when possible.

### 12.3 Efficiency

Measure:
- parameter count;
- FLOPs/MACs at the actual input resolution;
- peak GPU memory during training and inference;
- inference latency per image;
- throughput;
- checkpoint size;
- CPU latency if CPU deployment is relevant.

Measure the following components separately:
1. U-Net backbone and prediction heads;
2. neighborhood aggregation;
3. MLP assignment head;
4. full model.

The aggregation operation has a small fixed number of neighbor reads and weighted sums per pixel. The assignment output, however, scales with the number of slots:

\[
\text{assignment memory}=O(BKHW)
\]

Consequently, increasing \(K\) can dominate the cost even when the MLP itself is small.

---

## 13. Main risks and limitations

### 13.1 One-step aggregation has limited spatial reach

The aggregation directly combines only immediate neighbors. It does not propagate information across an entire cell in the way multiple recurrent iterations could.

The U-Net's receptive field supplies broader context, but whether this is sufficient must be tested.

### 13.2 Affinity prediction errors

If an affinity is incorrectly high across a cell boundary, features from neighboring cells may mix. If it is incorrectly low within a cell, useful within-instance information may be suppressed.

Inspect affinity maps and error cases, not only the final segmentation metrics.

### 13.3 Assignment remains a nontrivial component

Replacing GRU grouping with an MLP does not automatically make the entire model inexpensive. Dense assignment to \(K\) slots has a cost proportional to \(KHW\), and Hungarian matching adds training-time overhead.

Report these costs explicitly.

### 13.4 Fixed slot capacity

The model cannot represent more than \(K\) instances per image. This is a design limitation, not merely an implementation detail.

### 13.5 Slot collapse and under-utilization

Some slots may never be used, while multiple slots may predict overlapping regions. Monitor the number of active slots, matched slots, duplicate masks, and slot utilization across the dataset.

### 13.6 Affinity class imbalance

Within-instance neighbor pairs may substantially outnumber cross-instance pairs. Report class frequencies and per-direction metrics. Only introduce weighting or focal loss if the unweighted baseline demonstrates a problem.

### 13.7 Novelty

U-Net backbones, affinity supervision, local feature aggregation, MLP heads, and slot assignment are established building blocks. The research contribution should be framed as the proposed efficient combination and its adaptation/evaluation for brightfield cell instance segmentation. Do not claim that the individual mechanisms are unprecedented.

---

## 14. Implementation checklist

### Data and targets
- [ ] Preserve integer instance IDs.
- [ ] Use nearest-neighbor interpolation for masks.
- [ ] Generate foreground targets.
- [ ] Generate four-direction affinity targets and validity masks.
- [ ] Test affinity direction indexing with synthetic label maps.
- [ ] Verify border and padding handling.
- [ ] Track the instance-count distribution and choose \(K\) accordingly.

### Network
- [ ] Implement the original U-Net.
- [ ] Add foreground head.
- [ ] Add four-channel affinity head.
- [ ] Add projected feature map \(V\).
- [ ] Implement one-step affinity-weighted neighbor aggregation.
- [ ] Verify that reverse-direction affinities are indexed correctly.
- [ ] Fuse \(V\), \(R\), affinities, and foreground probability.
- [ ] Implement the pointwise MLP assignment head.
- [ ] Produce background plus \(K\) slot logits.

### Training
- [ ] Implement masked affinity BCE-with-logits.
- [ ] Implement foreground BCE + Dice.
- [ ] Implement per-image Hungarian matching.
- [ ] Implement assignment cross-entropy targets.
- [ ] Handle empty-instance images.
- [ ] Log each loss component.
- [ ] Verify gradient flow from assignment loss into the affinity head.
- [ ] Add optional matched mask loss only as an ablation.

### Evaluation
- [ ] Export background ID 0 and unique positive instance IDs.
- [ ] Evaluate object-level metrics.
- [ ] Inspect affinity maps and final instance maps.
- [ ] Report slot capacity and utilization.
- [ ] Measure aggregation and assignment costs separately.
- [ ] Run the no-aggregation control and the planned ablations.

---

## 15. Recommended initial configuration

| Component | Initial setting |
|---|---|
| Backbone | Original U-Net |
| Input channels | 1 |
| Final U-Net feature width | 64 |
| Foreground head | Small convolutional head, 1 output |
| Affinity head | Small convolutional head, 4 outputs |
| Affinity offsets | Right, down, down-right, down-left |
| Projected feature width \(C_V\) | 32 |
| Aggregation | One affinity-weighted neighborhood sum |
| Aggregation normalization | Divide by sum of valid affinity weights + epsilon |
| Aggregation fusion | Concatenate own feature, neighbor aggregate, four affinities |
| Assignment head | 1×1 Conv 69→64→32→K |
| Background head | Separate 1×1 Conv 32→1 |
| Assignment | Softmax over background + K slots |
| Matching | Hungarian matching during training |
| Assignment loss | Per-pixel cross-entropy |
| Foreground loss | BCE-with-logits + Dice |
| Affinity loss | Masked BCE-with-logits |
| Contrastive loss | Not used |
| Recurrent processing | Not used |
| Matched mask loss | Disabled initially |
| Inference | One forward pass + per-pixel argmax |

The exact input width of the assignment MLP assumes \(C_V=32\): concatenating \(V\) (32 channels), \(R\) (32), four affinities, and foreground probability (1) gives 69 channels.

---

## 16. First experiment and success criteria

The first decisive comparison should be:

- **Control:** U-Net + foreground head + affinity head + pointwise MLP assignment, without neighbor aggregation.
- **Proposed:** the same model with one-step affinity-guided neighbor aggregation.

Everything else should be held constant, including slot capacity, loss functions, training schedule, and random seed.

The proposed model should be considered promising only if it:
1. improves instance-level metrics or produces a meaningful improvement in a specific error category;
2. does not obtain that improvement through an unreported post-processing difference;
3. adds modest computational and memory overhead;
4. remains stable across more than one training run, where feasible.

The main question is not whether the aggregation operation makes the architecture more complicated. It is whether the single operation contributes enough measurable instance-segmentation benefit to justify its cost.

**Core architecture:**

\[
\boxed{
X
\rightarrow \text{U-Net}
\rightarrow
\begin{cases}
P_f\\
P_a\\
V
\end{cases}
\rightarrow
\text{one-step affinity aggregation}
\rightarrow
\text{pointwise MLP}
\rightarrow
P_{\text{assign}}
\rightarrow
\hat Y
}
\]

This provides a direct, experimentally testable alternative to recurrent affinity grouping and contrastive embedding-based assignment.