# Specification: Discriminative-Embedding Instance Segmentation with Learned Assignment

## 1. Overview

### 1.1 Working name

**Discriminative Embedding and Learned Assignment Network (DELA-Net)**

The name is provisional. The architecture is intended as a simpler alternative to affinity-guided recurrent grouping. It retains a conventional U-Net backbone and predicts three complementary representations:

1. **Foreground probability** — whether each pixel belongs to a cell.
2. **Directional pixel affinities** — whether neighboring pixels belong to the same cell.
3. **Compact pixel embeddings** — a learned feature vector that pulls pixels from the same instance together and pushes different instances apart.

A lightweight **learned assignment head** then assigns foreground pixels to instance slots. The embedding is used as an input to this head; it is not clustered by DBSCAN, K-means, or another external clustering algorithm.

### 1.2 Main design principle

The model separates the problem into two related stages:

- **Representation learning:** predict foreground, local affinities, and discriminative pixel embeddings.
- **Instance assignment:** use the predicted representations and learned instance slots to produce a per-pixel instance label map.

Unlike the recurrent alternative, this design does **not** propagate messages between neighboring pixels over multiple recurrent steps. The affinity maps provide local boundary/connectivity information, while the embedding space provides a compact instance-level representation. A learned assignment head combines these signals to produce the final instance masks.

The architecture is intended to be relatively inexpensive and straightforward to implement, train, and ablate.

---

## 2. Design goals and constraints

The initial implementation should:

- Use a standard, original U-Net as the backbone, without architectural modifications.
- Accept single-channel brightfield microscopy images.
- Preserve the original integer instance labels in the training masks.
- Predict a complete instance label map in one forward pass.
- Avoid DBSCAN, K-means, watershed, connected-component post-processing, local-maximum detection, and crop classification for instance assignment.
- Keep the instance-assignment mechanism learnable.
- Avoid recurrent pixel-level message passing.
- Keep the number of parameters and inference operations low enough for later backbone and deployment experiments.

This is an experimental architecture specification, not a claim that the proposed combination is already novel or empirically superior. Its value must be established through controlled experiments and ablations.

---

## 3. Inputs and ground-truth representation

### 3.1 Input

For a batch of grayscale images:

\[
X \in \mathbb{R}^{B \times 1 \times H \times W}
\]

The image is normalized using the same preprocessing for training, validation, and inference.

### 3.2 Instance-label map

For each image, the target mask is an integer-valued map:

\[
Y \in \{0,1,\ldots,N\}^{H \times W}
\]

where:

- `0` denotes background.
- Each positive integer denotes one cell instance.
- All cells belong to the same semantic class.
- The integer values are identifiers, not ordinal class labels.

The numerical magnitude of an instance ID must not affect the loss. For example, instance IDs `1`, `2`, and `17` are simply three distinct object identifiers.

### 3.3 Mask handling

- Do not convert instance maps to binary masks before generating instance-level targets.
- Use nearest-neighbor interpolation for geometric transformations of label maps.
- Do not apply image intensity normalization or interpolation to the integer instance IDs.
- After every spatial augmentation, regenerate affinity and embedding targets from the transformed instance map.
- Ignore invalid pixels and invalid neighbor pairs introduced by padding or cropping.

---

## 4. Overall architecture

The initial network consists of:

1. Standard U-Net encoder-decoder.
2. Shared high-resolution feature tensor.
3. Foreground prediction head.
4. Directional affinity prediction head.
5. Pixel embedding prediction head.
6. Learned instance-slot assignment head.

### 4.1 Backbone

Use a conventional U-Net with four encoder downsampling stages, a bottleneck, and four decoder stages.

A typical channel configuration is:

| Stage | Channels | Spatial resolution |
|---|---:|---:|
| Input | 1 | \(H \times W\) |
| Encoder 1 | 64 | \(H \times W\) |
| Encoder 2 | 128 | \(H/2 \times W/2\) |
| Encoder 3 | 256 | \(H/4 \times W/4\) |
| Encoder 4 | 512 | \(H/8 \times W/8\) |
| Bottleneck | 1024 | \(H/16 \times W/16\) |
| Decoder 4 | 512 | \(H/8 \times W/8\) |
| Decoder 3 | 256 | \(H/4 \times W/4\) |
| Decoder 2 | 128 | \(H/2 \times W/2\) |
| Decoder 1 | 64 | \(H \times W\) |

The exact convolution block, normalization, and activation should match the selected baseline U-Net implementation and remain fixed across the first experiments.

Let the final high-resolution decoder feature map be:

\[
F \in \mathbb{R}^{B \times C_F \times H \times W}
\]

For the typical configuration, \(C_F=64\).

### 4.2 Prediction heads

Each head is a small convolutional projection from \(F\). A reasonable initial implementation is a 3×3 convolution, normalization and activation, followed by a 1×1 output convolution. The hidden width can initially be 32 channels.

#### A. Foreground head

Output:

\[
Z_f \in \mathbb{R}^{B \times 1 \times H \times W}
\]

Foreground probability:

\[
P_f = \sigma(Z_f)
\]

This head distinguishes cell pixels from background. It does not distinguish one cell from another.

#### B. Directional affinity head

Output:

\[
Z_a \in \mathbb{R}^{B \times 4 \times H \times W}
\]

The four channels correspond to the following offsets:

\[
\Delta = \{(0,1),(1,0),(1,1),(1,-1)\}
\]

That is:

1. Right.
2. Down.
3. Down-right.
4. Down-left.

Affinity probabilities are:

\[
P_a = \sigma(Z_a)
\]

For a pixel \(p\) and a valid neighbor \(q=p+\Delta_k\), the target affinity is:

\[
A^*_{p,k} =
\begin{cases}
1, & Y_p=Y_q \text{ and } Y_p>0\\
0, & Y_p\ne Y_q \text{ and } Y_p>0 \text{ and } Y_q>0\\
\text{ignore}, & \text{otherwise}
\end{cases}
\]

Thus, affinities are supervised only when both pixels belong to foreground instances. Pairs involving background are ignored in the initial design. This avoids making the affinity head learn background connectivity as if it were an instance-separation decision.

Pairs whose neighbor falls outside the image are also ignored.

The four directions provide local connectivity and boundary cues. They are not themselves used as a hand-coded graph traversal or connected-component algorithm at inference.

#### C. Pixel embedding head

Output:

\[
E \in \mathbb{R}^{B \times D \times H \times W}
\]

Use an initial embedding dimension of:

\[
D=8 \text{ or } 16
\]

The lower dimension is preferable if it provides sufficient separation, because it reduces memory and computation in the assignment head.

Normalize each pixel embedding to unit length:

\[
\hat e_p = \frac{e_p}{\max(\|e_p\|_2,\epsilon)}
\]

where \(\epsilon\) is a small numerical constant.

The embedding head should produce a compact representation in which pixels from the same cell occupy a compact region of the embedding space and pixels from different cells are separated.

---

## 5. Discriminative embedding objective

The embedding head is trained using an instance-discriminative objective with a pull term and a push term.

Let \(S_i\) be the set of pixels belonging to ground-truth instance \(i\), and let \(N\) be the number of instances in an image.

The mean embedding of instance \(i\) is:

\[
\mu_i = \frac{1}{|S_i|}\sum_{p\in S_i}\hat e_p
\]

For numerical stability, normalize \(\mu_i\) when it is used for distance calculations.

### 5.1 Pull loss

The pull loss encourages pixels belonging to the same instance to remain close to their instance mean:

\[
\mathcal L_{\text{pull}} =
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{|S_i|}
\sum_{p\in S_i}
\left[
\max(0,\|\hat e_p-\mu_i\|_2-\delta_v)
\right]^2
\]

Here, \(\delta_v\) is the within-instance distance margin.

Initial value to test:

\[
\delta_v=0.5
\]

This term penalizes only deviations outside the allowed radius. It does not force all embeddings to collapse to an identical vector.

### 5.2 Push loss

The push loss encourages the means of different instances to remain separated:

\[
\mathcal L_{\text{push}} =
\frac{1}{N(N-1)}
\sum_{i\ne j}
\left[
\max(0,2\delta_d-\|\mu_i-\mu_j\|_2)
\right]^2
\]

where \(\delta_d\) is the separation margin.

Initial value to test:

\[
\delta_d=1.5
\]

The margins must be interpreted together with the embedding normalization and should be validated empirically. They are starting points, not universal constants.

For images containing fewer than two instances, define the push loss as zero.

### 5.3 Optional regularization

An optional regularization term can discourage excessively large or unstable instance means:

\[
\mathcal L_{\text{reg}} =
\frac{1}{N}\sum_i \|\mu_i\|_2
\]

If embeddings are explicitly unit-normalized, this term may be unnecessary. Do not include it by default unless training diagnostics show a clear need.

### 5.4 Practical sampling

Computing the pull term over every foreground pixel can be expensive for large images. The initial implementation may:

- Compute each instance mean using all its pixels.
- Sample a fixed maximum number of pixels per instance for the pull term.
- Sample instance pairs for the push term when an image contains many objects.

Sampling must be deterministic under a fixed seed during controlled experiments, or its randomness must be controlled.

The implementation must avoid allowing large instances to dominate the loss solely because they contain more pixels. The per-instance averaging in the formula above is intentional.

---

## 6. Learned instance assignment head

### 6.1 Purpose

The assignment head converts pixel-level features into a discrete instance label map.

It must be a learned component of the neural network. It must not invoke DBSCAN, K-means, watershed, connected components, or another external clustering procedure.

The initial design uses a fixed set of \(K\) learnable instance slots plus one background class.

### 6.2 Inputs

The assignment head receives:

- U-Net features \(F\).
- Pixel embeddings \(\hat E\).
- Foreground probability \(P_f\).
- Directional affinity probabilities \(P_a\).

The affinity channels are compressed by a small projection before being combined with the other features.

Construct an assignment feature tensor:

\[
G = \phi\left(
\operatorname{Concat}
[
F,\hat E,P_f,P_a
]
\right)
\]

where \(\phi\) is a small 1×1 convolutional projection, optionally followed by normalization and a nonlinearity.

An initial projection width of 64 channels is reasonable. This value should be reduced if the assignment head becomes a significant fraction of total model cost.

The assignment head should receive the predicted affinity and foreground probabilities during training, rather than ground-truth maps. This ensures that it learns to operate on the model's actual predictions at inference.

### 6.3 Slot representation

Maintain \(K\) learnable slot vectors:

\[
Q \in \mathbb{R}^{K\times D_s}
\]

where \(D_s\) is the slot dimension, initially 64.

Use one additional background representation, either as a separate learned vector or as a dedicated background logit generated by a convolutional head.

The slots represent candidate cell instances, not semantic cell types.

### 6.4 Image-conditioned slot initialization

A practical starting point is to initialize the slot vectors from learned slot embeddings plus a global image context:

\[
c = \operatorname{GlobalPool}(G)
\]

\[
q_k^{(0)} = q_k^{\text{learned}} + W_c c
\]

This gives the slots image-specific information while retaining distinct learned slot identities.

Avoid initializing every slot with exactly the same vector. Symmetric initialization can cause slots to learn identical behavior.

### 6.5 Pixel-to-slot assignment

Project each pixel feature and slot into a shared assignment dimension:

\[
u_p = W_g G_p
\]

\[
v_k = W_q q_k
\]

Compute a compatibility score:

\[
s_{k,p} =
\frac{u_p^\top v_k}{\sqrt{D_s}}
\]

A small learned bias or an additional compatibility MLP may be tested later, but the dot-product formulation is the preferred initial baseline.

The background score is produced separately:

\[
s_{0,p}=h_{\text{bg}}(G_p)
\]

where \(h_{\text{bg}}\) is a lightweight 1×1 convolution.

Combine the background score and slot scores, then normalize over the background and all \(K\) slots:

\[
P_{k,p} =
\operatorname{softmax}_{k\in\{0,\ldots,K\}}
(s_{k,p})
\]

The output assignment tensor is:

\[
P_{\text{assign}}\in
\mathbb R^{B\times(K+1)\times H\times W}
\]

At inference, the predicted label for pixel \(p\) is:

\[
\hat Y_p=\arg\max_{k\in\{0,\ldots,K\}}P_{k,p}
\]

Slot index 0 is background; positive slot indices are predicted cell instances.

### 6.6 Optional slot refinement

The simplest initial model should use one assignment pass. If this is insufficient, test a small number of slot-refinement steps.

For each slot, aggregate pixel features using its current soft assignment:

\[
r_k =
\frac{\sum_p P_{k,p}u_p}
{\epsilon+\sum_p P_{k,p}}
\]

Update the slot with a lightweight gated update, such as a GRU cell:

\[
q_k^{(t+1)}=\operatorname{GRU}(r_k,q_k^{(t)})
\]

Then recompute pixel-to-slot scores.

This is **slot refinement**, not recurrent pixel-level message passing. It updates a small set of instance representations rather than propagating hidden states across the entire image.

However, the one-pass assignment head is the primary baseline. Compare one pass against one or two refinement steps; do not assume iterative refinement improves performance.

### 6.7 Role of the embeddings and affinities

The three predicted representations have distinct roles:

- Foreground probability identifies cell pixels.
- Affinities provide local evidence about whether neighboring cell pixels belong together or are separated by a boundary.
- Embeddings provide global instance-discriminative information that makes pixels from the same object more compatible with the same slot.

The learned assignment head combines these signals. The affinity map is not converted into connected components, and the embedding map is not clustered by an external algorithm.

### 6.8 Fixed slot capacity

The model can predict at most \(K\) foreground instances in one image.

Choose \(K\) using the empirical distribution of instance counts in the training set. A reasonable starting rule is to use the 99th percentile of instances per image plus a safety margin.

Report the chosen \(K\), the observed maximum instance count, and the fraction of images whose ground-truth count exceeds \(K\).

Do not silently truncate ground-truth instances. If an image contains more than \(K\) instances, either:
- increase \(K\);
- use a documented crop/tiling strategy with an appropriate overlap policy; or
- explicitly exclude the image from a particular controlled experiment and report this limitation.

A fixed \(K\) is a genuine limitation of this design and must be included in the thesis discussion.

---

## 7. Training targets for the assignment head

### 7.1 Matching predicted slots to ground-truth instances

Slot indices have no inherent correspondence to ground-truth instance IDs. Therefore, training must establish a permutation-invariant correspondence between predicted slots and ground-truth objects.

For each image:

1. Construct one binary ground-truth mask for each ground-truth instance.
2. Obtain the soft mask predicted by each slot from \(P_{\text{assign}}\).
3. Calculate a matching cost between every predicted slot and every ground-truth instance.
4. Use Hungarian bipartite matching to find the minimum-cost one-to-one assignment.
5. Apply assignment and mask losses to the matched pairs.
6. Treat unmatched slots as no-object/background slots.

Hungarian matching is used only to construct the training targets. It is not part of inference and does not generate the final segmentation.

### 7.2 Matching cost

A reasonable initial matching cost is:

\[
C_{k,i} =
\lambda_{\text{BCE}}C_{\text{BCE}}(P_k,M_i)
+
\lambda_{\text{Dice}}C_{\text{Dice}}(P_k,M_i)
\]

where:
- \(P_k\) is the soft mask for predicted slot \(k\).
- \(M_i\) is the binary ground-truth mask for instance \(i\).

The matching cost should be calculated over valid image pixels. It may be useful to include a foreground or embedding compatibility term later, but keep the initial cost simple.

### 7.3 Assignment classification target

After matching, each foreground ground-truth pixel is assigned the slot index matched to its instance.

Background pixels receive target index 0.

The assignment classification loss is:

\[
\mathcal L_{\text{assign}} =
\operatorname{CE}(P_{\text{assign}},T_{\text{assign}})
\]

where \(T_{\text{assign}}\) is the per-pixel slot-index target map.

Unmatched slots receive no positive pixel targets. The softmax classification objective should therefore discourage them from claiming pixels.

### 7.4 Matched mask loss

For each matched slot-instance pair, optionally calculate:

\[
\mathcal L_{\text{mask}} =
\frac{1}{N}
\sum_i
\left[
\operatorname{BCE}(P_{\pi(i)},M_i)
+
\operatorname{DiceLoss}(P_{\pi(i)},M_i)
\right]
\]

where \(\pi(i)\) is the slot matched to instance \(i\).

This term provides direct object-mask supervision, but it overlaps partly with per-pixel assignment cross-entropy. Therefore, treat it as an ablation rather than assuming it is necessary.

If an image contains no foreground instances, the matched mask loss is zero.

---

## 8. Complete loss function

The initial full objective is:

\[
\mathcal L =
\lambda_f\mathcal L_{\text{foreground}}
+\lambda_a\mathcal L_{\text{affinity}}
+\lambda_e\mathcal L_{\text{embedding}}
+\lambda_{\text{assign}}\mathcal L_{\text{assign}}
+\lambda_{\text{mask}}\mathcal L_{\text{mask}}
\]

with:

\[
\mathcal L_{\text{embedding}} =
\mathcal L_{\text{pull}}+\mathcal L_{\text{push}}
\]

### 8.1 Foreground loss

Use binary cross-entropy with logits and Dice loss:

\[
\mathcal L_{\text{foreground}} =
\mathcal L_{\text{BCE}}(Z_f,Y_f)
+
\mathcal L_{\text{Dice}}(P_f,Y_f)
\]

where \(Y_f=1\) for any positive instance ID and 0 for background.

### 8.2 Affinity loss

Use masked binary cross-entropy with logits:

\[
\mathcal L_{\text{affinity}} =
\frac{
\sum_{p,k} M_{p,k}
\operatorname{BCEWithLogits}(Z_{a,p,k},A^*_{p,k})
}{
\sum_{p,k}M_{p,k}+\epsilon
}
\]

where \(M\) marks valid affinity pairs.

### 8.3 Initial loss weights

Start with:

| Loss | Initial weight |
|---|---:|
| Foreground | 1.0 |
| Affinity | 1.0 |
| Embedding | 1.0 |
| Assignment | 1.0 |
| Matched mask | 0.0 initially; test 0.5 or 1.0 in an ablation |

These values are starting points only. Loss magnitudes can differ substantially, especially for discriminative embedding losses. Log each component separately and inspect gradient magnitudes or gradient norms before tuning weights.

A useful first experiment is to train without the optional matched mask loss, then enable it as a controlled comparison.

---

## 9. Inference procedure

Inference must consist entirely of neural-network operations followed by a per-pixel argmax:

1. Normalize the input image.
2. Run the U-Net backbone.
3. Predict foreground logits/probabilities.
4. Predict four directional affinity maps.
5. Predict the pixel embedding map.
6. Construct assignment features from the U-Net features and predicted outputs.
7. Initialize image-conditioned instance slots.
8. Compute pixel-to-slot compatibility scores.
9. Optionally refine slots if the selected model variant uses refinement.
10. Compute the background-plus-slot softmax.
11. Take the argmax over the slot dimension to obtain the instance label map.

No external clustering, watershed, connected-component extraction, or rule-based merging is used.

### 9.1 Optional foreground gating

A later ablation may use foreground probability to suppress slot assignments at pixels with very low foreground confidence. For example, foreground probability could be incorporated as a learned bias or as an additional input to the assignment head.

Do not initially hard-threshold foreground probability before assignment. Hard thresholding can remove faint cell boundaries or dim cell regions and can prevent the assignment head from learning how to use uncertain foreground evidence.

### 9.2 Small-object filtering

Do not include arbitrary minimum-area filtering in the primary model definition. If small-object filtering is evaluated, report it as a separate post-processing variant, with the threshold selected on validation data only.

---

## 10. Training strategy

### 10.1 Recommended staged implementation

Implement the architecture incrementally. This is important because a poor final result should be traceable to a particular component.

**Stage 0 — U-Net foreground baseline**
- Train the standard U-Net to predict foreground versus background.
- Record foreground Dice, precision, recall, and inference cost.

**Stage 1 — Add affinity head**
- Add the four-direction affinity head.
- Train foreground and affinity objectives.
- Confirm that affinity predictions distinguish within-cell neighbor pairs from cross-instance pairs.

**Stage 2 — Add embedding head**
- Add the compact pixel embedding head.
- Train pull and push losses.
- Inspect embeddings on held-out images, using ground-truth instance colors only for visualization.
- Verify that within-instance distances are smaller than between-instance distances.

**Stage 3 — Add one-pass learned assignment**
- Add image-conditioned slots and the pixel-to-slot assignment head.
- Use Hungarian matching to build training targets.
- Train the assignment classification loss.
- Evaluate full instance segmentation.

**Stage 4 — Test matched mask loss**
- Compare assignment CE alone against assignment CE plus matched mask BCE/Dice.

**Stage 5 — Test slot refinement**
- Compare one-pass assignment with one and two slot-refinement steps.

**Stage 6 — Joint fine-tuning**
- Fine-tune the full model end-to-end after the component-wise stages.
- Compare with a jointly trained model from initialization if resources permit.

### 10.2 Data augmentation

All spatial transformations must be applied consistently to images and instance maps. Regenerate affinity and embedding targets after augmentation.

For instance maps:
- Use nearest-neighbor interpolation.
- Preserve integer IDs.
- Do not use bilinear interpolation.
- Ensure that cropping and geometric transformations do not accidentally merge distinct instance IDs.

Photometric transformations may be applied to the image only.

### 10.3 Batch composition

The discriminative embedding objective requires images containing instances. Track the number of instances per image and the number of valid instance pairs.

If some images contain no cells or only one cell, handle the push term as zero rather than producing invalid values.

---

## 11. Experimental plan and ablations

The following experiments are designed to identify whether each representation contributes meaningfully.

| ID | Model variant | Main question |
|---|---|---|
| E0 | U-Net foreground only | What is the foreground baseline? |
| E1 | Foreground + affinities | Do local affinities improve boundary awareness? |
| E2 | Foreground + embeddings | Does discriminative embedding learning provide useful instance information? |
| E3 | Foreground + affinities + embeddings | Do the representations complement each other? |
| E4 | E3 + learned assignment | Can the assignment head produce reliable instance masks? |
| E5 | E4 without affinity input to assignment | Does the assignment head benefit from affinity maps? |
| E6 | E4 without embedding input to assignment | Does the embedding map contribute beyond U-Net features? |
| E7 | E4 without embedding pull/push losses | Does explicit discriminative embedding supervision help? |
| E8 | E4 with different embedding dimensions | What is the smallest adequate embedding dimension? |
| E9 | E4 with different slot capacities \(K\) | How does slot capacity affect quality and efficiency? |
| E10 | One-pass vs one/two slot refinement steps | Is slot refinement worth its cost? |
| E11 | Assignment CE vs CE + matched mask loss | Is direct matched mask supervision beneficial? |
| E12 | Joint vs staged training | Which training strategy is more stable? |

The most important controlled comparisons are E4, E5, E6, and E7. They isolate whether the affinity maps and discriminative embeddings actually improve learned assignment.

### 11.1 Suggested embedding dimensions

Test:

\[
D\in\{4,8,16,32\}
\]

A compact representation is preferred if performance remains comparable.

### 11.2 Suggested slot capacities

Choose candidate values based on the observed object-count distribution, for example:

\[
K\in\{K_{50},K_{90},K_{99+}\}
\]

where each value is derived from training-set counts, with \(K_{99+}\) including explicit headroom. Do not choose arbitrary values without checking the dataset.

### 11.3 Reproducibility

Keep the following fixed when comparing variants:
- Data split.
- Random seeds.
- Input resolution.
- Augmentation pipeline.
- Optimizer and learning-rate schedule, unless the experiment explicitly studies them.
- Training duration and early-stopping criterion.
- Evaluation and post-processing policy.

Run more than one seed for the most important comparisons if computational resources permit.

---

## 12. Evaluation

### 12.1 Segmentation quality

Report instance-level metrics, not only foreground overlap:

- Object-level precision, recall, and F1.
- Average precision at one or more IoU thresholds.
- Average Jaccard Index (AJI).
- Panoptic Quality (PQ), if the implementation and evaluation protocol are appropriate.
- Foreground Dice and IoU.
- Boundary F1 or another boundary-sensitive metric.

Define the matching and IoU conventions used by every object-level metric. Do not compare results from different evaluation conventions as if they were identical.

### 12.2 Error analysis

Classify errors into:
- Merged neighboring cells.
- One cell split into multiple predicted instances.
- Missed cells.
- False-positive instances.
- Boundary leakage.
- Incorrect assignment of pixels to a neighboring slot.
- Slot under-utilization or slot duplication.
- Images whose true instance count exceeds \(K\).

Inspect whether errors correlate with cell size, cell density, weak contrast, touching boundaries, or image location.

### 12.3 Efficiency

Measure:
- Parameter count.
- FLOPs or MACs at the actual input resolution.
- Peak GPU memory during training and inference.
- Inference latency per image.
- Throughput (images per second).
- Checkpoint size.
- CPU inference latency, if CPU deployment is a target.

Measure the assignment head separately as well as the full model. Dense pixel-to-slot scores have memory proportional to:

\[
O(BKHW)
\]

This can become expensive when \(K\), \(H\), and \(W\) are large. The slot count and assignment dimension therefore matter directly to resource efficiency.

---

## 13. Main risks and limitations

### 13.1 Fixed number of slots

The model has a fixed maximum number of predicted instances per image. A high \(K\) avoids truncation but increases assignment memory and computation.

### 13.2 Slot collapse and duplicate slots

Multiple slots may learn similar representations, or some slots may remain unused. Image-conditioned initialization and permutation-invariant matching reduce some difficulties but do not guarantee balanced slot usage.

Monitor:
- Number of active predicted slots.
- Number of matched slots.
- Duplicate or highly overlapping predicted masks.
- Slot usage across images and training epochs.

### 13.3 Embedding-space degeneracy

The embedding head may produce representations that satisfy the pull/push objective but are not useful for pixel assignment. For example, it may separate instance means while leaving ambiguous pixels near instance boundaries poorly represented.

This is why the embedding objective must be evaluated together with the learned assignment head, not only through embedding visualizations.

### 13.4 Affinity imbalance

Most valid neighboring foreground pairs may be within the same cell, depending on image density and annotation style. If positive and negative affinity pairs are imbalanced, inspect per-direction precision/recall and consider class weighting or focal loss as a later ablation.

### 13.5 Assignment-head cost

The dense slot-pixel compatibility tensor scales with \(KHW\). If this becomes a bottleneck, investigate:
- A lower assignment dimension.
- Lower slot capacity justified by the data.
- Chunked computation over slots.
- A lower-resolution assignment head followed by a learned or carefully evaluated upsampling strategy.

Any lower-resolution variant must be checked carefully for small cells and thin boundaries.

### 13.6 Dependence on annotation quality

Pull/push targets are derived directly from instance IDs. Incorrect boundaries, inconsistent annotation conventions, or merged labels can teach the embedding space incorrect separation behavior.

### 13.7 Novelty

Foreground prediction, affinity learning, discriminative embeddings, slot-based assignment, and Hungarian matching are all established ideas. The thesis contribution should therefore be framed around the specific architecture, its efficiency-oriented design, its microscopy adaptation, and the empirical findings—not around claiming that these individual mechanisms are unprecedented.

---

## 14. Implementation checklist

### Data and targets
- [ ] Load integer instance maps without converting IDs to binary.
- [ ] Apply nearest-neighbor interpolation to masks.
- [ ] Generate foreground targets.
- [ ] Generate four-direction affinity targets and validity masks.
- [ ] Generate per-instance embedding supervision from transformed masks.
- [ ] Handle images with zero or one instance safely.
- [ ] Verify that no image contains more instances than the configured slot capacity, or handle such cases explicitly.

### Network
- [ ] Implement the original U-Net baseline.
- [ ] Add foreground head.
- [ ] Add four-channel affinity head.
- [ ] Add normalized low-dimensional embedding head.
- [ ] Add assignment feature projection.
- [ ] Add image-conditioned slot initialization.
- [ ] Add pixel-to-slot compatibility and background score.
- [ ] Add optional slot refinement only after the one-pass baseline works.

### Training
- [ ] Implement pull loss with per-instance averaging.
- [ ] Implement push loss with safe handling of fewer than two instances.
- [ ] Implement masked affinity BCE.
- [ ] Implement Hungarian matching using a documented matching cost.
- [ ] Implement assignment cross-entropy targets.
- [ ] Implement optional matched mask loss as a separate switch.
- [ ] Log each loss component and relevant diagnostics.
- [ ] Check gradients and verify that all heads receive useful gradients.

### Evaluation
- [ ] Export a label map with background ID 0 and unique positive IDs for predicted slots.
- [ ] Evaluate instance-level metrics.
- [ ] Report slot capacity and slot utilization.
- [ ] Measure model and assignment-head efficiency.
- [ ] Run the planned ablations.
- [ ] Inspect merge, split, miss, and false-positive cases.

---

## 15. Recommended first implementation

To avoid introducing too many interacting components at once, the first complete implementation should use:

| Component | Initial choice |
|---|---|
| Backbone | Original U-Net |
| Foreground head | Small convolutional head, 1 output channel |
| Affinity head | Small convolutional head, 4 output channels |
| Embedding head | Small convolutional head, 8-dimensional output |
| Embedding normalization | L2 normalization |
| Embedding objective | Pull + push |
| Assignment feature | Projected concatenation of U-Net features, embedding, foreground and affinity predictions |
| Slot initialization | Learned slot vectors + global image context |
| Assignment | One-pass pixel-to-slot dot-product compatibility |
| Background | Separate learned background logit |
| Matching | Hungarian matching during training |
| Assignment loss | Per-pixel cross-entropy |
| Matched mask loss | Disabled in the first run; add as an ablation |
| Slot refinement | Disabled in the first run |
| Inference | Softmax followed by argmax over background and slots |

This is the cleanest initial test of the central hypothesis:

**Can a compact discriminative pixel embedding, combined with local affinity cues and a lightweight learned slot-assignment head, produce accurate cell instances without recurrent pixel grouping or an external clustering algorithm?**

The first meaningful milestone is not simply obtaining a plausible-looking label map. It is demonstrating, through E4–E7, that the embedding and affinity branches improve instance-level quality over a foreground-only U-Net and over the assignment head without those representations, while keeping the added computational cost modest.
