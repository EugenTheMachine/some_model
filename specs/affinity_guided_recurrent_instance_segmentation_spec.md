# Affinity-Guided Recurrent Instance Segmentation Network

## Architecture and experimental specification

**Document status:** Research specification / initial baseline\
**Version:** 0.1\
**Task:** 2D brightfield microscopy cell instance segmentation\
**Initial backbone:** Original U-Net\
**Primary contribution under investigation:** Learned pixel-affinity
representation combined with recurrent, affinity-guided grouping and
direct instance-slot assignment.

------------------------------------------------------------------------

## 1. Purpose and scope

This document specifies a first research implementation of an
affinity-guided recurrent instance segmentation network. It is intended
to serve as an implementation and experimentation reference that can be
revised as the project develops.

The first implementation deliberately uses a conventional, original
U-Net. The encoder and decoder are not intended to be the final
architectural contribution. They provide a familiar, reproducible
baseline against which the proposed instance-separation mechanism can be
evaluated. The U-Net can later be replaced by a lightweight or otherwise
improved encoder-decoder without changing the definition of the instance
representation or its training targets.

The central research question is:

> Can a network learn local relationships between pixels and use
> recurrent neural message passing to assign foreground pixels to
> individual cell instances, without relying on centroid detection,
> per-instance crop classification, DBSCAN, K-means, or watershed
> reconstruction?

The proposed system has four conceptual stages:

1.  **Feature extraction:** U-Net produces a high-resolution feature
    map.
2.  **Dense prediction:** parallel heads predict foreground probability,
    local pixel affinities, and pixel features.
3.  **Recurrent grouping:** an affinity-guided recurrent module
    exchanges information between neighboring pixels and forms a set of
    instance representations.
4.  **Instance assignment:** a learned assignment head predicts, for
    each foreground pixel, its membership in one of a fixed number of
    instance slots. A final argmax and validity filtering operation
    converts predictions into an integer label image.

The model is designed for instance-labelled training masks. It does not
require additional manually annotated targets: foreground, affinity, and
all auxiliary targets are derived from the instance label map.

### 1.1. Important design decision: direct slot assignment

The primary implementation in this specification uses a **fixed maximum
number of instance slots**, denoted by (K). Each slot represents one
potential cell instance. The network predicts a dense assignment
distribution over these slots for every pixel.

This choice avoids an external clustering algorithm at inference time.
It also means that the model can produce at most (K) instances in one
image. (K) must therefore be selected from the training and validation
data, with sufficient headroom for the densest expected image.

This is a deliberate trade-off. A variable-count affinity-graph
implementation with connected-component labelling is a possible later
ablation, but it is not the primary implementation specified here.

Hungarian matching is used **during training only** to match predicted
slots to ground-truth instances. It is not used to cluster pixels at
inference.

------------------------------------------------------------------------

## 2. Task definition and notation

Let the input image be

\[ I
`\in `{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}C\_{in}`\times `{=tex}H`\times `{=tex}W},
\]

where (B) is batch size, (C\_{in}=1) for grayscale brightfield images,
and (H,W) are image dimensions.

The ground-truth instance label map for one image is

\[ Y `\in `{=tex}{0,1,`\ldots`{=tex},N}\^{H`\times `{=tex}W}, \]

where:

-   (Y(p)=0) denotes background;
-   (Y(p)=i), (i\>0), denotes instance (i);
-   \(N\) is the number of annotated instances in the image;
-   the numeric value of an instance ID has no semantic meaning.

The model returns an integer instance map

\[
`\hat{Y}`{=tex}`\in`{=tex}{0,1,`\ldots`{=tex},K}\^{H`\times `{=tex}W}.
\]

The output convention is:

-   `0`: background;
-   `1...K`: predicted instance IDs;
-   unused slot IDs do not appear in the final map.

The model must support images containing different numbers of instances,
provided the number does not exceed (K).

### 2.1. Pixel and neighborhood notation

Let (p=(y,x)) be a pixel coordinate. Define four directed neighbor
offsets:

\[ `\Delta`{=tex}\_1=(0,1),`\quad`{=tex}
`\Delta`{=tex}\_2=(1,0),`\quad`{=tex}
`\Delta`{=tex}\_3=(1,1),`\quad`{=tex} `\Delta`{=tex}\_4=(1,-1). \]

These correspond to right, down, down-right, and down-left neighbors.

For valid image coordinates, define the neighborhood

\[ `\mathcal{N}`{=tex}(p)={p+`\Delta`{=tex}\_d: d`\in`{=tex}{1,2,3,4}}.
\]

The four directions provide a compact local affinity representation. The
implementation should treat invalid out-of-image neighbors using a
validity mask rather than padding them as background or as a negative
affinity target.

------------------------------------------------------------------------

## 3. Overall architecture

### 3.1. Data flow

``` text
Input image [B, 1, H, W]
          |
          v
     Original U-Net
          |
          v
High-resolution decoder feature map F
          |
          +------------------+--------------------+
          |                  |                    |
          v                  v                    v
   Foreground head     Affinity head       Pixel feature head
   P_f [B,1,H,W]       A [B,4,H,W]          Z [B,Cz,H,W]
          |                  |                    |
          +------------------+--------------------+
                             |
                             v
                 Affinity-guided recurrent
                      grouping module
                             |
                  +----------+----------+
                  |                     |
                  v                     v
           Recurrent pixel         Instance slot
             features H_t          representations S
                  |                     |
                  +----------+----------+
                             |
                             v
                  Pixel-to-slot assignment
                       head / decoder
                             |
                             v
                 Assignment logits Q
                    [B,K+1,H,W]
                             |
                             v
               Softmax over K+1 classes
                             |
                             v
             Background / instance assignment
                             |
                             v
                Integer instance label map
                       [B,H,W]
```

The `K+1` assignment channels consist of one background class and (K)
instance slots. The foreground head is an auxiliary prediction and also
provides foreground context to the grouping module. The assignment head
is responsible for the final mutually exclusive pixel labels.

### 3.2. Original U-Net feature extractor

Use the standard encoder-decoder U-Net as the initial backbone.

Recommended baseline configuration:

-   Input channels: 1.
-   Encoder levels: 4 downsampling stages plus the bottleneck.
-   Two (3`\times3`{=tex}) convolutions per block.
-   ReLU activations.
-   Batch normalization may be enabled or disabled, but the choice must
    remain fixed across the initial ablation series.
-   (2`\times2`{=tex}) max pooling for downsampling.
-   (2`\times2`{=tex}) up-convolutions or transposed convolutions for
    upsampling.
-   Skip connections by concatenation.
-   Final decoder feature map restored to the input spatial resolution.

A representative channel schedule is:

  Stage                 Spatial resolution            Channels
  ------------ --------------------------- -------------------
  Encoder 1            (H`\times `{=tex}W)                  64
  Encoder 2        (H/2`\times `{=tex}W/2)                 128
  Encoder 3        (H/4`\times `{=tex}W/4)                 256
  Encoder 4        (H/8`\times `{=tex}W/8)                 512
  Bottleneck     (H/16`\times `{=tex}W/16)                1024
  Decoder           progressively restored   512, 256, 128, 64

For non-divisible input dimensions, pad or resize consistently and crop
the output back to the original dimensions. Avoid silently changing the
target-mask geometry.

Let the final U-Net decoder feature map be

\[
F`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}C_f`\times `{=tex}H`\times `{=tex}W},
\]

where (C_f) is typically 64 for the configuration above.

The initial implementation should not add attention modules or alter the
U-Net blocks. This keeps the instance-separation mechanism as the main
experimental variable.

### 3.3. Prediction heads

Use small convolutional heads operating on (F). Each head should be
independent initially, apart from sharing the U-Net feature extractor.

#### 3.3.1. Foreground head

Predict one foreground logit per pixel:

\[ L_f = h_f(F), `\qquad`{=tex} P_f=`\sigma`{=tex}(L_f), \]

where

\[
P_f`\in[0,1]`{=tex}\^{B`\times1`{=tex}`\times `{=tex}H`\times `{=tex}W}.
\]

A simple head is a (3`\times3`{=tex}) convolution, ReLU, and
(1`\times1`{=tex}) output convolution.

#### 3.3.2. Affinity head

Predict four local same-instance affinity logits:

\[ L_a=h_a(F), `\qquad `{=tex}A=`\sigma`{=tex}(L_a), \]

where

\[
A`\in[0,1]`{=tex}\^{B`\times4`{=tex}`\times `{=tex}H`\times `{=tex}W}.
\]

Channel (d) predicts whether pixel (p) and pixel (p+`\Delta`{=tex}\_d)
belong to the same ground-truth instance.

The four channels are directional. Do not assume that the output at (p)
and the reverse-direction output at (p+`\Delta`{=tex}\_d) are
automatically identical: the implementation uses a defined set of
forward offsets and constructs the corresponding targets consistently.

#### 3.3.3. Pixel feature head

Predict a compact feature vector for every pixel:

\[ Z=h_z(F), `\qquad`{=tex}
Z`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}C_z`\times `{=tex}H`\times `{=tex}W}.
\]

A practical initial value is (C_z=16) or (32). The feature head may use
a (3`\times3`{=tex}) convolution, normalization, and a
(1`\times1`{=tex}) projection. Optionally L2-normalize the vectors along
the channel dimension.

These features provide content information to the grouping module. They
are not intended to be clustered by DBSCAN or K-means.

------------------------------------------------------------------------

## 4. Affinity target construction

Affinity targets are derived directly from the instance label map.

For each direction (d), define a valid-pair mask (V_d(p)), which is 1
only when both (p) and (p+`\Delta`{=tex}\_d) are valid image pixels.

For valid pairs, define:

\[ A\^{gt}\_d(p)=
```{=tex}
\begin{cases}
1, & Y(p)=Y(p+\Delta_d)>0,\\
0, & Y(p)\ne Y(p+\Delta_d),\\
\text{ignored}, & Y(p)=0\text{ or }Y(p+\Delta_d)=0.
\end{cases}
```
\]

Thus:

-   Neighboring pixels inside the same cell are positive affinity pairs.
-   Neighboring pixels belonging to different cells are negative
    affinity pairs.
-   Pairs involving background are ignored in the affinity loss.
-   Out-of-image pairs are ignored.

The affinity loss must use the valid-pair mask. It must not treat
ignored pairs as negative examples.

### 4.1. Boundary emphasis and class imbalance

Most valid neighboring pairs inside a cell are positive. Depending on
cell density and boundary length, positive and negative pair counts may
be imbalanced.

Recommended initial approach:

-   Use masked binary cross-entropy with logits.
-   Calculate positive and negative counts over valid pairs.
-   Apply a positive-class weight or focal loss only if the training
    distribution demonstrates a meaningful imbalance.
-   Report affinity precision, recall, and F1 separately for positive
    and negative pairs.

Do not add complicated boundary weighting in the first experiment. First
establish whether the basic affinity representation works.

### 4.2. Optional affinity consistency

An optional auxiliary constraint can encourage agreement between
affinity predictions for equivalent undirected neighbor relationships
when both directions are represented. The initial four-offset design
does not require duplicating reverse directions, so this consistency
term is not necessary initially.

------------------------------------------------------------------------

## 5. Affinity-guided recurrent grouping module

This module is the central architectural component.

Its purpose is to let pixel features exchange information with spatial
neighbors, with the amount of information passed controlled by the
learned affinity predictions.

The grouping module must not use connected-component clustering, DBSCAN,
K-means, centroid peak detection, or watershed to determine instance
membership.

### 5.1. Inputs

For one image, the module receives:

-   Pixel feature map (Z), shape (C_z`\times `{=tex}H`\times `{=tex}W).
-   Foreground probability (P_f), shape
    (1`\times `{=tex}H`\times `{=tex}W).
-   Affinity probabilities (A), shape
    (4`\times `{=tex}H`\times `{=tex}W).
-   Optionally, the shared U-Net feature map (F), projected to a smaller
    channel count.

Concatenate or project these features to form the recurrent input:

\[ X=`\psi`{=tex}(\[Z,P_f,A,F'\]), \]

where (`\psi`{=tex}) is a lightweight convolutional projection and (F')
is an optional reduced-channel projection of (F).

For the first implementation, use (Z), (P_f), and (A). Add (F') only as
a controlled ablation.

### 5.2. Initial recurrent state

Initialize a pixel-wise hidden state:

\[ H\^{(0)}=`\operatorname{Conv}`{=tex}\_{3`\times3`{=tex}}(X). \]

The hidden dimension (C_h) should be small relative to the U-Net feature
dimension. Test (C_h=32) first, with 16 and 64 as ablations.

### 5.3. Affinity-guided message passing

At recurrent iteration (t), each pixel receives messages from its four
defined neighbors.

Let (W_d) be a learned linear or (1`\times1`{=tex}) convolutional
transformation for direction (d). Define:

\[ m_p\^{(t)} = `\sum`{=tex}*{d=1}\^{4} `\left[
a_d(p)\,W_d h_{p+\Delta_d}^{(t)}
\right]`{=tex}+ `\sum`{=tex}*{d=1}\^{4} `\left[
a_d(p-\Delta_d)\,W_{-d}h_{p-\Delta_d}^{(t)}
\right]`{=tex}, \]

where:

-   (h_p\^{(t)}) is the hidden vector at pixel (p);
-   (a_d(p)) is the predicted affinity for the pair
    ((p,p+`\Delta`{=tex}\_d));
-   (W_d) and (W\_{-d}) are learned transformations;
-   invalid neighbors are excluded using spatial validity masks.

For a simpler first implementation, directional messages may share
weights. A later ablation can compare shared and direction-specific
transformations.

The affinity value gates message strength. High affinity allows more
information exchange; low affinity suppresses it.

The message can be normalized by the sum of valid incoming affinity
weights:

\[ `\tilde `{=tex}m_p\^{(t)} = `\frac{m_p^{(t)}}{
\epsilon+\sum_{q\in\mathcal N(p)}a_{pq}
}`{=tex}. \]

Normalization prevents hidden-state magnitude from depending excessively
on the number of neighbors or the absolute affinity scale. The
implementation must define whether the normalization includes both
incoming and outgoing directions and apply that convention consistently.

### 5.4. Recurrent update

Update each pixel's hidden state using a ConvGRU or a spatially shared
GRU cell:

\[ H\^{(t+1)} = `\operatorname{GRU}`{=tex} `\left`{=tex}(
H^{(t)},`\tilde `{=tex}M^{(t)},X `\right`{=tex}). \]

Here, (`\tilde `{=tex}M\^{(t)}) is the spatial map of aggregated
messages.

The recurrent cell is shared across iterations. This limits parameter
growth and allows the model to refine grouping over multiple steps.

Initial recommended configuration:

-   Recurrent iterations (T=3).
-   Hidden channels (C_h=32).
-   Shared recurrent weights across iterations.
-   No spatial downsampling inside the grouping module.
-   Residual connection from (H\^{(t)}) to (H\^{(t+1)}), if supported by
    the chosen GRU formulation.

Test (T`\in`{=tex}{1,2,3,5}) after the basic model trains successfully.

### 5.5. Why recurrent message passing may help

Pixels inside one cell are expected to have high affinity along paths
within that cell. Repeated gated message passing can therefore propagate
consistent features across a cell.

At a shared boundary, affinity should be low, suppressing communication
between adjacent cells.

The recurrent module is not guaranteed to separate instances merely
because affinity is high within an object. It must learn a
representation that preserves differences between separate cells while
making within-cell representations coherent. The assignment supervision
described below is essential for this.

A key risk is **over-smoothing**: repeated message passing may make
neighboring instances too similar, especially when the predicted
boundary affinity is incorrect. The number of iterations, hidden
dimension, residual update, and affinity calibration must therefore be
evaluated experimentally.

------------------------------------------------------------------------

## 6. Instance slots and learned pixel assignment

### 6.1. Motivation

Affinity-guided message passing produces refined pixel features, but the
system still needs to assign each foreground pixel to a particular
instance.

The primary implementation uses (K) learned instance slots plus one
background class.

This provides a direct neural assignment mechanism rather than a
conventional clustering algorithm.

### 6.2. Slot representation

Define a trainable set of slot embeddings:

\[ S^{(0)}={s_1^{(0)},`\ldots`{=tex},s_K\^{(0)}}, `\qquad`{=tex}
s_k^{(0)}`\in`{=tex}`\mathbb `{=tex}R^{C_s}. \]

The slot embeddings are learned parameters. They do not represent fixed
cell identities across images. Slot 3 in one image has no required
relationship to slot 3 in another image.

Project recurrent pixel features and slot embeddings into a shared
latent dimension.

### 6.3. Recurrent slot interaction

Use an iterative slot-update mechanism to let slots gather information
from pixel features and refine their representations.

At slot iteration (r), calculate pixel-to-slot compatibility:

\[ e\_{pk}\^{(r)} = `\frac{
(W_q h_p)^\top(W_k s_k^{(r)})
}{
\sqrt{d}
}`{=tex}. \]

Normalize across slots for each pixel:

\[ w\_{pk}\^{(r)} = `\operatorname{softmax}`{=tex}*{k} (e*{pk}\^{(r)}).
\]

Foreground probability may be used as a weighting factor so that
background pixels contribute less to slot updates:

\[ `\bar `{=tex}w\_{pk}\^{(r)} = P_f(p)w\_{pk}\^{(r)}. \]

Aggregate pixel information for each slot:

\[ u_k\^{(r)} = `\frac{
\sum_p \bar w_{pk}^{(r)}W_vh_p
}{
\epsilon+\sum_p\bar w_{pk}^{(r)}
}`{=tex}. \]

Update each slot using a GRU:

\[ s_k\^{(r+1)} = `\operatorname{GRU}`{=tex}\_{slot}
(s_k^{(r)},u_k^{(r)}). \]

A small number of slot updates, for example 2 or 3, should be tested
initially.

This operation is a learned neural grouping mechanism. It does not use a
distance threshold, DBSCAN, K-means, or manually detected seed
locations.

### 6.4. Final assignment logits

After slot refinement, compute a pixel-to-slot compatibility score:

\[ q\_{pk} = `\frac{
(W_q h_p)^\top(W_k s_k)
}{
\sqrt d
}`{=tex} +b_k. \]

Add a separate background logit:

\[ q\_{p0}=h\_{bg}(h_p,P_f(p)). \]

The assignment tensor is:

\[
Q`\in`{=tex}`\mathbb `{=tex}R\^{B`\times`{=tex}(K+1)`\times `{=tex}H`\times `{=tex}W}.
\]

Apply softmax over the (K+1) classes:

\[ P\_{assign}(k`\mid `{=tex}p) =
`\operatorname{softmax}`{=tex}\_{k=0,`\ldots`{=tex},K}(Q_p). \]

Class 0 denotes background; classes (1,`\ldots`{=tex},K) denote instance
slots.

The assignment head should be trained to make all pixels belonging to
the same ground-truth object select the same slot.

### 6.5. Fixed-slot limitation

The architecture can represent no more than (K) instances per image.
Choose (K) from the observed distribution of instance counts, not from
an arbitrary default.

Recommended procedure:

1.  Count ground-truth instances in every training and validation image.
2.  Calculate median, 95th percentile, 99th percentile, and maximum
    counts.
3.  Select an initial (K) above the 99th percentile, with a documented
    safety margin.
4.  Record the fraction of images for which (N\>K).
5.  Exclude or explicitly handle images with more than (K) instances
    during training; never silently truncate ground truth.

If the required (K) makes the slot mechanism too expensive, investigate
tiled inference, a larger but more efficient slot interaction design, or
the variable-count affinity-graph alternative as a separate architecture
variant.

------------------------------------------------------------------------

## 7. Training-time matching and loss functions

### 7.1. Why matching is necessary

Slot indices are permutation-invariant. The network may assign a
particular ground-truth cell to any available slot.

For each training image, the predicted slots must therefore be matched
to ground-truth instances.

Use Hungarian bipartite matching **during training**.

Let the ground-truth instances be (G_1,`\ldots`{=tex},G_N) and predicted
slot masks be (`\hat `{=tex}M_1,`\ldots`{=tex},`\hat `{=tex}M_K).

Construct a matching cost:

\[ C\_{ik} = `\lambda`{=tex}*{bce}C*{bce}(G_i,`\hat `{=tex}M_k) +
`\lambda`{=tex}*{dice}C*{dice}(G_i,`\hat `{=tex}M_k) +
`\lambda`{=tex}*{iou}C*{iou}(G_i,`\hat `{=tex}M_k). \]

Use the Hungarian algorithm to find a one-to-one assignment between
ground-truth instances and a subset of predicted slots.

The matching is computed from detached predictions and is not
differentiated through. Gradients flow through the losses calculated for
the selected matches.

Unmatched slots are trained as empty/no-object slots.

### 7.2. Ground-truth slot targets

For a matched ground-truth instance (G_i) and its assigned slot (k), all
pixels in (G_i) receive slot label (k).

Pixels outside all instances receive background label 0.

The assignment target is therefore a per-pixel categorical map:

\[ T(p)`\in`{=tex}{0,1,`\ldots`{=tex},K}. \]

The mapping from ground-truth object IDs to slot IDs is image-specific
and determined by matching.

### 7.3. Foreground loss

Define:

\[ F\_{gt}(p)=`\mathbb{1}`{=tex}\[Y(p)\>0\]. \]

Use BCE with logits and Dice loss:

\[ `\mathcal `{=tex}L_f = `\mathcal `{=tex}L\_{BCE}(L_f,F\_{gt}) +
`\mathcal `{=tex}L\_{Dice}(P_f,F\_{gt}). \]

Dice should be implemented with a numerical stabilizer and a clearly
documented empty-foreground convention.

### 7.4. Affinity loss

Use masked BCE with logits:

\[ `\mathcal `{=tex}L_a = `\frac{
\sum_{p,d}V_d(p)\,
BCEWithLogits(L_{a,d}(p),A_d^{gt}(p))
}{
\epsilon+\sum_{p,d}V_d(p)
}`{=tex}. \]

The valid mask excludes background pairs and invalid image-border pairs.

If class imbalance is severe, evaluate weighted BCE or focal loss as an
ablation rather than introducing both immediately.

### 7.5. Assignment loss

For every pixel, use categorical cross-entropy over the background and
(K) instance slots:

\[ `\mathcal `{=tex}L\_{assign} = CE(Q,T). \]

Because background may dominate the image, evaluate class weighting or
foreground/background balancing. Avoid allowing background pixels to
overwhelm the instance assignment gradients.

For matched slot masks, also calculate binary mask losses:

\[ `\mathcal `{=tex}L\_{mask} = `\frac{1}{N}`{=tex}
`\sum`{=tex}\_{i=1}\^{N} `\left[
BCE(\hat M_{\pi(i)},G_i)
+
DiceLoss(\hat M_{\pi(i)},G_i)
\right]`{=tex}. \]

Here (`\pi`{=tex}(i)) is the matched slot for ground-truth instance (i).

The per-pixel assignment loss and per-instance mask loss are related but
not identical. Keep both only if ablations show that they provide
complementary supervision.

### 7.6. Optional slot regularization

Potential auxiliary terms include:

-   Slot usage regularization to discourage all objects being assigned
    to one slot.
-   Slot diversity regularization to discourage multiple slots from
    predicting the same object.
-   Entropy regularization to encourage confident assignments after a
    suitable warm-up period.

Do not enable these by default. They can destabilize training or
conflict with the direct assignment targets. First train the model using
foreground, affinity, and assignment supervision.

### 7.7. Total objective

The initial total loss is:

\[ `\boxed{
\mathcal L
=
\lambda_f\mathcal L_f
+
\lambda_a\mathcal L_a
+
\lambda_{assign}\mathcal L_{assign}
+
\lambda_{mask}\mathcal L_{mask}
}`{=tex} \]

A sensible initial experiment is to set all top-level weights to 1, then
inspect the magnitudes and gradient norms of the individual losses. If
one task dominates, adjust weights based on measured training behaviour.

Geometry is intentionally not part of the primary Option A architecture.
It can be introduced later as an additional branch in a separate
experiment, rather than complicating the first affinity-based study.

------------------------------------------------------------------------

## 8. Inference algorithm

Inference must not use ground-truth masks, Hungarian matching, DBSCAN,
K-means, watershed, local maxima, or per-instance crop classification.

### 8.1. Forward pass

For an input image:

1.  Run the image through U-Net.
2.  Generate foreground logits and probabilities.
3.  Generate four affinity maps.
4.  Generate pixel feature maps.
5.  Initialize pixel hidden states.
6.  Run (T) affinity-guided recurrent message-passing iterations.
7.  Initialize and refine (K) learned instance slots.
8.  Calculate background and instance-slot assignment logits.
9.  Apply softmax over background plus all slots.

### 8.2. Integer label reconstruction

For each pixel (p):

\[ `\hat `{=tex}Y(p) =
`\arg`{=tex}`\max`{=tex}*{k`\in`{=tex}{0,`\ldots`{=tex},K}}
P*{assign}(k`\mid `{=tex}p). \]

This produces an integer label map directly.

The output is mutually exclusive: every pixel belongs to at most one
instance slot.

### 8.3. Empty-slot filtering

Some slots may be unused or may receive negligible assignment
probability.

For each slot (k\>0), calculate:

\[ n_k=`\sum`{=tex}\_p`\mathbb{1}`{=tex}\[`\hat `{=tex}Y(p)=k\]. \]

Remove slots whose predicted area is below a predefined minimum area
(A\_{min}), if such filtering is needed.

This is a simple output-validity rule, not an instance grouping
algorithm. The value of (A\_{min}) must be selected using validation
data and reported.

Optionally, use a learned slot-existence score and discard slots below a
calibrated confidence threshold. If this is added, train the existence
head explicitly.

### 8.4. Final output

Return:

\[ `\hat `{=tex}Y`\in`{=tex}`\mathbb `{=tex}N\^{H`\times `{=tex}W}. \]

-   Background: 0.
-   Each retained predicted instance: one positive integer ID.
-   IDs should be compacted to (1,`\ldots`{=tex},`\hat `{=tex}N) for
    convenient evaluation and storage. ID compaction must not alter
    pixel membership.

No pairwise mask IoU merging is performed. If two slots predict the same
cell, that is a model error to be addressed through training and
evaluation, not silently repaired by proposal merging.

------------------------------------------------------------------------

## 9. Training data and preprocessing

### 9.1. Required data format

Each training sample consists of:

-   One microscopy image.
-   One integer instance label map with identical spatial dimensions.

The mask must preserve integer instance IDs. Do not convert an instance
label map into a binary mask before generating affinity and assignment
targets.

If source masks are stored as 16-bit PNG images, load them without lossy
conversion. Ensure that the data pipeline preserves the integer values
and does not interpolate masks using bilinear interpolation.

### 9.2. Image preprocessing

For grayscale brightfield images:

-   Convert image data to floating point.
-   Apply a consistent intensity normalization strategy.
-   Keep the normalization fixed across train, validation, and test.
-   Do not apply image intensity scaling to the instance label map.

If the input is originally 16-bit or has a nonstandard intensity range,
determine the scaling from the dataset rather than assuming an 8-bit
range.

### 9.3. Augmentation

Apply identical spatial transformations to the image and instance mask.

Suitable candidate augmentations include:

-   Horizontal and vertical flips.
-   Rotations by multiples of 90 degrees.
-   Small arbitrary rotations, if interpolation and label integrity are
    handled correctly.
-   Moderate scale changes.
-   Cropping and resizing.
-   Brightness and contrast variation.
-   Gamma variation.
-   Mild Gaussian noise.
-   Mild blur.

For masks, use nearest-neighbor interpolation for geometric
transformations. After transformations, regenerate affinity and
assignment targets from the transformed instance mask rather than
transforming precomputed affinity targets independently.

Avoid augmentations that create unrealistic microscopy appearances or
merge/split instances in ways that contradict the annotation semantics.

### 9.4. Crop sampling

If full images are too large for the available memory, train on patches.

However, patch sampling must consider object truncation:

-   Prefer crops containing complete instances.
-   If an instance is cut by a crop boundary, define a consistent policy
    for whether it is retained as a truncated object or ignored.
-   Do not assign the same original instance ID to disconnected
    fragments without considering whether the crop has removed the
    connecting region.
-   Record the fraction of truncated objects.

The maximum number of instances (K) applies to each training crop or
inference image, depending on the deployment configuration.

------------------------------------------------------------------------

## 10. Experimental plan

The experiments should isolate the value of each proposed component. Do
not change the U-Net architecture, optimizer, augmentation, and instance
grouping mechanism simultaneously.

### 10.1. E0 --- Foreground-only U-Net baseline

**Architecture:** Original U-Net with one foreground output channel.

**Purpose:**

-   Establish a semantic segmentation baseline.
-   Measure foreground Dice and IoU.
-   Establish training stability and inference speed.
-   Confirm that the data pipeline and masks are correct.

This experiment does not perform instance separation and should not be
compared with instance metrics as though it were a complete instance
segmentation model.

### 10.2. E1 --- Foreground + affinity heads

**Architecture:** Original U-Net with foreground and four affinity
output channels.

**Purpose:**

-   Determine whether the network can learn local same-instance
    relationships.
-   Evaluate affinity precision, recall, F1, and calibration.
-   Inspect predicted affinities at touching-cell boundaries.

At this stage, the model may not yet produce final instance assignments.
The purpose is to validate the affinity representation independently.

### 10.3. E2 --- Affinity-guided recurrent features

**Architecture:** E1 plus the recurrent message-passing module.

**Purpose:**

-   Determine whether affinity-gated message passing improves pixel
    representations.
-   Evaluate whether recurrent features become more consistent within
    individual cells.
-   Check for over-smoothing between adjacent cells.

Possible diagnostic measures:

-   Within-instance feature similarity.
-   Between-instance feature similarity for touching pairs.
-   Assignment performance once the assignment head is introduced.

### 10.4. E3 --- Full learned slot assignment

**Architecture:** E2 plus learned slots and the assignment head.

**Purpose:**

-   Produce complete instance segmentation outputs.
-   Evaluate whether the network can learn to assign all pixels of a
    cell to one slot.
-   Determine the effect of Hungarian matching and slot supervision.

This is the first complete model in the proposed architecture family.

### 10.5. E4 --- Number of recurrent iterations

Test:

\[ T`\in`{=tex}{1,2,3,5}. \]

Measure:

-   Instance F1.
-   AP at IoU thresholds.
-   PQ and AJI.
-   Inference latency.
-   Peak memory.
-   Training convergence.

The purpose is to find whether additional message-passing iterations
improve separation enough to justify their cost.

### 10.6. E5 --- Hidden feature dimension

Test:

\[ C_h`\in`{=tex}{16,32,64}. \]

Measure the same quality and efficiency metrics.

### 10.7. E6 --- Affinity message-passing variants

Compare:

1.  Four directional affinities with shared message weights.
2.  Four directional affinities with direction-specific message weights.
3.  Affinity-weighted normalized messages.
4.  Affinity-weighted messages without normalization.
5.  Optional inclusion of the original U-Net feature map in the
    recurrent input.

Keep all other components fixed.

### 10.8. E7 --- Affinity supervision variants

Compare:

-   Masked BCE.
-   Weighted masked BCE.
-   Focal loss.
-   Optional boundary-focused weighting.

Use validation metrics to determine whether improved affinity
classification actually improves final instance segmentation. Better
affinity F1 does not automatically imply better instance masks.

### 10.9. E8 --- Assignment and matching variants

Compare:

-   Per-pixel categorical assignment loss only.
-   Matched instance mask BCE + Dice only.
-   Combined assignment and mask losses.
-   Different matching-cost weight combinations.

Also inspect slot collapse, duplicate predictions, unused slots, and
missed objects.

### 10.10. E9 --- Slot capacity

Evaluate several values of (K) based on the observed instance-count
distribution.

Record:

-   Fraction of images with more than (K) ground-truth instances.
-   Fraction of slots used.
-   Duplicate slot predictions.
-   Missed instances.
-   Memory and latency.

A model should not be credited with good recall if the selected (K) is
too small to represent the dense images in the test set.

### 10.11. E10 --- Backbone replacement (later phase)

Only after E3 has been implemented and evaluated with original U-Net
should the backbone be replaced.

Potential later comparisons:

-   Original U-Net.
-   Lightweight depthwise-separable U-Net.
-   MobileNetV2-style encoder with a lightweight decoder.
-   Other compact encoder-decoder designs.

Keep the affinity head, recurrent grouping, slot assignment, losses, and
data protocol fixed as far as possible. This isolates the effect of the
feature extractor.

------------------------------------------------------------------------

## 11. Evaluation protocol

### 11.1. Instance segmentation metrics

Report multiple complementary metrics.

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  Object-level precision              Fraction of predicted instances
                                      that match a ground-truth instance

  Object-level recall                 Fraction of ground-truth instances
                                      successfully detected

  Object-level F1                     Balance between precision and
                                      recall

  AP at multiple IoU thresholds       Instance detection and mask quality

  AJI                                 Aggregated instance overlap quality

  PQ                                  Recognition and segmentation
                                      quality

  Foreground Dice                     Quality of cell-versus-background
                                      segmentation

  Boundary F1                         Quality of predicted instance
                                      boundaries
  -----------------------------------------------------------------------

Define the exact matching protocol for object-level precision, recall,
F1, AP, AJI, and PQ. In particular, specify the IoU threshold and
one-to-one matching policy used for object-level metrics.

Do not compare scores from different metric implementations without
checking their definitions.

### 11.2. Efficiency metrics

Report:

-   Number of trainable parameters.
-   FLOPs or MACs, with input size and counting convention.
-   Peak GPU memory during inference.
-   Inference latency per image.
-   Throughput in images per second.
-   Model checkpoint size.
-   CPU inference latency if CPU deployment is a target.

Measure latency after warm-up and synchronize GPU execution when timing
CUDA operations. Report hardware, software versions, image dimensions,
precision mode, batch size, and whether preprocessing/postprocessing is
included.

### 11.3. Error analysis

Inspect representative examples of:

-   Isolated cells.
-   Touching pairs.
-   Dense clusters.
-   Cells with low contrast.
-   Irregular or elongated cells.
-   Very small cells.
-   Large cells.
-   Cells near image borders.
-   Images with high instance counts.

Classify errors into:

1.  Merge: two or more ground-truth cells assigned to one predicted
    instance.
2.  Split: one ground-truth cell represented by multiple predicted
    instances.
3.  Miss: a ground-truth cell has no corresponding prediction.
4.  False positive: predicted instance has no valid ground-truth match.
5.  Boundary error: instance detected but its contour is inaccurate.
6.  Slot-capacity error: number of objects exceeds the available slots.

Report merge and split errors separately. They are particularly
important for evaluating the proposed instance separation mechanism.

------------------------------------------------------------------------

## 12. Implementation considerations

### 12.1. Tensor shapes

For a batch of size (B):

  Tensor                   Shape
  ------------------------ ------------------------------------
  Input image              `[B, 1, H, W]`
  U-Net features           `[B, C_f, H, W]`
  Foreground logits        `[B, 1, H, W]`
  Affinity logits          `[B, 4, H, W]`
  Pixel features           `[B, C_z, H, W]`
  Recurrent hidden state   `[B, C_h, H, W]`
  Slot embeddings          `[B, K, C_s]` or shared `[K, C_s]`
  Assignment logits        `[B, K+1, H, W]`
  Final labels             `[B, H, W]`

The exact internal tensor layout may be changed for implementation
efficiency, but the semantics and dimensions must remain documented.

### 12.2. Memory considerations

The full-resolution recurrent module can be expensive because it
operates at (H`\times `{=tex}W).

Potential optimizations:

-   Keep (C_h) small.
-   Use shared recurrent weights.
-   Use grouped or depthwise transformations where appropriate.
-   Avoid materializing large tensors containing all pixel-to-pixel pair
    combinations.
-   Compute messages using shifted feature maps and directional affinity
    maps.
-   Use mixed precision where numerically stable.
-   Consider processing large images in overlapping tiles if
    full-resolution memory is insufficient.

The slot assignment stage also has memory proportional to (KHW). This is
a major design constraint. Do not assume that a large (K) is inexpensive
merely because slot embeddings are small.

### 12.3. Numerical stability

-   Use logits-based BCE and cross-entropy functions.
-   Apply softmax in a numerically stable implementation.
-   Add epsilon to all normalization denominators.
-   Ensure invalid neighbors do not contribute messages.
-   Avoid NaNs when a slot receives almost no pixel mass.
-   Clamp or otherwise safely handle empty-slot normalization.
-   Monitor gradient norms for the affinity and recurrent modules.
-   Verify mixed-precision behaviour, particularly in slot normalization
    and matching costs.

### 12.4. Training stability

Recommended initial training strategy:

1.  Train the U-Net and foreground/affinity heads.
2.  Add the recurrent grouping module.
3.  Add the slot assignment mechanism.
4.  Fine-tune the full model jointly.

This staged procedure is an optimization strategy, not a requirement
that the final model contain separately trained components.

An alternative is joint training from initialization. Compare both only
after a stable baseline is available.

Use the same train/validation/test split, random seeds, augmentation
policy, and evaluation implementation across ablations.

### 12.5. Reproducibility

For every experiment, save:

-   Configuration.
-   Random seed.
-   Dataset split identifiers.
-   Model checkpoint.
-   Optimizer and scheduler state.
-   Training and validation loss history.
-   Per-metric validation history.
-   Inference configuration.
-   Software and hardware information.

Store the selected (K), recurrent iteration count, hidden dimensions,
loss weights, and any output filtering thresholds in the experiment
configuration rather than hardcoding them.

------------------------------------------------------------------------

## 13. Risks and limitations

The proposed architecture is a research hypothesis, not a guarantee of
improved performance.

### 13.1. Fixed number of slots

The maximum number of instances is limited by (K). This can be
problematic for dense microscopy fields.

Mitigation:

-   Measure instance-count distributions before selecting (K).
-   Evaluate slot capacity explicitly.
-   Consider tiled inference or a variable-count affinity-graph variant
    if necessary.

### 13.2. Slot collapse

Several slots may learn to represent the same object, or the model may
use only a small subset of slots.

Mitigation:

-   Use one-to-one Hungarian matching.
-   Supervise unmatched slots as no-object.
-   Inspect slot usage and duplicate masks.
-   Test optional slot regularization only if the basic matching
    objective is insufficient.

### 13.3. Affinity errors

Incorrectly high affinity across a boundary can cause neighboring cells
to exchange information. Incorrectly low affinity inside a cell can
fragment its representation.

Mitigation:

-   Use valid-pair masked affinity supervision.
-   Inspect affinity maps directly.
-   Evaluate errors specifically at touching boundaries.
-   Test recurrent iteration count and affinity calibration.

### 13.4. Over-smoothing

Repeated message passing may make adjacent instances less
distinguishable.

Mitigation:

-   Use a small number of iterations.
-   Share recurrent weights.
-   Use residual updates.
-   Compare hidden feature similarity within and between instances.

### 13.5. Computational cost

The recurrent module and dense slot assignment may be expensive at high
image resolutions.

Mitigation:

-   Keep hidden dimensions small.
-   Profile each component separately.
-   Avoid unnecessary full-resolution intermediate tensors.
-   Investigate lower-resolution grouping with high-resolution mask
    refinement only if the first implementation is too expensive.

### 13.6. Boundary ambiguity in brightfield microscopy

Brightfield cell boundaries may be weak, overlapping, or visually
ambiguous. A local affinity target is only as reliable as the
ground-truth annotation.

Mitigation:

-   Review annotation consistency.
-   Inspect difficult boundary examples.
-   Consider reduced loss weights for ambiguous annotation regions only
    if a defensible annotation policy exists.
-   Report performance separately for crowded and isolated cells.

------------------------------------------------------------------------

## 14. Definition of the first implementation milestone

The first complete implementation should be considered successful
technically when it can:

1.  Load a grayscale image and its integer instance label map.
2.  Generate foreground and four-direction affinity targets.
3.  Run the original U-Net and all prediction heads.
4.  Execute affinity-guided recurrent message passing.
5.  Refine a fixed set of instance slots.
6.  Predict per-pixel background/instance assignments.
7.  Train using Hungarian matching and the specified losses.
8.  Produce an integer instance label map without clustering or
    watershed.
9.  Evaluate instance-level metrics.
10. Save reproducible checkpoints and configurations.

Technical completion does not imply that the model outperforms
InstanSeg. Performance must be established experimentally.

------------------------------------------------------------------------

## 15. Recommended initial configuration

The following values are starting points for the first prototype, not
optimized hyperparameters.

  Parameter                                                         Initial value
  ------------------------------------------ ------------------------------------
  Backbone                                                         Original U-Net
  Input channels                                                                1
  Foreground output channels                                                    1
  Affinity directions                                                           4
  Pixel feature dimension (C_z)                                                16
  Recurrent hidden dimension (C_h)                                             32
  Recurrent message-passing iterations (T)                                      3
  Number of slot-update iterations                                              2
  Maximum instance slots (K)                   Determined from dataset statistics
  Assignment classes                                                        (K+1)
  Foreground loss                                                      BCE + Dice
  Affinity loss                                            Masked BCE with logits
  Assignment loss                                                   Cross-entropy
  Instance mask loss                                                   BCE + Dice
  Matching                                               Hungarian, training only
  Inference assignment                           Argmax over background and slots
  Clustering                                                                 None
  Watershed                                                                  None
  Local maximum detection                                                    None
  Per-instance crop classification                                           None
  Proposal IoU merging                                                       None

------------------------------------------------------------------------

## 16. Summary of the proposed contribution

The proposed architecture is a U-Net-based, affinity-guided recurrent
instance segmentation network.

Its central idea is to represent cell instances through learned
relationships between neighboring pixels. A recurrent neural module uses
these affinities to propagate and refine pixel features. A set of
learned instance slots then gathers information from the refined
features, and a learned assignment head assigns each pixel to background
or one of the available instances.

The architecture differs from the supplied InstanSeg procedure in
several fundamental ways:

  -----------------------------------------------------------------------
  InstanSeg mechanism                 Proposed mechanism
  ----------------------------------- -----------------------------------
  Spatial embedding coordinates       Local same-instance affinity maps

  Seed map and local maxima           Learned instance slots

  Per-instance crop extraction        Full-image recurrent feature
                                      interaction

  Centroid-relative pixel MLP         Affinity-guided recurrent grouping

  Sparse instance proposals           Dense pixel-to-slot assignment

  Greedy proposal competition         Learned categorical assignment

  Pairwise proposal IoU merging       No proposal merging

  Rule-based reconstruction stages    Learned assignment, followed by
                                      integer label decoding
  -----------------------------------------------------------------------

The initial backbone is intentionally a standard U-Net. This makes it
possible to evaluate the instance representation and grouping mechanism
independently of future encoder-decoder improvements.

The main hypothesis to test is:

> A compact network trained to predict local pixel affinities and to
> refine pixel representations through recurrent affinity-guided message
> passing can learn reliable cell-instance assignments directly,
> providing a viable alternative to centroid-based embedding and
> proposal reconstruction pipelines.

The hypothesis should be accepted, refined, or rejected based on
controlled experiments measuring instance segmentation quality,
merge/split errors, computational cost, and robustness across microscopy
images.

------------------------------------------------------------------------

## 17. Open design decisions to resolve during implementation

These points are intentionally left configurable rather than treated as
settled facts:

1.  **Slot initialization:** shared learned slot embeddings versus
    image-conditioned slot initialization.
2.  **Slot update:** GRU-based updates versus a simpler recurrent
    attention update.
3.  **Slot count:** the appropriate (K) based on actual dataset density
    and memory limits.
4.  **Affinity directions:** four directions versus eight directions.
5.  **Message normalization:** normalized versus unnormalized
    affinity-weighted aggregation.
6.  **Recurrent update:** ConvGRU versus a lighter gated convolutional
    update.
7.  **Assignment objective:** categorical assignment alone versus
    categorical assignment plus matched mask losses.
8.  **Background handling:** explicit background class versus
    foreground-gated assignment.
9.  **Training schedule:** staged training versus joint training from
    initialization.
10. **Variable-count alternative:** learned affinity refinement followed
    by connected-component labelling, to be evaluated separately if
    fixed slot capacity proves impractical.

These decisions should be resolved through implementation tests and
ablations. They should not all be changed simultaneously.

**End of specification.**
