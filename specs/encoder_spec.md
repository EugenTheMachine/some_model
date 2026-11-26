# `instanseg_unet` Encoder Specification

## 1. Scope

This specification covers only the encoder used by the `instanseg_unet` model type in [InstanSeg_UNet.py](../instanseg/utils/models/InstanSeg_UNet.py):

- the `EncoderBlock` module;
- how the `InstanSeg_UNet` constructor creates and orders its encoder stages;
- how the encoder returns the final feature map and intermediate skip features.

Decoder details and task-specific output semantics are outside scope.

The code snippets below are adapted from [InstanSeg_UNet.py](../instanseg/utils/models/InstanSeg_UNet.py). The model's current base default is `act="ReLu"` and `EncoderBlock` defaults to `"ReLU"`, which select ReLU through the case-insensitive activation helper. As requested, the reconstruction snippets change those defaults to `"GELU"`. The helper already supports GELU by constructing `nn.GELU()`.

## 2. Encoder overview

The baseline encoder is a sequence of `EncoderBlock`s. The first block preserves the input resolution; each later block applies 2×2 max pooling before its convolutions. Each block maps its input to the requested output width and uses two residual additions.

For the default four-stage encoder, the effective channel widths are:

```text
input channels -> 32 -> 64 -> 128 -> 256
```

For an input tensor of shape `(N, C, H, W)`, with default `C=3` and H/W divisible by 8:

| Stage | Pool before block | Input channels | Output channels | Output spatial size | Used as skip |
|---|---:|---:|---:|---:|---|
| 0 | No | `C` (normally 3) | 32 | `H × W` | Yes |
| 1 | 2×2, stride 2 | 32 | 64 | `H/2 × W/2` | Yes |
| 2 | 2×2, stride 2 | 64 | 128 | `H/4 × W/4` | Yes |
| 3 | 2×2, stride 2 | 128 | 256 | `H/8 × W/8` | No; final bottleneck |

Thus, for `(N, 3, 512, 512)`, the encoder returns the final feature tensor `(N, 256, 64, 64)` and a shallow-to-deep skip list with shapes `(N, 32, 512, 512)`, `(N, 64, 256, 256)`, and `(N, 128, 128, 128)`.

The required spatial divisibility is derived from the number of encoder blocks:

```python
divisor = 2 ** (len(self.encoder) - 1)
```

With four blocks, `H` and `W` must be divisible by 8. This requirement allows each max-pooling stage and corresponding decoder upsampling to align spatially.

## 3. Encoder block

### 3.1 Components

Each non-shallow block contains five convolution-normalization-activation modules:

- `conv0`: 1×1 projection from `in_channels` to `out_channels`, used by the first residual path;
- `conv1`: 3×3 convolution from `in_channels` to `out_channels`;
- `conv2`: 3×3 convolution from `out_channels` to `out_channels`;
- `conv3`: 3×3 convolution from `out_channels` to `out_channels`;
- `conv4`: 3×3 convolution from `out_channels` to `out_channels`.

All convolutions preserve spatial size: `conv_norm_act()` uses padding `kernel_size // 2`, and its `nn.Conv2d` uses the default stride of 1. If pooling is enabled, the 2×2 max pool reduces the spatial dimensions by half before all five convolution modules.

The forward equations, with `z` denoting the optionally pooled input, are:

```text
projection = conv0(z)
main       = conv1(z)
first      = projection + conv2(main)
output     = first + conv4(conv3(first))
```

Both additions are residual connections. The 1×1 projection ensures that the first addition has matching channel dimensions.

The optional `shallow=True` variant replaces `conv2` and `conv3` with identities. The default `InstanSeg_UNet` encoder does not set `shallow`, so its stages use the full block.

### 3.2 Convolution, normalization, activation helper

The source helper supports batch, instance, and group/layer normalization, and several activation names. For the GELU-based reconstruction, use its normalization logic and the GELU activation path below; other activation choices are intentionally omitted from this excerpt:

```python
def conv_norm_act(
    in_channels: int,
    out_channels: int,
    sz: int,
    norm: str = "BATCH",
    act: str = "GELU",
    depthwise: bool = False,
) -> nn.Sequential:
    if norm is None or norm == "None":
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(
            out_channels, eps=1e-5, track_running_stats=False, affine=True
        )
    elif norm.lower() in ("layer", "group"):
        norm_layer = nn.GroupNorm(1, out_channels, eps=1e-5)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    if act is None or act == "None":
        act_layer = nn.Identity()
    elif act.lower() == "gelu":
        act_layer = nn.GELU()
    else:
        raise ValueError(f"Unsupported act: {act}")

    groups = in_channels if depthwise else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            sz,
            padding=sz // 2,
            groups=groups,
            bias=(norm is None or norm == "None"),
        ),
        norm_layer,
        act_layer,
    )
```

For the default encoder configuration, each convolution is followed by `BatchNorm2d(eps=1e-5, momentum=0.05)` and `GELU`. The encoder does not enable depthwise convolution.
The model's `dropout` argument is stored on the parent model but no dropout layer is applied inside these encoder blocks.

### 3.3 `EncoderBlock` implementation

This is the source block structure with the default activation changed from ReLU to GELU:

```python
class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True,
        norm="BATCH",
        act="GELU",
        shallow=False,
    ):
        super().__init__()

        if pool:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = nn.Identity()

        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x
```

## 4. Encoder construction in `InstanSeg_UNet`

The source model's direct constructor defaults are `layers=[256, 128, 64, 32]`, `norm="BATCH"`, and `act="ReLu"`. The GELU-adjusted reconstruction should use `act="GELU"`. The class reverses the `layers` sequence before constructing its encoder. Consequently, the direct constructor defaults produce encoder widths `[32, 64, 128, 256]`.

The high-level model factory uses a configured layer sequence defaulting to `(32, 64, 128, 256)`, reverses it before calling the model constructor, and the constructor reverses it again. The effective encoder order is therefore also `[32, 64, 128, 256]`.

Relevant source excerpt, with the activation default changed. Factory-compatibility handling, bookkeeping assignments, and subsequent decoder initialization are omitted here; none of them changes the encoder stages:

```python
class InstanSeg_UNet(UNet):
    def __init__(
        self,
        in_channels=3,
        out_channels=[[2, 2, 1]],
        layers=[256, 128, 64, 32],
        norm="BATCH",
        dropout=0,
        act="GELU",
        model_type="instanseg_unet",
        **kwargs,
    ):
        super().__init__()

        if in_channels == 0 or in_channels is None:
            in_channels = 3
        self.in_channels = in_channels

        layers = list(layers)[::-1]
        self.encoder = nn.ModuleList(
            [EncoderBlock(
                in_channels, layers[0], pool=False, norm=norm, act=act
            )]
            + [
                EncoderBlock(
                    layers[i], layers[i + 1], norm=norm, act=act
                )
                for i in range(len(layers) - 1)
            ]
        )
        layers = layers[::-1]
```

`layers = layers[::-1]` after encoder construction restores the channel ordering needed by the decoder constructor; it does not change the already-created encoder blocks.

The forward path validates the input, collects all encoder outputs except the final bottleneck as skips, and returns both values to the decoder stage:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    divisor = 2 ** (len(self.encoder) - 1)
    self._validate_input_tensor(x, divisor=divisor)
    skips = []
    for n, layer in enumerate(self.encoder):
        x = layer(x)
        if n < len(self.encoder) - 1:
            skips.append(x)

    return torch.cat(
        [decoder(x, skips) for decoder in self.decoders],
        dim=1,
    )
```

The encoder's independent forward contract is therefore:

```python
bottleneck, skips = run_encoder(input_tensor)
```

where `bottleneck` is the output of the final stage and `skips` contains outputs from earlier stages in shallow-to-deep order.

## 5. GELU change and compatibility

To replace ReLU in this encoder:

1. Change `EncoderBlock`'s default `act` from `"ReLU"` to `"GELU"`.
2. Change `InstanSeg_UNet.__init__`'s default `act` from `"ReLu"` to `"GELU"`.
3. Keep passing `act` from the model constructor into every `EncoderBlock`.
4. Keep the existing GELU branch in `conv_norm_act()`, which constructs `nn.GELU()`.

The activation change does not alter parameter tensor shapes, so state dictionaries remain structurally loadable. It does change forward activations and numerical outputs; checkpoints trained with ReLU will not be behaviorally equivalent when evaluated with GELU.

## 6. Reconstruction checklist

When implementing this encoder from scratch:

1. Preserve the `pool -> projection/main convolutions -> two residual additions` order.
2. Use a first stage with identity pooling, then one max-pool in every subsequent stage.
3. For the default four stages, use widths `32, 64, 128, 256`.
4. Use 1×1 projection and 3×3 main/residual convolutions with same-size padding.
5. Apply batch normalization and GELU after each convolution in the default configuration.
6. Collect all stage outputs except the final bottleneck as skips in shallow-to-deep order.
7. Require spatial dimensions divisible by `2 ** (number_of_stages - 1)`.
8. Test the default output and skip shapes for a representative input, as well as invalid input rank, channel count, dtype, and spatial divisibility.

