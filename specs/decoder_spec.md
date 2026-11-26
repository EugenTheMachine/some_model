# `instanseg_unet` Decoder Specification

## 1. Scope

This specification describes the baseline decoder used by `model_type="instanseg_unet"` in [InstanSeg_UNet.py](../instanseg/utils/models/InstanSeg_UNet.py). It documents the current source behavior without changing activation, upsampling, normalization, or block structure.

The baseline `InstanSeg_UNet` class uses `DecoderBlock`. Other model variants in the same source file may select different decoder blocks; those variant-specific implementations are not part of this baseline description.

## 2. Decoder inputs and stage layout

The decoder receives:

- the final encoder feature map (the bottleneck);
- the encoder's intermediate skip tensors in shallow-to-deep order.

The `Decoder.forward()` method reverses the skip list before pairing it with decoder blocks. Thus, the deepest skip is consumed first.

For the default four encoder stages, the decoder channel sequence is `[256, 128, 64, 32]`. This is the reversed version of the encoder widths `[32, 64, 128, 256]`. The three decoder blocks are constructed as follows:

| Decoder block | Input feature channels | Skip channels | Output channels | Spatial scale after upsampling |
|---|---:|---:|---:|---:|
| 0 | 256 | 128 | 128 | `H/4 × W/4` |
| 1 | 128 | 64 | 64 | `H/2 × W/2` |
| 2 | 64 | 32 | 32 | `H × W` |

For an input `(N, 3, 512, 512)`, the bottleneck is `(N, 256, 64, 64)`. The decoder's final feature map is `(N, 32, 512, 512)`.

The source model constructor normalizes `out_channels` into a list of decoder specifications:

- an integer `n` becomes `[[n]]`;
- a flat list such as `[2, 2, 1]` becomes `[[2, 2, 1]]`;
- an already nested list is used as the list of decoder specifications.

Each decoder specification gets a separate `Decoder` instance. Every `Decoder` returns the concatenation of its final projection tensors, and the model concatenates the outputs from all decoder instances along channel dimension 1. For example, `out_channels=[[2, 2, 1]]` creates one decoder with final projection channel widths 2, 2, and 1, for an output with 5 channels.

## 3. `DecoderBlock`

### 3.1 Components and behavior

Each baseline decoder block has:

- `conv0`: 1×1 projection of the upsampled input to `out_channels`;
- `conv_skip`: optional 1×1 projection of the skip tensor to `out_channels`;
- `conv1`: 3×3 convolution from `in_channels` to `out_channels`;
- `conv2`, `conv3`, and `conv4`: 3×3 convolutions from `out_channels` to `out_channels`.

The block first upsamples the input by a factor of 2. The default mode is nearest-neighbor interpolation. The optional exact string `"bilinear"` selects bilinear interpolation with `align_corners=False`; any other value follows the nearest-neighbor branch.

When a skip and its projection exist, the block combines the projected skip with the `conv1` result before `conv2`, and adds the `conv0` projection as a residual. Without a skip, it applies `conv2` to the `conv1` result and adds the same projection residual. Both branches then use a second residual path through `conv3` and `conv4`.

The optional `shallow=True` replaces `conv3` with `nn.Identity()`. The baseline constructor uses the default `shallow=False`.

### 3.2 Source implementation

The following is the `DecoderBlock` implementation from the codebase, unchanged:

```python
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm="BATCH",
        act="ReLU",
        shallow=False,
        upsample_mode="nearest",
    ):
        super().__init__()
        self.upsample_mode = upsample_mode

        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        if skip_channels and skip_channels > 0:
            self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        else:
            self.conv_skip = None
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        if self.upsample_mode == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        if skip is not None and self.conv_skip is not None:
            x = proj + self.conv2(x + self.conv_skip(skip))
        else:
            x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x
```

### 3.3 Shared convolution helper defaults

`DecoderBlock` passes its `norm` and `act` values to the shared `conv_norm_act()` helper. The baseline `InstanSeg_UNet` constructor defaults to `norm="BATCH"` and `act="ReLu"`; the helper compares activation names case-insensitively, so the baseline decoder uses ReLU. The default `DecoderBlock` arguments are `norm="BATCH"` and `act="ReLU"` as shown above.

The shared helper's convolution uses same-size padding (`padding=kernel_size // 2`) and the default stride of 1. With the baseline batch normalization setting, each convolution module is convolution, `BatchNorm2d`, then activation. The final output projections instead set `act=None`.

## 4. `Decoder` sequence and final projections

The `Decoder` creates one `DecoderBlock` for each adjacent pair of channel widths. Each block maps `layers[i]` to `layers[i + 1]`, using `layers[i + 1]` for both skip and output width. Its final 1×1 projections map the final feature width to each requested output width. The projection outputs are concatenated along channel dimension.

The following source implementation is unchanged:

```python
class Decoder(nn.Module):
    def __init__(self, layers, out_channels, norm, act, decoder_block=DecoderBlock, upsample_mode="nearest"):
        super().__init__()

        self.decoder = nn.ModuleList(
            [decoder_block(layers[i], layers[i + 1], layers[i + 1], norm=norm, act=act, upsample_mode=upsample_mode)
             if decoder_block is DecoderBlock
             else decoder_block(layers[i], layers[i + 1], layers[i + 1], norm=norm, act=act)
             for i in range(len(layers) - 1)]
        )

        self.final_block = nn.ModuleList(
            [
                conv_norm_act(
                    layers[-1],
                    out_channel,
                    1,
                    norm=norm if (norm is not None) and norm.lower() != "instance" else None,
                    act=None,
                )
                for out_channel in out_channels
            ]
        )

    def forward(self, x, skips):
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)

        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)
        return x
```

Important details from this implementation:

- `upsample_mode` is passed to a block only when `decoder_block is DecoderBlock`; custom block classes use the alternate constructor call shown above.
- Final projections use normalization unless `norm` is `None` or equals `"instance"` ignoring case.
- Final projections have no activation because they pass `act=None`.
- A final projection's spatial size matches the last decoder block's feature size.

## 5. Wiring in `InstanSeg_UNet`

The base class declares `decoder_block = DecoderBlock`. Its constructor reverses the layer list for the encoder, then reverses it back before creating each decoder. The following excerpt is unchanged from the relevant decoder wiring:

```python
layers = list(layers)[::-1]
self.encoder = nn.ModuleList(
    [EncoderBlock(in_channels, layers[0], pool=False, norm=norm, act=act)]
    + [EncoderBlock(layers[i], layers[i + 1], norm=norm, act=act) for i in range(len(layers) - 1)]
)
layers = layers[::-1]

if type(out_channels) == int:
    out_channels = [[out_channels]]
if type(out_channels[0]) == int:
    out_channels = [out_channels]

decoder_block = getattr(self, "decoder_block", DecoderBlock)
self.decoders = nn.ModuleList([
    Decoder(layers, out_channel, norm, act, decoder_block=decoder_block)
    for out_channel in out_channels
])
```

The baseline model forward method runs all encoder blocks, stores all but the final encoder output as skips, then applies each decoder to the same bottleneck and skip list. The concatenated result is the model output:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    divisor = 2 ** (len(self.encoder) - 1)
    self._validate_input_tensor(x, divisor=divisor)
    skips = []
    for n, layer in enumerate(self.encoder):
        x = layer(x)
        if n < len(self.encoder) - 1:
            skips.append(x)

    return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)
```

## 6. Reconstruction notes

To reproduce the baseline decoder exactly:

1. Use `DecoderBlock` for the `instanseg_unet` base model.
2. Keep the current ReLU behavior: the model's `"ReLu"` value resolves to ReLU in the shared helper; do not substitute GELU.
3. Default to nearest-neighbor 2× upsampling.
4. Project both the upsampled feature and each available skip to the block's output width with 1×1 convolutions.
5. Preserve both residual additions and the optional `shallow` behavior.
6. Reverse the skip list before iterating through decoder stages.
7. Build one decoder per normalized `out_channels` specification and concatenate decoder outputs and final projection outputs along channel dimension 1.
8. For four encoder widths `[32, 64, 128, 256]`, the decoder's internal `layers` sequence is `[256, 128, 64, 32]`; the three blocks upsample from H/8 to H/4, H/2, then H.
9. Keep decoder variants such as `ImprovedDecoderBlock` and `V4DecoderBlock` distinct; they are not the baseline `instanseg_unet` decoder.

