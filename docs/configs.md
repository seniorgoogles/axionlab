# Configs

Two kinds of yaml: a **model (net) yaml** (the architecture) and an **experiment
yaml** (model + dataset + training). They are separate so one net can be reused
across experiments/datasets. Experiment yamls may inherit via a `base:` key
(deep-merged).

## Experiment yaml

Consumed by `Runner.from_experiment()`.

```yaml
name: "jsc_2l_qat"
model: "configs/models/jsc_2l.yaml"   # path to a net yaml
dataset: "jsc"                        # registered dataset name (see datasets.md)
# dataset wiring (forwarded to the dataset factory):
train_path: null
test_path: null
num_workers: 4
distributed: false
batch_size: [1024, 1024]              # [train, test]
# training:
epochs: 30
learning_rate: 0.001
output_dir: "./outputs/jsc_2l_qat"
best_metric_name: "accuracy"
best_metric_mode: "max"               # max | min
validate_every: 1
device: null                          # null = auto-detect
tracking_db: "./outputs/jsc_2l_qat/experiments.db"   # optional
```

CLI flags override these keys at run time.

## Model yaml — layer-list format

Same idea as Ultralytics: a list of `[from, repeats, module, name, args]` leaves,
grouped under `backbone` (and optionally `neck` / `head`). `module` is a string
resolved by a registry, so the **same yaml builds float or quantized** layers by
swapping the module name / adding `bit_width`.

`build_network(yaml)` picks the builder automatically: a config with `head:` or
`scales:` uses the **graph builder** (YOLO-style, `from`-routing); otherwise the
**sequential builder**.

### Sequential (ResNet / MLP)

Names match torchvision, so a pretrained float state_dict loads by name
(`load_float_state_dict`). Supports nested blocks (`BasicBlock`, `downsample`).

```yaml
name: "jsc_2l"
nc: 5
backbone:
  - [-1, 1, Linear, "fc1",   {in_features: 16, out_features: 64}]
  - [-1, 1, ReLU,   "relu1", {inplace: True}]
  - [-1, 1, Linear, "fc3",   {in_features: 64, out_features: 5}]
```

Quantized variant — just change the module + add `bit_width`:

```yaml
  - [-1, 1, QuantLinear, "fc1", {in_features: 16, out_features: 64, bias: False, bit_width: 4}]
  - [-1, 1, QuantReLU,   "relu1", {bit_width: 4}]
```

`bit_width` maps to Brevitas' `weight_bit_width`; a `quantizer: <Name>` arg is only
needed for a custom quantizer not built into Brevitas (e.g. `LearnedBitWidthQuantizer`).

### Graph (YOLO-style)

`from` may be `-1` or a list (multi-input, e.g. `Concat`/`Detect`); channels are
inferred; `scales` does depth/width scaling. Per-block quant in two ways:

```yaml
scales: {depth: 0.33, width: 0.25, max_channels: 1024}
backbone:
  - [-1, 1, Conv, {c2: 64, k: 3, s: 2}]
  - [-1, 3, C2f,  {c2: 128, shortcut: True}]
  # swap surgery (quantize a float block):
  - [-1, 3, C2f,  {c2: 256, quant: {weight_bit_width: 4, act_bit_width: 4}}]
  # OR a dedicated quant block (shared-scale at cat/add, FINN-friendly):
  - [-1, 3, QuantC2f, {c2: 256, weight_bit_width: 4, act_bit_width: 4}]
head:
  - [[-1, 6], 1, Concat, {dim: 1}]
  - [[15, 18, 21], 1, Detect, {}]
```

See `configs/models/` for full ResNet-18, YOLOv8n (float + quant) and JSC examples.
