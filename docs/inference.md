# Inference

Inference is **backend-agnostic**: callers depend only on the `Predictor` interface
(`src.inference.predictor`), so a PyTorch model and a future FINN-deployed
accelerator are interchangeable. Works standalone, without FINN.

## Predictor

```python
from src.inference import load_classifier
predictor = load_classifier("configs/models/resnet18.yaml", "best.pth",
                            class_names=["cat", "dog", ...], device="cpu")

from PIL import Image
pred = predictor.predict(Image.open("img.jpg").convert("RGB"))
print(pred.label, pred.score)
```

`load_classifier` builds the net from its yaml (`build_network`), loads the
checkpoint, and returns a `TorchClassifier`. A `Prediction` is `(label, score, raw)`.

## GUI

A Tkinter GUI (`src.gui.InferenceGUI`) runs a predictor on:

- a single image,
- a dataset folder (iterate sample by sample),
- the live webcam (needs `opencv-python`),

overlays the prediction, and saves screenshots (with the overlay) to `screenshots/`.

From the CLI:

```sh
python run.py infer --exp configs/models/resnet18.yaml --weights best.pth --source webcam --gui
python run.py infer --exp configs/models/resnet18.yaml --weights best.pth --source image.jpg
```

GUI deps: `tkinter` (stdlib), `Pillow`; webcam also needs `opencv-python`.

## FINN backend (future)

`src/finn/build_finn.py` compiles an exported QONNX model to an FPGA dataflow
accelerator. A `FinnPredictor` implementing the same `Predictor.predict()` contract
would let the GUI run on hardware without any other change — see [qat.md](qat.md).
