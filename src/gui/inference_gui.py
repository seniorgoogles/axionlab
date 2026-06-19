"""Tkinter inference GUI: run a predictor on images, a dataset folder, or webcam,
overlay the prediction, and save screenshots.

Backend-agnostic: takes any object implementing the Predictor interface
(src.inference.predictor), so the PyTorch and (later) FINN backends both work.
Works fully standalone without FINN.

    python -m src.gui.inference_gui --model configs/models/resnet18.yaml \
        --ckpt outputs/resnet18_imagenet/best.pth --classes imagenet_classes.txt

GUI deps: tkinter (stdlib), Pillow. Webcam additionally needs opencv-python.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class InferenceGUI:
    def __init__(self, predictor, title: str = "axionlab inference"):
        import tkinter as tk

        self.predictor = predictor
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.canvas.pack()
        self.status = tk.Label(self.root, text="load an image, a folder, or start the webcam")
        self.status.pack(fill="x")

        bar = tk.Frame(self.root)
        bar.pack(fill="x")
        tk.Button(bar, text="Open image", command=self.open_image).pack(side="left")
        tk.Button(bar, text="Open folder", command=self.open_folder).pack(side="left")
        tk.Button(bar, text="Next →", command=self.next_in_folder).pack(side="left")
        tk.Button(bar, text="Webcam", command=self.toggle_webcam).pack(side="left")
        tk.Button(bar, text="Screenshot", command=self.save_screenshot).pack(side="left")

        self._folder: list = []
        self._folder_idx = 0
        self._webcam = None
        self._current_image = None     # the displayed PIL image (with overlay)
        self._tk_image = None          # keep a ref so Tk doesn't GC it

    # ---- sources ---------------------------------------------------------
    def open_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self._stop_webcam()
            self._show_path(path)

    def open_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory()
        if not d:
            return
        self._stop_webcam()
        self._folder = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in _IMG_EXT)
        self._folder_idx = 0
        if self._folder:
            self._show_path(str(self._folder[0]))
        else:
            self.status.config(text="no images in folder")

    def next_in_folder(self):
        if not self._folder:
            return
        self._folder_idx = (self._folder_idx + 1) % len(self._folder)
        self._show_path(str(self._folder[self._folder_idx]))

    def toggle_webcam(self):
        if self._webcam is not None:
            self._stop_webcam()
            return
        import cv2
        self._webcam = cv2.VideoCapture(0)
        self._cv2 = cv2
        self._webcam_loop()

    def _stop_webcam(self):
        if self._webcam is not None:
            self._webcam.release()
            self._webcam = None

    def _webcam_loop(self):
        if self._webcam is None:
            return
        from PIL import Image
        ok, frame = self._webcam.read()
        if ok:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            self._show_image(Image.fromarray(frame))
        self.root.after(30, self._webcam_loop)

    # ---- rendering -------------------------------------------------------
    def _show_path(self, path: str):
        from PIL import Image
        self._show_image(Image.open(path).convert("RGB"))

    def _show_image(self, pil_image):
        from PIL import ImageDraw, ImageTk

        pred = self.predictor.predict(pil_image)
        disp = pil_image.copy()
        disp.thumbnail((640, 480))
        draw = ImageDraw.Draw(disp)
        text = f"{pred.label}  {pred.score:.2f}"
        draw.rectangle([0, 0, disp.width, 22], fill=(0, 0, 0))
        draw.text((6, 4), text, fill=(0, 255, 0))

        self._current_image = disp
        self._tk_image = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self._tk_image)
        self.status.config(text=text)

    def save_screenshot(self, out_dir: str = "screenshots"):
        if self._current_image is None:
            self.status.config(text="nothing to save yet")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"
        self._current_image.save(path)
        self.status.config(text=f"saved {path}")

    def run(self):
        self.root.mainloop()


def main():
    import argparse

    from src.inference import load_classifier

    p = argparse.ArgumentParser(description="axionlab inference GUI")
    p.add_argument("--model", required=True, help="model net yaml")
    p.add_argument("--ckpt", default="", help="checkpoint .pth (best.pth)")
    p.add_argument("--classes", default="", help="optional text file: one class name per line")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    class_names = None
    if args.classes and Path(args.classes).exists():
        class_names = Path(args.classes).read_text().splitlines()

    predictor = load_classifier(args.model, args.ckpt, class_names=class_names, device=args.device)
    InferenceGUI(predictor).run()


if __name__ == "__main__":
    main()
