"""Draw-a-digit inference: canvas PNG → MNIST-distribution tensor → prediction.

A canvas drawing is not MNIST-distributed (DESIGN.md §9): MNIST digits are
white-on-black, size-normalized to a 20×20 box, and centered by pixel mass in
a 28×28 field. Skip any of those steps and a 97%-accurate model feels broken
on hand-drawn input. The preprocessed image is returned so the UI can show
exactly what the model sees.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image, ImageOps

from app.training.checkpoint import MNIST_NORMALIZATION, load_checkpoint
from app.training.models import SimpleMLP


class EmptyCanvasError(ValueError):
    pass


def preprocess_canvas(image_b64: str) -> tuple[torch.Tensor, list[list[int]]]:
    raw = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw)).convert("L")

    # White digit on black background, like MNIST. Canvas drawings arriving
    # dark-on-light get inverted (decided by overall brightness).
    if np.asarray(img).mean() > 127:
        img = ImageOps.invert(img)

    # Drop antialiasing noise, then crop to the ink's bounding box.
    img = img.point(lambda p: 0 if p < 20 else p)
    bbox = img.getbbox()
    if bbox is None:
        raise EmptyCanvasError("canvas is empty")
    img = img.crop(bbox)

    # Scale the longest side to 20px (MNIST's digit box), keeping aspect.
    w, h = img.size
    scale = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.LANCZOS)

    # Paste into 28×28 so the center of mass lands at the center, as in the
    # original MNIST preparation.
    arr = np.asarray(img, dtype=np.float32)
    total = arr.sum()
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total
    top = int(round(14 - cy))
    left = int(round(14 - cx))
    top = max(0, min(28 - arr.shape[0], top))
    left = max(0, min(28 - arr.shape[1], left))
    field = np.zeros((28, 28), dtype=np.float32)
    field[top:top + arr.shape[0], left:left + arr.shape[1]] = arr

    mean, std = MNIST_NORMALIZATION["mean"][0], MNIST_NORMALIZATION["std"][0]
    tensor = (torch.from_numpy(field / 255.0) - mean) / std
    preview = field.astype(np.uint8).tolist()
    return tensor.reshape(1, 1, 28, 28), preview


def predict(checkpoint_path: str, image_b64: str) -> dict:
    ckpt = load_checkpoint(checkpoint_path)
    model = SimpleMLP(hidden=tuple(ckpt["config"].get("hidden", [128, 64])))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tensor, preview = preprocess_canvas(image_b64)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)
    return {
        "prediction": int(probs.argmax().item()),
        "probs": [round(p, 5) for p in probs.tolist()],
        "preprocessed": preview,  # the 28×28 the model actually saw
        "class_mapping": ckpt.get("class_mapping", list(range(10))),
    }
