"""
Grad-CAM / CAM visual explanations.

Two implementations of the SAME heatmap concept:

* TFGradCAM  - true gradient-weighted class-activation for the TensorFlow
               backend (online path).  Requires access to the last conv layer
               and model gradients.
* ONNXCAM    - weights-based CAM for the ONNX backend (offline/Pyodide path).
               MobileNetV2's head is GlobalAveragePooling -> Dense, and for a
               GAP network CAM == Grad-CAM up to a constant that vanishes on
               normalization.  We export the last-conv feature maps together
               with the classifier kernel/bias (see scripts/export_onnx.py) and
               combine them: saliency = ReLU(sum_k W[class,k] * fm_k).

The offline path therefore still produces a genuine, explainable heatmap - it
does not silently degrade to a placeholder.
"""
from __future__ import annotations

import numpy as np

_CONFIDENCE_EXPLAIN_SAMPLE = 16  # spatial downsample for cheap CAM


def _normalize(heatmap: np.ndarray) -> np.ndarray:
    h = heatmap.astype(np.float32)
    mn, mx = float(h.min()), float(h.max())
    if mx - mn < 1e-8:
        return np.zeros_like(h)
    return (h - mn) / (mx - mn)


class TFGradCAM:
    """Grad-CAM for a tf.keras model (online path)."""

    def __init__(self, model, layer=None, input_size: int = 224):
        import tensorflow as tf
        self.tf = tf
        self.model = model
        self.input_size = input_size
        # The last 4D (conv) output of the backbone - works for MobileNetV2
        # (layer 'out_relu') and generic feature extractors.
        self.target_layer = layer or self._find_last_conv(model)

    @staticmethod
    def _find_last_conv(model):
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:   # N,H,W,C
                return layer.name
        raise ValueError("No 4D layer found - cannot compute Grad-CAM.")

    def heatmap(self, rgb: np.ndarray, class_index: int) -> np.ndarray:
        import tensorflow as tf
        x = tf.convert_to_tensor(rgb.astype(np.float32) / 255.0)[None, ...]
        x = tf.image.resize(x, (self.input_size, self.input_size))
        with tf.GradientTape() as tape:
            tape.watch(x)
            conv_out = self.model.get_layer(self.target_layer)(x)
            logits = self.model(x)
            # class_index handled AFTER softmax is unnecessary; use raw logit
            class_score = logits[0, class_index]
        grads = tape.gradient(class_score, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))          # alpha_k
        cam = tf.reduce_sum(weights * conv_out[0], axis=-1)   # weighted maps
        cam = tf.nn.relu(cam).numpy()
        cam = np.asarray(self.tf.image.resize(
            cam[..., None], (self.input_size, self.input_size))[..., 0])
        return _normalize(cam)


class ONNXCAM:
    """
    Weights-based CAM for an ONNX export.

    Expects a cam-model with outputs named:
        features : (1, H, W, C) last-conv feature maps (pre-GAP)
        kernel   : (C, num_classes) classifier weights
        bias     : (num_classes,)
    """

    def __init__(self, session, kernel_name: str = "kernel", bias_name: str = "bias",
                 features_name: str = "features", input_size: int = 224,
                 input_name: str = "input"):
        self.session = session
        self.kernel_name = kernel_name
        self.bias_name = bias_name
        self.features_name = features_name
        self.input_size = input_size
        self.input_name = input_name

    def heatmap(self, rgb: np.ndarray, class_index: int) -> np.ndarray:
        x = _preprocess_input(rgb, self.input_size)
        outputs = self.session.run(
            [self.features_name, self.kernel_name, self.bias_name],
            {self.input_name: x},
        )
        features, kernel, bias = outputs
        fm = features[0]                                  # (H, W, C)
        w = np.asarray(kernel)                            # (C, classes)
        b = np.asarray(bias)
        saliency = np.einsum("hwk,k->hw", fm, w[:, class_index])
        if saliency.ndim == 2 and b.size:
            saliency = saliency + float(b[class_index]) / max(1.0, fm.shape[0] * fm.shape[1])
        saliency = np.maximum(saliency, 0.0)
        heatmap = _resize_to(saliency, self.input_size)
        return _normalize(heatmap)


# ---------------------------------------------------------------------------
def _preprocess_input(rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 (any HxW) -> float32 (1, size, size, 3) in [0,1].

    Mirrors the training validation pipeline (Resize(256) -> CenterCrop(224)):
    aspect-preserving resize of the shorter side to a fixed length, then a
    center crop.  Directly stretching to a square changes aspect ratio and
    includes border content the model never saw during training, which
    collapses real photos onto a few common classes.
    """
    import numpy as np
    from PIL import Image
    h, w = rgb.shape[:2]
    if (h, w) == (size, size):
        return (rgb.astype(np.float32) / 255.0)[None, ...]
    short = size * 256 // 224  # torchvision Resize target for the shorter side
    if h < w:
        nh, nw = short, int(round(w * short / h))
    else:
        nw, nh = short, int(round(h * short / w))
    im = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
    top = (nh - size) // 2
    left = (nw - size) // 2
    im = im.crop((left, top, left + size, top + size))
    return (np.asarray(im).astype(np.float32) / 255.0)[None, ...]


def _resize_to(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    h, w = arr.shape[:2]
    if (h, w) != (size, size):
        arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size)))
        return arr.astype(np.float32) / 255.0
    return arr
