"""
SmartQC — Grad-CAM explainability module.
 
Loads the trained PyTorch weights (model/smartqc_weights.pt, produced by
train.py) and computes a Grad-CAM heatmap showing which region of the
image most influenced the model's prediction.
 
This runs on PyTorch directly (not the ONNX/NPU path) since Grad-CAM
needs access to intermediate activations and gradients — it's an
explainability aid alongside the main ONNX Runtime inference path, not
a replacement for it.
"""
 
from pathlib import Path
 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
 
WEIGHTS_PATH = Path(__file__).parent / "model" / "smartqc_weights.pt"
IMG_SIZE = 224
 
 
class GradCAM:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        checkpoint = torch.load(weights_path, map_location="cpu")
        self.class_names = checkpoint["class_names"]
 
        self.model = models.mobilenet_v2()
        self.model.classifier[1] = torch.nn.Linear(self.model.last_channel, len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
 
        # Last convolutional block of MobileNetV2's feature extractor —
        # gives a good spatial resolution / semantic depth trade-off for CAM.
        self.target_layer = self.model.features[-1]
 
        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
 
    def _save_activation(self, module, input, output):
        self._activations = output.detach()
 
    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()
 
    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        return torch.from_numpy(img)
 
    def generate(self, frame_bgr: np.ndarray):
        """
        Returns (overlay_bgr, label, confidence) where overlay_bgr is the
        original frame with a Grad-CAM heatmap alpha-blended on top.
        """
        x = self._preprocess(frame_bgr)
        x.requires_grad_(False)
 
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())
 
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
 
        activations = self._activations[0]        # (C, H, W)
        gradients = self._gradients[0]              # (C, H, W)
 
        weights = gradients.mean(dim=(1, 2))         # (C,) — global average pool of gradients
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * activations[c]
        cam = F.relu(cam)
 
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
 
        cam_resized = cv2.resize(cam, (frame_bgr.shape[1], frame_bgr.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.4, 0)
 
        return overlay, self.class_names[class_idx], confidence
 