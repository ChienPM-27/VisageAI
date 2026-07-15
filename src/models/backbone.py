"""
backbone.py — DINOv2 feature extractor with true GPU batch inference.

v0.4 additions:
  - extract_batch(): processes a list of BGR images as a single GPU tensor,
    ~10-16x faster than looping extract_features() one image at a time.
  - extract_features() still available for single-image use (main.py).

Supported models:
    dinov2_vits14 : 384-d  (default — fast, accurate enough for rating)
    dinov2_vitb14 : 768-d
    dinov2_vitl14 : 1024-d
    dinov2_vitg14 : 1536-d
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class DINOv2Extractor:

    # ImageNet normalisation used by DINOv2
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_name: str = "dinov2_vits14"):
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        dim_map = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        self.embed_dim = dim_map.get(model_name, 384)

        print(f"Initializing DINOv2 ({model_name}) on device: {self.device}...")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.model.to(self.device)
            self.model.eval()
            print("DINOv2 backbone loaded successfully.")
        except Exception as e:
            print(f"Error loading DINOv2: {e}\nFalling back to mock model.")
            self.model = self._mock_model()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _mock_model(self):
        dim = self.embed_dim

        class MockModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.num_patches = 256  # (224/14)²

            def forward_features(self, x):
                B = x.shape[0]
                cls   = torch.randn(B, self.dim, device=x.device)
                patch = torch.randn(B, self.num_patches, self.dim, device=x.device)
                return {"x_norm_clstoken": cls,
                        "x_norm_patchtokens": patch}

            def forward(self, x):
                return self.forward_features(x)["x_norm_clstoken"]

        return MockModel(dim).to(self.device)

    def _bgr_to_tensor(self, bgr_image: np.ndarray) -> torch.Tensor:
        """Convert a single BGR numpy image to a normalised CHW tensor."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._transform(pil)  # (3, 224, 224)

    def _forward_batch(self, batch_tensor: torch.Tensor):
        """
        Run one forward pass on a (B, 3, 224, 224) GPU tensor.
        Returns cls_tokens (B, D) and mean_patch_tokens (B, D).
        """
        batch_tensor = batch_tensor.to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                out = self.model.forward_features(batch_tensor)
                cls   = out.get("x_norm_clstoken",
                                batch_tensor.new_zeros(batch_tensor.shape[0], self.embed_dim))
                patch = out.get("x_norm_patchtokens",
                                batch_tensor.new_zeros(batch_tensor.shape[0], 1, self.embed_dim))
            else:
                cls   = self.model(batch_tensor)          # (B, D)
                patch = cls.unsqueeze(1)                  # dummy

        mean_patch = patch.mean(dim=1)                    # (B, D)
        return cls, mean_patch                            # both on GPU

    # ─── Public API: single image ─────────────────────────────────────────────

    def extract_features(self, cv2_image: np.ndarray,
                         pool_strategy: str = "both") -> dict:
        """
        Extracts DINOv2 features from a single BGR OpenCV image.

        pool_strategy:
            'cls'  — CLS token only  (384-d for vits14)
            'mean' — Mean of patch tokens (384-d), better for regression
            'both' — Returns CLS, mean-pool, and their concatenation (768-d)

        Returns dict with requested embeddings as numpy arrays.
        """
        tensor = self._bgr_to_tensor(cv2_image).unsqueeze(0)  # (1,3,224,224)
        cls, mean_patch = self._forward_batch(tensor)

        cls_np  = cls[0].cpu().numpy()
        mean_np = mean_patch[0].cpu().numpy()

        result = {"pool_strategy": pool_strategy}
        if pool_strategy in ("cls", "both"):
            result["cls_token"] = {
                "embedding":    cls_np,
                "shape":        list(cls_np.shape),
                "sample_values": [float(x) for x in cls_np[:10]],
            }
        if pool_strategy in ("mean", "both"):
            result["mean_pool"] = {
                "embedding":    mean_np,
                "shape":        list(mean_np.shape),
                "sample_values": [float(x) for x in mean_np[:10]],
            }
        if pool_strategy == "both":
            concat_np = np.concatenate([cls_np, mean_np])
            result["concat"] = {
                "embedding": concat_np,
                "shape":     list(concat_np.shape),
            }
        result["note"] = (
            "CLS token = global semantic repr. "
            "Mean pool = stable for regression (El Banani et al., 2023). "
        )
        return result

    # ─── Public API: batch inference ─────────────────────────────────────────

    def extract_batch(self, cv2_images: list[np.ndarray],
                      pool_strategy: str = "cls") -> list[np.ndarray]:
        """
        True GPU batch inference: processes all images in a single forward pass.

        Args:
            cv2_images    : list of BGR numpy arrays (any length)
            pool_strategy : 'cls' | 'mean' | 'both'
                            'both' returns concatenated [CLS ; mean] = 768-d per image.

        Returns:
            list of 1-D numpy arrays, one per input image.
            Shape per element: (384,) for cls/mean, (768,) for both.
        """
        if not cv2_images:
            return []

        # Preprocess all images on CPU → stack → single GPU transfer
        tensors = torch.stack([self._bgr_to_tensor(img) for img in cv2_images])
        cls, mean_patch = self._forward_batch(tensors)   # (B, D) each, on GPU

        cls_np  = cls.cpu().numpy()         # (B, D)
        mean_np = mean_patch.cpu().numpy()  # (B, D)

        if pool_strategy == "cls":
            return [cls_np[i] for i in range(len(cv2_images))]
        elif pool_strategy == "mean":
            return [mean_np[i] for i in range(len(cv2_images))]
        else:  # "both"
            return [np.concatenate([cls_np[i], mean_np[i]])
                    for i in range(len(cv2_images))]
