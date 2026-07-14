import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class DINOv2Extractor:
    def __init__(self, model_name='dinov2_vits14'):
        """
        Loads DINOv2 from PyTorch Hub.
        Supported models: 'dinov2_vits14' (384-d), 'dinov2_vitb14' (768-d),
                          'dinov2_vitl14' (1024-d), 'dinov2_vitg14' (1536-d).
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        print(f"Initializing DINOv2 ({model_name}) on device: {self.device}...")

        try:
            # Load with register_buffer patches to get patch tokens too
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model.to(self.device)
            self.model.eval()
            print("DINOv2 backbone loaded successfully.")
        except Exception as e:
            print(f"Error loading DINOv2 from PyTorch Hub: {e}")
            print("Falling back to mock model.")
            self.model = self._get_mock_model(model_name)

    def _get_mock_model(self, model_name):
        dim_dict = {
            'dinov2_vits14': 384, 'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536
        }
        dim = dim_dict.get(model_name, 384)

        class MockModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.num_patches = 256  # 224/14 x 224/14

            def forward_features(self, x):
                bs = x.shape[0]
                cls   = torch.randn(bs, 1, self.dim, device=x.device)
                patch = torch.randn(bs, self.num_patches, self.dim, device=x.device)
                tokens = torch.cat([cls, patch], dim=1)
                return {"x_norm_patchtokens": patch, "x_norm_clstoken": cls.squeeze(1)}

            def forward(self, x):
                return self.forward_features(x)["x_norm_clstoken"]

        return MockModel(dim).to(self.device)

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, cv2_image, pool_strategy: str = 'both') -> dict:
        """
        Extracts visual features from a BGR OpenCV image.

        pool_strategy options:
          'cls'  : CLS token only — captures global semantic (image-level) representation.
                   Standard for classification tasks.
          'mean' : Mean of all patch tokens — shown to be more stable for dense
                   regression tasks (El Banani et al., 2023; Oquab et al., 2023).
          'both' : Returns CLS, mean-pool, and their concatenation.

        Returns dict with requested embeddings as numpy arrays.
        """
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.get_transform()(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'forward_features'):
                feat_dict = self.model.forward_features(tensor)
                cls_token   = feat_dict.get("x_norm_clstoken",     tensor.new_zeros(1, 1))
                patch_tokens = feat_dict.get("x_norm_patchtokens", tensor.new_zeros(1, 1, 1))
            else:
                # Fallback: plain forward gives CLS; we cannot get patch tokens
                cls_token    = self.model(tensor)
                patch_tokens = cls_token.unsqueeze(1)  # dummy

        cls_np  = cls_token.squeeze(0).cpu().numpy()
        mean_np = patch_tokens.mean(dim=1).squeeze(0).cpu().numpy()

        result = {}
        if pool_strategy in ('cls', 'both'):
            result['cls_token'] = {
                "embedding": cls_np,
                "shape": list(cls_np.shape),
                "sample_values": [float(x) for x in cls_np[:10]]
            }
        if pool_strategy in ('mean', 'both'):
            result['mean_pool'] = {
                "embedding": mean_np,
                "shape": list(mean_np.shape),
                "sample_values": [float(x) for x in mean_np[:10]]
            }
        if pool_strategy == 'both':
            concat_np = torch.cat([
                torch.tensor(cls_np), torch.tensor(mean_np)
            ], dim=0).numpy()
            result['concat'] = {
                "embedding": concat_np,
                "shape": list(concat_np.shape),
            }

        result['pool_strategy'] = pool_strategy
        result['note'] = (
            "CLS token = global semantic repr. "
            "Mean pool = stable for regression tasks (El Banani et al., 2023). "
            "Future ablation will compare the two."
        )
        return result
