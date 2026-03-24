"""
LIME (Local Interpretable Model-agnostic Explanations) with superpixel segmentation.
"""
import torch
import numpy as np
from lime import lime_image

from data.preprocessing import denormalize


class LIMEExplainer:
    """LIME explainer using superpixel segmentation for cross-verification."""

    def __init__(self, model, num_samples: int = 1000, num_features: int = 10):
        self.model = model
        self.model.eval()
        self.explainer = lime_image.LimeImageExplainer()
        self.num_samples = num_samples
        self.num_features = num_features

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction function for LIME (expects HWC uint8/float images)."""
        from torchvision import transforms
        from config.settings import DATASET_CONFIG

        device = next(self.model.parameters()).device
        mean, std = DATASET_CONFIG["mean"], DATASET_CONFIG["std"]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        batch = []
        for img in images:
            if img.dtype == np.float64:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            from PIL import Image
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            batch.append(transform(pil_img))

        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)
        return probs.cpu().numpy()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate LIME explanation heatmap.

        Args:
            input_tensor: (1, 3, H, W) normalized tensor.
            target_class: Class index to explain.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        if target_class is None:
            with torch.no_grad():
                pred = self.model(input_tensor)
            target_class = pred.argmax(dim=1).item()

        # Denormalize for LIME (expects HWC [0,1] float image)
        img_denorm = denormalize(input_tensor.squeeze(0)).cpu().numpy()
        img_hwc = np.transpose(img_denorm, (1, 2, 0))  # (H, W, 3)

        explanation = self.explainer.explain_instance(
            img_hwc,
            self._predict_fn,
            top_labels=max(target_class + 1, 5),
            hide_color=0,
            num_samples=self.num_samples,
        )

        # Get mask from explanation
        _, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=False,
            num_features=self.num_features,
            hide_rest=False,
        )

        # Convert to importance map
        ind = explanation.local_exp.get(target_class, [])
        segments = explanation.segments
        attr_map = np.zeros_like(segments, dtype=np.float32)
        for seg_id, weight in ind:
            attr_map[segments == seg_id] = weight

        # Normalize
        attr_map = np.abs(attr_map)
        a_min, a_max = attr_map.min(), attr_map.max()
        if a_max - a_min > 1e-8:
            attr_map = (attr_map - a_min) / (a_max - a_min)
        else:
            attr_map = np.zeros_like(attr_map)
        return attr_map
