import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment


def get_clip_iqa(path: str) -> float:
    """Get the CLIP-IQA score of an image.

    Args:
        path (`str`):
            The path to the image.

    Returns:
        `str`:
            The CLIP-IQA score.
    """
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32)
    return metric(tensor.unsqueeze(0)).item()
