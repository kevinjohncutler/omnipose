import torch
import torch.nn.functional as F


def torch_zoom(img, scale_factor=1.0, dim=2, size=None, mode='bilinear'):
    """Resize torch tensor using interpolate."""
    target_size = [int(d * scale_factor) for d in img.shape[-dim:]] if size is None else size
    return F.interpolate(img, size=target_size, mode=mode, align_corners=False)
