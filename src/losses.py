import torch

def masked_l1_loss(prediction, target, mask):
    """
    prediction: Model output       (B, 3, H, W)
    target:     Original image     (B, 3, H, W)
    mask:       Missing-region mask (B, 1, H, W)
    """

    pixel_difference = torch.abs(prediction - target)

    masked_difference = pixel_difference * mask

    loss = masked_difference.sum() / (
        mask.sum() * prediction.shape[1] + 1e-8
    )

    return loss
