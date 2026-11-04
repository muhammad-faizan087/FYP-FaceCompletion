import os
import csv
import math

import torch
import lpips

from torch.utils.data import DataLoader

from configs.config import (
    IMG_DIR,
    ACTIVE_SPLIT_DIR,
    CHECKPOINT_DIR,
    PREVIEW_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.dataset import CelebAHQ256
from src.masks import make_mask
from src.models.unet import UNet

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


# Four evaluation categories
MASK_RANGES = {
    "easy_10_20": (0.10, 0.20),
    "medium_20_30": (0.20, 0.30),
    "hard_30_40": (0.30, 0.40),
    "very_hard_40_50": (0.40, 0.50),
}


def evaluate_category(
    model,
    dataloader,
    device,
    psnr_metric,
    ssim_metric,
    lpips_metric,
    mask_range,
):
    total_hole_error = 0.0
    total_hole_squared_error = 0.0
    total_hole_pixels = 0

    total_lpips = 0.0
    total_samples = 0
    skipped_samples = 0

    psnr_metric.reset()
    ssim_metric.reset()

    with torch.no_grad():

        for batch in dataloader:

            target = batch["image"].to(device)
            masked = batch["masked"].to(device)
            mask = batch["mask"].to(device)

            # Actual missing-area percentage for each image
            actual_ratios = mask.mean(dim=(1, 2, 3))

            # Keep only samples whose actual mask ratio
            # belongs to the requested category
            valid = (
                (actual_ratios >= mask_range[0]) &
                (actual_ratios <= mask_range[1])
            )

            skipped_samples += (~valid).sum().item()

            if not valid.any():
                continue

            target = target[valid]
            masked = masked[valid]
            mask = mask[valid]

            # Model input: masked RGB + mask channel
            model_input = torch.cat([masked, mask], dim=1)

            prediction = model(model_input)

            # Preserve visible pixels and fill missing pixels
            completed = (
                masked * (1 - mask) +
                prediction * mask
            )

            # 1. Hole MAE and Hole PSNR

            error = (prediction - target) * mask

            total_hole_error += error.abs().sum().item()

            total_hole_squared_error += error.square().sum().item()

            total_hole_pixels += (
                mask.sum().item() * target.shape[1]
            )

            # 2. Full-image PSNR and SSIM

            psnr_metric.update(completed, target)

            ssim_metric.update(completed, target)

            # 3. Full-image LPIPS

            lpips_values = lpips_metric(
                completed, target
            ).view(-1)

            total_lpips += lpips_values.sum().item()

            total_samples += target.size(0)

    if total_samples == 0:
        return None

    # Hole MAE
    hole_mae = total_hole_error / max(total_hole_pixels, 1)

    # Hole PSNR
    hole_mse = (
        total_hole_squared_error /
        max(total_hole_pixels, 1)
    )

    if hole_mse == 0:
        hole_psnr = float("inf")
    else:
        hole_psnr = 10 * math.log10(4.0 / hole_mse)

    # Full-image metrics
    full_psnr = psnr_metric.compute().item()

    full_ssim = ssim_metric.compute().item()

    full_lpips = total_lpips / total_samples

    return {
        "samples": total_samples,
        "skipped": skipped_samples,
        "hole_mae": hole_mae,
        "hole_psnr": hole_psnr,
        "full_psnr": full_psnr,
        "full_ssim": full_ssim,
        "full_lpips": full_lpips,
    }


def main():

    os.makedirs(PREVIEW_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    # Load best checkpoint

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = UNet(
        base_channels=checkpoint["base_channels"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint loaded successfully.")

    # Initialize evaluation metrics

    psnr_metric = PeakSignalNoiseRatio(
        data_range=2.0
    ).to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=2.0
    ).to(device)

    lpips_metric = lpips.LPIPS(
        net="alex"
    ).to(device)

    lpips_metric.eval()

    # Evaluate all four mask categories

    all_results = []

    for category, ratio_range in MASK_RANGES.items():

        print("\n" + "=" * 55)
        print("Evaluating:", category)
        print("Mask range:", ratio_range)
        print("=" * 55)

        test_dataset = CelebAHQ256(
            img_dir=IMG_DIR,
            split_file=f"{ACTIVE_SPLIT_DIR}/test.txt",
            augment=False,
            mask_fn=make_mask,
            fixed_masks=True,
            mask_seed=42,
            mask_ratio_range=ratio_range
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        results = evaluate_category(
            model=model,
            dataloader=test_dataloader,
            device=device,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            lpips_metric=lpips_metric,
            mask_range=ratio_range,
        )

        if results is None:
            print("No valid samples found for this category.")
            continue

        results["category"] = category

        all_results.append(results)

        print("Valid samples:", results["samples"])
        print("Skipped samples:", results["skipped"])
        print("Hole MAE:", round(results["hole_mae"], 5))
        print("Hole PSNR:", round(results["hole_psnr"], 3))
        print("Full PSNR:", round(results["full_psnr"], 3))
        print("Full SSIM:", round(results["full_ssim"], 4))
        print("Full LPIPS:", round(results["full_lpips"], 4))

    # Save evaluation results as CSV

    csv_path = os.path.join(
        PREVIEW_DIR,
        "evaluation_metrics.csv"
    )

    fieldnames = [
        "category",
        "samples",
        "skipped",
        "hole_mae",
        "hole_psnr",
        "full_psnr",
        "full_ssim",
        "full_lpips",
    ]

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("Results saved to:", csv_path)
    print("=" * 55)


if __name__ == "__main__":
    main()
