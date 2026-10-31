import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import (
    IMG_DIR,
    ACTIVE_SPLIT_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    BASE_CHANNELS,
    NUM_WORKERS,
    SEED,
)

from src.dataset import CelebAHQ256
from src.masks import make_mask
from src.models.unet import UNet
from src.engine import run_epoch

def main():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
  )

  train_dataset = CelebAHQ256(
    img_dir=IMG_DIR,
    split_file=f"{ACTIVE_SPLIT_DIR}/train.txt",
    augment=True,
    mask_fn=make_mask,
    fixed_masks=False
  )

  val_dataset = CelebAHQ256(
      img_dir=IMG_DIR,
      split_file=f"{ACTIVE_SPLIT_DIR}/val.txt",
      augment=False,
      mask_fn=make_mask,
      fixed_masks=True,
  )

  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

  model = UNet(base_channels=BASE_CHANNELS).to(device)

  optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=LEARNING_RATE,
  )

  best_val_loss = float("inf")

  print("Device:", device)
  print("Train samples:", len(train_dataset))
  print("Validation samples:", len(val_dataset))

  for epoch in range(1, EPOCHS + 1):
      train_loss = run_epoch(model, train_dataloader, device, optimizer)
      val_loss = run_epoch(model, val_dataloader, device)

      print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

      checkpoint = {
          "epoch": epoch,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "train_loss": train_loss,
          "val_loss": val_loss,
          "base_channels": BASE_CHANNELS,
      }

      torch.save(
            checkpoint,
            os.path.join(CHECKPOINT_DIR, "last_model.pth"),
        )
      
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(
              checkpoint,
              os.path.join(CHECKPOINT_DIR, "best_model.pth"),
          )
          print("Best checkpoint updated.")

  print("Training complete.")
  print("Best validation loss:", best_val_loss)

if __name__ == "__main__":
    main()
