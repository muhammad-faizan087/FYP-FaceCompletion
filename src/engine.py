import torch

from src.losses import masked_l1_loss


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        target = batch['image'].to(device)
        masked = batch['masked'].to(device)
        mask = batch['mask'].to(device)

        model_input = torch.cat([masked, mask], dim=1)

        with torch.set_grad_enabled(training):
            prediction = model(model_input)
            loss = masked_l1_loss(prediction, target, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = target.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples
