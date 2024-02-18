"""Validation step.
"""
import torch


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            data, _ = batch
            data = data.to(device)

            recon_batch = model(data)

            loss = loss_fn(recon_batch, data.view(-1, 784))

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss
