"""Training step.
"""


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        data, _ = batch
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch = model(data)

        loss = loss_fn(recon_batch, data.view(-1, 784))

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss
