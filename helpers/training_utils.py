import torch
from tqdm import tqdm
import wandb


def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Log training metrics
        wandb.log({"train_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

        if batch_idx % 100 == 0:
            pbar.set_postfix({"Loss": loss.item()})
