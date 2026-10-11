import torch
import torch.optim as optim
import os
import wandb
from lib.models.mnist_model import MNISTNet
from lib.data.dataset import get_mnist_loaders
from helpers.training_utils import train_epoch
from helpers.eval_utils import evaluate


def main():
    # Initialize wandb
    wandb.init(
        project="mnist-pytorch",
        config={
            "architecture": "CNN",
            "dataset": "MNIST",
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 0.01,
        },
    )

    # Training settings
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr = wandb.config.learning_rate
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model
    model = MNISTNet().to(device)
    wandb.watch(model, log="all")  # Log gradients and parameters
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Training loop
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = evaluate(model, device, test_loader)

        print(f"Test set: Average loss: {test_loss:.4f}, " f"Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(save_dir, "mnist_best.pt")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)  # Save model to wandb

    wandb.finish()


if __name__ == "__main__":
    main()
