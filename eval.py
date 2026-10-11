import torch
from lib.models.mnist_model import MNISTNet
from lib.data.dataset import get_mnist_loaders
from helpers.eval_utils import evaluate


def main():
    # Evaluation settings
    batch_size = 1000
    checkpoint_path = "checkpoints/mnist_best.pt"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model and load weights
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    # Get test loader
    _, test_loader = get_mnist_loaders(batch_size)

    # Evaluate
    test_loss, accuracy = evaluate(model, device, test_loader)
    print(f"Test set: Average loss: {test_loss:.4f}, " f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
