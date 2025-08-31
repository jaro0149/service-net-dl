import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from torchmetrics import ConfusionMatrix


def plot_loss_and_accuracy(losses: list[float], accuracies: list[float]) -> None:
    """Plot both training loss and accuracy on the same graph with a single y-axis.

    :param losses: A list of floating-point numbers representing the training loss values for each epoch.
    :param accuracies: A list of floating-point numbers representing the accuracy values for each epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot both loss and accuracy on the same y-axis
    color1 = "tab:red"
    color2 = "tab:blue"

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")

    ax.plot(losses, color=color1, label="Training Loss")
    ax.plot(accuracies, color=color2, label="Accuracy")

    # Add title and grid
    plt.title("Training Loss and Accuracy")
    ax.grid(visible=True, alpha=0.3)

    # Add legend
    ax.legend(loc="center right")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
        classes: list[str],
        forecasts: list[int],
        targets: list[int],
) -> None:
    """Plot a confusion matrix using given forecasted and target class labels.

    :param classes: A list of string labels for each class.
    :param forecasts: A list of integers representing predicted class indices.
    :param targets: A list of integers representing true class indices.
    """
    # Initialize PyTorch ConfusionMatrix metric
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=len(classes), normalize="true")

    # Convert to tensors and compute confusion matrix
    forecasts_tensor = torch.tensor(forecasts)
    targets_tensor = torch.tensor(targets)
    confusion = confusion_matrix(forecasts_tensor, targets_tensor)

    # Set up a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
