import logging

import torch
from torch.nn import Module, utils
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from settings import TrainingSettings

logger = logging.getLogger(__name__)


class RnnTrainer:
    """A class for training a machine learning model using PyTorch."""

    def __init__(
            self,
            rnn: Module,
            loss_fn: Module,
            accuracy_metric: Metric,
            optimizer: Optimizer,
    ) -> None:
        """Initialize the components required for processing recurrent neural networks.

        :param rnn: Recurrent neural network module used for model definition.
        :param loss_fn: Loss function module used for calculating the model's error.
        :param accuracy_metric: Metric module used for evaluating the model's accuracy.
        :param optimizer: Optimizer module used for updating the model's parameters.
        """
        self.rnn = rnn
        self.loss_fn = loss_fn
        self.accuracy_metric = accuracy_metric
        self.optimizer = optimizer

    def train_rnn(
            self,
            training_dataloader: DataLoader,
            testing_dataloader: DataLoader,
            settings: TrainingSettings,
    ) -> tuple[list[float], list[float]]:
        """Train a recurrent neural network (RNN) on a provided dataset using stochastic gradient descent (SGD).

        The model's performance is logged at regular intervals, and a record of the losses
        across epochs is maintained for potential analysis.

        :param training_dataloader: DataLoader providing training samples with batching and shuffling.
        :param testing_dataloader: DataLoader providing testing samples for evaluation.
        :param settings: Training configuration containing epochs, batch size, reporting interval, and learning rate.
        :return: A tuple containing two lists:
            - The first list includes the training loss values for each epoch.
            - The second list includes the accuracy values for each epoch.
        """
        # Keep track of losses and accuracies for plotting
        all_losses: list[float] = []
        all_accuracies: list[float] = []

        # Train the model
        self.rnn.train()

        for epoch_idx in range(1, settings.n_epochs + 1):
            # prepare the model for training - turn on dropout, batch norm, etc.
            self.rnn.train()
            # clear the gradients
            self.rnn.zero_grad()

            # train on each batch and compute mean loss
            loss_score = self._train_on_batch(training_dataloader)

            # compute accuracy on the test dataset
            accuracy_score = self._compute_accuracy(testing_dataloader)

            # save and optionally log the results
            all_accuracies.append(accuracy_score)
            all_losses.append(loss_score)

            if epoch_idx % settings.report_every == 0:
                logger.info("%d (%d%%): \t average batch loss = %.4f, accuracy = %.4f",
                            epoch_idx, int(epoch_idx / settings.n_epochs * 100), all_losses[-1], accuracy_score)

        return all_losses, all_accuracies

    def _train_on_batch(self, training_dataloader: DataLoader) -> float:
        current_loss = 0.0
        batch_count = 0

        for label_tensor, text_tensor in training_dataloader:
            output = self.rnn.forward(text_tensor)
            loss = self.loss_fn(output, label_tensor)

            loss.backward()
            utils.clip_grad_norm_(self.rnn.parameters(), 3)
            self.optimizer.step()
            self.optimizer.zero_grad()

            current_loss += loss.item()
            batch_count += 1

        return current_loss / batch_count if batch_count > 0 else 0.0

    def _compute_accuracy(self, testing_dataloader: DataLoader) -> float:
        self.accuracy_metric.reset()
        self.rnn.eval()

        with torch.no_grad():
            for label_tensor, text_tensor in testing_dataloader:
                output = self.rnn.forward(text_tensor)
                self.accuracy_metric.update(output, label_tensor)

        return self.accuracy_metric.compute().item()
