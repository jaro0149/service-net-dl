import logging
import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, utils
from torch.optim import Optimizer
from torch.utils.data import Subset
from torchmetrics import Metric

from settings import TrainingSettings
from utils.torch_utils import IterableSubset

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
            training_data: Subset,
            testing_data: Subset,
            settings: TrainingSettings,
    ) -> tuple[list[float], list[float]]:
        """Train a recurrent neural network (RNN) on a provided dataset using stochastic gradient descent (SGD).

        The function processes the dataset in minibatches without the aid of data loaders, as input samples
        vary in size. The model's performance is logged at regular intervals, and a record of the losses
        across epochs is maintained for potential analysis.

        :param training_data: A list of training samples is represented as tensors, where each sample should
            include elements such as (label_tensor, text_tensor, label, text).
        :param testing_data: Similar to the training data, this is a list of testing samples represented as tensors.
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

            # create some minibatches
            batches = self._prepare_batches(training_data, settings.n_batch_size)

            # train on each minibatch and compute mean loss
            loss_score = self._train_on_batch(
                batches=batches,
                training_data=training_data,
            )

            # compute accuracy on the test dataset
            accuracy_score = self._compute_accuracy(
                testing_data=testing_data,
            )

            # save and optionally log the results
            all_accuracies.append(accuracy_score)
            all_losses.append(loss_score)

            if epoch_idx % settings.report_every == 0:
                logger.info("%d (%d%%): \t average batch loss = %.4f, accuracy = %.4f",
                            epoch_idx, int(epoch_idx / settings.n_epochs * 100), all_losses[-1], accuracy_score)

        return all_losses, all_accuracies

    def _train_on_batch(
            self,
            batches: list[np.ndarray],
            training_data: Subset,
    ) -> float:
        current_loss = 0.0
        for batch in batches:
            batch_loss: Tensor = torch.tensor(0.0)
            for label_tensor, text_tensor in IterableSubset(training_data, batch):
                output = self.rnn.forward(text_tensor)
                loss = self.loss_fn(output, label_tensor)
                batch_loss += loss

            batch_loss.backward()
            utils.clip_grad_norm_(self.rnn.parameters(), 3)
            self.optimizer.step()
            self.optimizer.zero_grad()
            current_loss += batch_loss.item() / len(batch)
        return current_loss / len(batches)

    def _compute_accuracy(
            self,
            testing_data: Subset,
    ) -> float:
        self.accuracy_metric.reset()
        self.rnn.eval()
        with torch.no_grad():
            for label_tensor, text_tensor in IterableSubset(testing_data):
                output = self.rnn.forward(text_tensor)
                self.accuracy_metric.update(output, label_tensor)
        return self.accuracy_metric.compute().item()

    @staticmethod
    def _prepare_batches(training_data: Subset, n_batch_size: int) -> list[np.ndarray]:
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        return np.array_split(batches, len(batches) // n_batch_size)
