import logging

import torch
from torch.nn import Module, utils
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from network.early_stopping import EarlyStopping
from settings import EarlyStoppingSettings, MonitorType, TrainingSettings

logger = logging.getLogger(__name__)


class LstmTrainer:
    """A class for training a machine learning model using PyTorch."""

    def __init__(
            self,
            lstm: Module,
            loss_fn: Module,
            accuracy_metric: Metric,
            optimizer: Optimizer,
            early_stopping_settings: EarlyStoppingSettings,
    ) -> None:
        """Initialize the components required for processing recurrent neural networks.

        :param lstm: Recurrent neural network module used for model definition.
        :param loss_fn: Loss function module used for calculating the model's error.
        :param accuracy_metric: Metric module used for evaluating the model's accuracy.
        :param optimizer: Optimizer module used for updating the model's parameters.
        :param early_stopping_settings: Settings for early stopping of the training process.
        """
        self.lstm = lstm
        self.loss_fn = loss_fn
        self.accuracy_metric = accuracy_metric
        self.optimizer = optimizer
        self.early_stopping_settings = early_stopping_settings
        self.device = lstm.parameters().__next__().device

    def train_lstm(
            self,
            training_dataloader: DataLoader,
            testing_dataloader: DataLoader,
            settings: TrainingSettings,
    ) -> tuple[list[float], list[float]]:
        """Train a recurrent neural network (LSTM) on a provided dataset using stochastic gradient descent (SGD).

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

        # Initialize early stopping if enabled
        early_stopping = EarlyStopping(self.early_stopping_settings) if self.early_stopping_settings.enabled else None

        # Train the model
        self.lstm.train()

        for epoch_idx in range(1, settings.n_epochs + 1):
            # prepare the model for training - turn on dropout, batch norm, etc.
            self.lstm.train()
            # clear the gradients
            self.lstm.zero_grad()

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

            # Check early stopping if enabled
            if early_stopping is not None:
                monitor_score = accuracy_score if self.early_stopping_settings.monitor == MonitorType.ACCURACY\
                    else loss_score

                if early_stopping(monitor_score, self.lstm):
                    logger.info("Early stopping at epoch %d", epoch_idx)
                    if early_stopping.best_model_state is not None:
                        self.lstm.load_state_dict(early_stopping.best_model_state)
                        logger.info("Restored best model state with %s: %.4f",
                                    self.early_stopping_settings.monitor.value, early_stopping.best_score)
                    break

        return all_losses, all_accuracies

    def _train_on_batch(self, training_dataloader: DataLoader) -> float:
        current_loss = 0.0
        batch_count = 0

        for label_tensor, text_tensor in training_dataloader:
            label_tensor_dev = label_tensor.to(self.device)
            text_tensor_dev = text_tensor.to(self.device)

            output = self.lstm.forward(text_tensor_dev)
            loss = self.loss_fn(output, label_tensor_dev)

            loss.backward()
            utils.clip_grad_norm_(self.lstm.parameters(), 3)
            self.optimizer.step()
            self.optimizer.zero_grad()

            current_loss += loss.item()
            batch_count += 1

        return current_loss / batch_count if batch_count > 0 else 0.0

    def _compute_accuracy(self, testing_dataloader: DataLoader) -> float:
        self.accuracy_metric.reset()
        self.lstm.eval()

        with torch.no_grad():
            for label_tensor, text_tensor in testing_dataloader:
                label_tensor_dev = label_tensor.to(self.device)
                text_tensor_dev = text_tensor.to(self.device)

                output = self.lstm.forward(text_tensor_dev)
                self.accuracy_metric.update(output, label_tensor_dev)

        return self.accuracy_metric.compute().item()
