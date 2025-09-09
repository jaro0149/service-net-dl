import logging
import time

import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torchmetrics import Accuracy

from input.dataloader_factory import TestingDatasetLoaderFactory, TrainingDatasetLoaderFactory
from input.input_transforms import N_LETTERS
from network.char_lstm import CharLSTM
from network.lstm_trainer import LstmTrainer
from network.text_classifier import TextClassifier
from settings import EarlyStoppingSettings, LstmModelSettings, TextAugmentationSettings, TrainingSettings
from utils.plotter import plot_confusion_matrix, plot_loss_and_accuracy
from utils.torch_utils import get_device, log_model_info, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class _TrainingProcess:

    def run(self) -> None:
        self._load_settings()
        set_seed(self.training_settings.seed)
        self.device = get_device()
        self._create_dataloaders()
        self._create_model()
        self._prepare_training_functions()
        self._train_model()
        self._plot_results()
        self._save_model()

    def _load_settings(self) -> None:
        self.training_settings = TrainingSettings()
        self.lstm_model_settings = LstmModelSettings()
        self.aug_settings = TextAugmentationSettings()
        self.early_stopping_settings = EarlyStoppingSettings()
        logger.info("Settings: %s, %s, %s", self.training_settings, self.lstm_model_settings, self.aug_settings)

    def _create_dataloaders(self) -> None:
        self.labels, self.train_dataloader = TrainingDatasetLoaderFactory().create_dataset_loader(
            training_settings=self.training_settings,
            aug_settings=self.aug_settings,
        )
        _, self.test_dataloader = TestingDatasetLoaderFactory().create_dataset_loader(
            training_settings=self.training_settings,
            aug_settings=self.aug_settings,
        )
        self.num_classes = len(self.labels)
        logger.info("Loaded dataset classes: %s", self.labels)

    def _create_model(self) -> None:
        self.lstm = CharLSTM(
            input_size=N_LETTERS,
            output_size=self.num_classes,
            model_settings=self.lstm_model_settings,
        ).to(self.device)
        log_model_info(self.lstm, (1, 1, N_LETTERS))

    def _prepare_training_functions(self) -> None:
        self.loss_fn = NLLLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        self.optimizer = Adam(params=self.lstm.parameters(), lr=self.training_settings.learning_rate)

    def _train_model(self) -> None:
        logger.info("Starting training...")
        start = time.time()
        model_trainer = LstmTrainer(
            lstm=self.lstm,
            loss_fn=self.loss_fn,
            accuracy_metric=self.accuracy_metric,
            optimizer=self.optimizer,
            early_stopping_settings=self.early_stopping_settings,
        )
        self.all_losses, self.all_accuracies = model_trainer.train_lstm(
            training_dataloader=self.train_dataloader,
            testing_dataloader=self.test_dataloader,
            settings=self.training_settings,
        )
        end = time.time()
        logger.info("Training took %f seconds", end - start)

    def _plot_results(self) -> None:
        forecaster = TextClassifier(
            lstm=self.lstm,
            classes=self.labels,
        )
        all_forecasts, all_targets = forecaster.classify_testing_data(self.test_dataloader)

        plot_loss_and_accuracy(
            losses=self.all_losses,
            accuracies=self.all_accuracies,
        )
        plot_confusion_matrix(
            classes=self.labels,
            forecasts=all_forecasts,
            targets=all_targets,
        )

    def _save_model(self) -> None:
        model_path = f"models/model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(self.lstm.state_dict(), model_path)
        logger.info("Saved trained model to %s", model_path)


if __name__ == "__main__":
    _TrainingProcess().run()
