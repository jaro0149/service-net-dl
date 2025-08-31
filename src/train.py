import logging
import time

import torch
from torch.nn import NLLLoss
from torch.optim import SGD
from torchmetrics import Accuracy

from input.dataloader_factory import TestingDatasetLoaderFactory, TrainingDatasetLoaderFactory
from input.input_transforms import N_LETTERS
from network.char_rnn import CharRNN
from network.rnn_trainer import RnnTrainer
from network.text_classifier import TextClassifier
from settings import ModelSettings, TextAugmentationSettings, TrainingSettings
from utils.plotter import plot_confusion_matrix, plot_loss_and_accuracy
from utils.torch_utils import get_device, log_model_info, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run() -> None:
    # load settings from the environment
    training_settings = TrainingSettings()
    model_settings = ModelSettings()
    aug_settings = TextAugmentationSettings()
    logger.info("Settings: %s, %s, %s", training_settings, model_settings, aug_settings)

    # set random seed for reproducibility
    set_seed(training_settings.seed)

    # get device used for storing tensors and training
    device = get_device()
    torch.set_default_device(device)

    # loading data with data-loaders
    labels, train_dataloader = TrainingDatasetLoaderFactory().create_dataset_loader(
        training_settings=training_settings,
        aug_settings=aug_settings,
    )
    _, test_dataloader = TestingDatasetLoaderFactory().create_dataset_loader(
        training_settings=training_settings,
        aug_settings=aug_settings,
    )
    num_classes = len(labels)
    logger.info("Loaded dataset classes: %s", labels)

    # create a model and log model info
    rnn = CharRNN(
        input_size=N_LETTERS,
        hidden_size=model_settings.n_hidden_units,
        output_size=num_classes,
    ).to(device)
    log_model_info(rnn, (1, 1, N_LETTERS))

    # preparation of the loss function, optimizer and metrics
    loss_fn = NLLLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes)
    optimizer = SGD(params=rnn.parameters(), lr=training_settings.learning_rate)

    # training loop
    logger.info("Starting training...")
    start = time.time()
    model_trainer = RnnTrainer(
        rnn=rnn,
        loss_fn=loss_fn,
        accuracy_metric=accuracy_metric,
        optimizer=optimizer,
    )
    all_losses, all_accuracies = model_trainer.train_rnn(
        training_dataloader=train_dataloader,
        testing_dataloader=test_dataloader,
        settings=training_settings,
    )
    end = time.time()
    logger.info("Training took %f seconds", end - start)

    # plot results
    forecaster = TextClassifier(
        rnn=rnn,
        classes=labels,
    )
    all_forecasts, all_targets = forecaster.classify_testing_data(test_dataloader)

    plot_loss_and_accuracy(
        losses=all_losses,
        accuracies=all_accuracies,
    )
    plot_confusion_matrix(
        classes=labels,
        forecasts=all_forecasts,
        targets=all_targets,
    )

    # save a trained model with the current time in the filename
    model_path = f"models/model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(rnn.state_dict(), model_path)
    logger.info("Saved trained model to %s", model_path)


if __name__ == "__main__":
    _run()
