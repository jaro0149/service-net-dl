import logging
import time

import torch
from torch import Generator
from torch.nn import NLLLoss
from torch.optim import SGD
from torch.utils import data
from torchmetrics import Accuracy

from char_rnn import CharRNN
from forecaster import Forecaster
from input_transforms import N_LETTERS
from model_trainer import ModelTrainer
from names_dataset import NamesDataset
from plotter import plot_confusion_matrix, plot_loss_and_accuracy
from settings import DatasetSettings, ModelSettings, TrainingSettings
from torch_utils import get_device, log_model_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _run() -> None:
    training_settings = TrainingSettings()
    model_settings = ModelSettings()
    dataset_settings = DatasetSettings()
    logger.info("Starting training with settings: %s, %s, %s", training_settings, model_settings, dataset_settings)

    device = get_device()
    torch.set_default_device(device)

    dataset = NamesDataset("data/names")
    num_classes = len(dataset.labels_uniq)
    logger.info("Loaded %d items of data", len(dataset))

    train_set, test_set = data.random_split(
        dataset=dataset,
        lengths=[dataset_settings.train_ratio, dataset_settings.test_ratio],
        generator=Generator(device=device).manual_seed(2024),
    )
    logger.info("Split into train set (%d) and test set (%d)", len(train_set), len(test_set))

    rnn = CharRNN(
        input_size=N_LETTERS,
        hidden_size=model_settings.n_hidden_units,
        output_size=num_classes,
    ).to(device)
    log_model_info(rnn, (1, 1, N_LETTERS))

    loss_fn = NLLLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes)
    optimizer = SGD(params=rnn.parameters(), lr=training_settings.learning_rate)

    logger.info("Training on data set with n = %d", len(train_set))
    start = time.time()
    model_trainer = ModelTrainer(
        rnn=rnn,
        loss_fn=loss_fn,
        accuracy_metric=accuracy_metric,
        optimizer=optimizer,
    )
    all_losses, all_accuracies = model_trainer.train_rnn(
        training_data=train_set,
        testing_data=test_set,
        settings=training_settings,
    )
    end = time.time()
    logger.info("Training took %f seconds", end - start)

    # plot results
    forecaster = Forecaster(
        rnn=rnn,
        classes=dataset.labels_uniq,
    )
    all_forecasts, all_targets = forecaster.forecast_on_testing_data(test_set)

    plot_loss_and_accuracy(losses=all_losses, accuracies=all_accuracies)
    plot_confusion_matrix(
        classes=dataset.labels_uniq,
        forecasts=all_forecasts,
        targets=all_targets,
    )

    # save a trained model with the current time in the filename
    model_path = f"models/model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(rnn.state_dict(), model_path)
    logger.info("Saved trained model to %s", model_path)

if __name__ == "__main__":
    _run()
