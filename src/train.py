import logging

import torch

from input.dataloader_factory import TestingDatasetLoaderFactory, TrainingDatasetLoaderFactory
from settings import ModelSettings, TextAugmentationSettings, TrainingSettings
from utils.torch_utils import get_device, set_seed

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
    logger.info("Starting training with settings: %s, %s, %s", training_settings, model_settings, aug_settings)

    # set random seed for reproducibility
    set_seed(training_settings.seed)

    # get device used for storing tensors and training
    device = get_device()
    torch.set_default_device(device)

    # loading data with data-loaders
    train_dataloader = TrainingDatasetLoaderFactory().create_dataset_loader(
        training_settings=training_settings,
        aug_settings=aug_settings,
    )
    test_dataloader = TestingDatasetLoaderFactory().create_dataset_loader(
        training_settings=training_settings,
        aug_settings=aug_settings,
    )


if __name__ == "__main__":
    _run()
