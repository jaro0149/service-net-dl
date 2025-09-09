from abc import ABC, abstractmethod
from collections.abc import Callable

from torch.utils.data import DataLoader

from input.input_transforms import combine_transforms, to_lowercase
from input.network_services_dataset import NetworkServicesDataset, collate_dataset_batch
from input.split_iterable_dataset import SplitIterableDataset
from input.text_augmentations import TextAugmentations
from settings import TextAugmentationSettings, TrainingSettings


class DatasetLoaderFactory(ABC):
    """Abstract factory for creating dataset loaders."""

    _train_mode: bool = False

    def create_dataset_loader(
            self,
            training_settings: TrainingSettings,
            aug_settings: TextAugmentationSettings,
    ) -> tuple[list[str], DataLoader]:
        """Create a DataLoader for training or evaluation based on provided training and augmentation settings.

        :param training_settings: Settings related to training, including parameters
             such as batch size and training ratio.
        :param aug_settings: Settings for text augmentation.
        :return: A tuple containing the unique labels list from the dataset and a DataLoader
             configured to use an iterable dataset with specified transformations and training
             parameters. The DataLoader is pinned for memory optimization, and a custom collate
             function is used for batching.
        """
        transforms = self._transforms(aug_settings)
        shuffle_buffer_size = self._shuffle_buffer_size(training_settings)

        base_dataset = NetworkServicesDataset(
            data_dir=training_settings.data_dir,
            transforms=transforms,
        )
        iterable_dataset = SplitIterableDataset(
            base_dataset=base_dataset,
            train_ratio=training_settings.train_ratio,
            is_train=self._train_mode,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=training_settings.seed,
        )
        dataloader = DataLoader(
            dataset=iterable_dataset,
            batch_size=training_settings.n_batch_size,
            pin_memory=self._train_mode,
            collate_fn=collate_dataset_batch,
        )
        return base_dataset.labels_uniq, dataloader

    @abstractmethod
    def _transforms(self, aug_settings: TextAugmentationSettings) -> Callable[[str], str] | None:
        """Get the transforms for dataset creation.

        :param aug_settings: Settings for text augmentations or transformations.
        :return: The text transformations.
        """
        ...

    @abstractmethod
    def _shuffle_buffer_size(self, training_settings: TrainingSettings) -> int:
        """Get the shuffle buffer size for dataset creation.

        :param training_settings: Settings related to model training.
        :return: The shuffle buffer size.
        """
        ...


class TrainingDatasetLoaderFactory(DatasetLoaderFactory):
    """Factory for creating training dataset loaders."""

    _train_mode = True

    def _transforms(self, aug_settings: TextAugmentationSettings) -> Callable[[str], str] | None:
        augmenter = TextAugmentations(aug_settings)
        return combine_transforms(
            augmenter.augment_text,
            to_lowercase,
        )

    def _shuffle_buffer_size(self, training_settings: TrainingSettings) -> int:
        return training_settings.shuffle_buffer_size


class TestingDatasetLoaderFactory(DatasetLoaderFactory):
    """Factory for creating testing dataset loaders."""

    _train_mode = False

    def _transforms(self, _: TextAugmentationSettings) -> Callable[[str], str] | None: # type: ignore[reportIncompatibleMethodOverride]
        return None

    def _shuffle_buffer_size(self, _: TrainingSettings) -> int: # type: ignore[reportIncompatibleMethodOverride]
        return 0
