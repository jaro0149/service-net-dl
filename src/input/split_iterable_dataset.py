import random
from collections.abc import Iterator
from typing import TypeVar

from torch.utils.data import IterableDataset

T = TypeVar("T")


class SplitIterableDataset[T](IterableDataset[T]):
    """A dataset wrapper that splits an iterable dataset into train and test splits.

    This class allows the user to specify a ratio for splitting the dataset into training and test subsets.
    It provides an option to shuffle the data using a specified buffer size for more flexible data loading.
    This is especially useful for large-scale datasets that need to be processed incrementally.
    """

    def __init__(
            self,
            base_dataset: IterableDataset,
            *,
            is_train: bool = True,
            train_ratio: float = 0.8,
            seed: int = 2025,
            shuffle_buffer_size: int = 0,
    ) -> None:
        """Initialize a dataset instance with the ability to split into train and test sets and shuffling.

        :param base_dataset: The base dataset to be split and used. It must support iterable operations.
        :param train_ratio: The proportion of the dataset to be used for training. Must be a float between 0 and 1.
        :param is_train: Specifies whether to use the training subset or the testing subset of the dataset.
        :param seed: The random seed is used for deterministic shuffling and splitting. Default is 2025.
        :param shuffle_buffer_size: Size of the buffer used for shuffling. If set to 0, no shuffling will take place.
        """
        self.base_dataset = base_dataset
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[T]:
        """Create an iterator that yields data in either shuffled or non-shuffled order based on the buffer provided.

        :return: An iterator that generates elements from the split data, potentially shuffled.
        """
        # Create a shuffled iterator from the split data
        split_data = self._get_split_data()
        shuffled_data = self._shuffle_with_buffer(split_data)

        if self.shuffle_buffer_size:
            yield from shuffled_data
        else:
            yield from split_data

    def _get_split_data(self) -> Iterator[T]:
        """Get data for the current split using modulo-based splitting."""
        train_size = int(10 * self.train_ratio)  # e.g., 8 out of 10

        for idx, item in enumerate(self.base_dataset):
            adjusted_idx = (idx + self.seed) % 10
            is_train_item = adjusted_idx < train_size

            if (is_train_item and self.is_train) or (not is_train_item and not self.is_train):
                yield item

    def _shuffle_with_buffer(self, data_iterator: Iterator[T]) -> Iterator[T]:
        """Shuffle data using a buffer of a specified size."""
        buffer: list[T] = []

        # Fill initial buffer
        for _ in range(self.shuffle_buffer_size):
            try:
                buffer.append(next(data_iterator))
            except StopIteration:
                break

        # Continue yielding shuffled items while refilling the buffer
        for item in data_iterator:
            if buffer:
                # Randomly select and yield an item from the buffer
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                # Replace the yielded item with a new item
                buffer[idx] = item

        # Yield remaining items in buffer in random order
        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)
