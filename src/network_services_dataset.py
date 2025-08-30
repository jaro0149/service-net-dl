from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from torch import Tensor, long, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from input_transforms import TXT_ENCODING, line_to_tensor


class NetworkServicesDataset(IterableDataset):
    """Represents a dataset of network service keywords and their corresponding labels derived from text files.

    This class is designed to load and preprocess text data from a specified directory,
    where each text file corresponds to a unique network service.
    Each file contains a list of service keywords.
    The class processes these keywords into tensors suitable for machine learning applications
    while also associating them with their labels.
    """

    def __init__(self, data_dir: str, transforms: Callable[[str], str] | None = None) -> None:
        """Create a new NetworkServicesDataset instance.

        :param data_dir: The path to the directory containing text files.
            Each text file should contain lines of text data where each line represents a service keyword.
        :param transforms: Text transformations to apply to each sample.
        """
        self.data_dir = data_dir
        self.transforms = transforms

        text_files = self._get_text_files(data_dir)
        self._create_labels_from_filenames(text_files)

    def _create_labels_from_filenames(self, text_files: list[Path]) -> None:
        labels_set = set()
        for filename in text_files:
            label = filename.stem
            labels_set.add(label)
        self.labels_uniq = list(labels_set)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over the dataset, yielding (label_tensor, data_tensor) tuples.

        :yield: A tuple containing the label tensor and data tensor for each sample.
        """
        text_files = self._get_text_files(self.data_dir)
        for filename in text_files:
            yield from self._process_text_file(filename)

    def _process_text_file(self, filename: Path) -> Iterator[tuple[Tensor, Tensor]]:
        label = filename.stem
        label_idx = self.labels_uniq.index(label)
        label_tensor = tensor([label_idx], dtype=long)

        with filename.open(encoding=TXT_ENCODING) as file:
            for line in file:
                keyword = line.strip()
                if not keyword:
                    # skip empty lines
                    continue
                if self.transforms:
                    keyword = self.transforms(keyword)

                data_tensor = line_to_tensor(keyword)
                yield label_tensor, data_tensor

    @staticmethod
    def _get_text_files(data_dir: str) -> list[Path]:
        return list(Path(data_dir).glob("*.txt"))


def collate_dataset_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Combine a batch of data samples into a format suitable for dataloader.

    This function takes a batch of data samples, where each sample is a tuple
    containing a label tensor and a data tensor. It stacks the label tensors
    into a single tensor and pads the sequence of data tensors to ensure they
    have the same length, making them suitable for further processing. Padding
    is applied with a value of 0.0.

    :param batch: A list of tuples, where each tuple contains a label tensor and a data tensor.
    :return: A tuple containing a tensor of stacked labels and a tensor of padded data tensors.
    """
    labels, data_tensors = zip(*batch, strict=False)
    stacked_labels = torch.stack(labels)
    padded_data = pad_sequence(list(data_tensors), batch_first=True, padding_value=0.0)
    return stacked_labels, padded_data
