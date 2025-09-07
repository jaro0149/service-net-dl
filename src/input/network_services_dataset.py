from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

import torch
from torch import Tensor, long, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from input.input_transforms import TXT_ENCODING, line_to_tensor


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
        """Iterate over the dataset in a round-robin fashion, yielding (label_tensor, data_tensor) tuples.

        This method processes all files simultaneously in a round-robin manner to ensure that
        consecutive batches contain entries from different files when possible.

        :yield: A tuple containing the label tensor and data tensor for each sample.
        """
        text_files = self._get_text_files(self.data_dir)
        file_handles = self._open_files(text_files)

        try:
            yield from self._round_robin_iteration(file_handles)
        finally:
            # Ensure all file handles are properly closed
            for file_handle, _, _ in file_handles:
                file_handle.close()

    def _round_robin_iteration(self, file_handles: list[tuple[TextIO, str, int]]) -> Iterator[tuple[Tensor, Tensor]]:
        """Perform round-robin iteration over file handles, yielding processed lines."""
        # Keep track of active files (files that still have lines to read)
        active_files = list(range(len(file_handles)))

        while active_files:
            files_to_remove = []

            # Process one line from each active file in round-robin fashion
            for file_idx in active_files:
                file_handle, label, label_idx = file_handles[file_idx]

                try:
                    line = next(file_handle)
                    result = self._process_line_from_file(line, label_idx)
                    if result is not None:
                        yield result
                except StopIteration:
                    # File is exhausted, mark it for removal
                    files_to_remove.append(file_idx)

            # Remove exhausted files from an active list
            for file_idx in reversed(files_to_remove):
                active_files.remove(file_idx)

    def _process_line_from_file(self, line: str, label_idx: int) -> tuple[Tensor, Tensor] | None:
        """Process a single line from a file and return tensors, or None if line should be skipped."""
        keyword = line.strip()

        if not keyword:
            # Skip empty lines
            return None

        if self.transforms:
            keyword = self.transforms(keyword)

        label_tensor = tensor(label_idx, dtype=long)
        data_tensor = line_to_tensor(keyword)
        return label_tensor, data_tensor

    def _open_files(self, text_files: list[Path]) -> list[tuple[TextIO, str, int]]:
        file_handles: list[tuple[TextIO, str, int]] = []
        for filename in text_files:
            label = filename.stem
            label_idx = self.labels_uniq.index(label)
            file_handle = filename.open(encoding=TXT_ENCODING)
            file_handles.append((file_handle, label, label_idx))
        return file_handles

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
