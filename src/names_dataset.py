from pathlib import Path

from torch import Tensor, long, tensor
from torch.utils.data import Dataset

from input_transforms import line_to_tensor


class NamesDataset(Dataset):
    """Represents a dataset of names and their corresponding labels derived from text files.

    This class is designed to load and preprocess text data from a specified directory,
    where each text file corresponds to a unique label. Each file contains a list of
    names. The class processes these names into tensors suitable for machine learning
    applications while also associating them with their labels.
    """

    def __init__(self, data_dir: str) -> None:
        """Create a new NamesDataset instance.

        :param data_dir: The path to the directory containing text files. Each text file should
            have a name that represents the label and contain lines of text data.
        """
        text_files = self._get_text_files(data_dir)
        self._create_labels_from_filenames(text_files)
        self._create_tensors(text_files)

    def _create_labels_from_filenames(self, text_files: list[Path]) -> None:
        labels_set = set()
        for filename in text_files:
            label = filename.stem
            labels_set.add(label)
        self.labels_uniq = list(labels_set)

    def _create_tensors(self, text_files: list[Path]) -> None:
        self.data_tensors: list[Tensor] = []
        self.labels_tensors: list[Tensor] = []
        for filename in text_files:
            label = filename.stem
            label_idx = self.labels_uniq.index(label)
            with filename.open(encoding="utf-8") as file:
                lines = file.read().strip().split("\n")
            for name in lines:
                self.data_tensors.append(line_to_tensor(name))
                self.labels_tensors.append(tensor([label_idx], dtype=long))

    def __len__(self) -> int:
        """Calculate and return the number of elements in the data structure.

        :return: The number of elements in the data structure.
        """
        return len(self.data_tensors)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Retrieve a specific entry from the dataset using the given index.

        :param idx: Index of the dataset entry to retrieve.
        :return: A tuple containing the label tensor and data tensor at the specified index.
        """
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]
        return label_tensor, data_tensor

    @staticmethod
    def _get_text_files(data_dir: str) -> list[Path]:
        return list(Path(data_dir).glob("*.txt"))
