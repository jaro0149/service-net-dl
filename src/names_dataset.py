from pathlib import Path

from torch import Tensor, long, tensor
from torch.utils.data import Dataset

from src.input_transforms import line_to_tensor


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
        self.data_dir = data_dir

        self.data: list[str] = []
        self.data_tensors: list[Tensor] = []
        self.labels: list[str] = []
        self.labels_tensors: list[Tensor] = []

        labels_set = set()
        text_files = Path(data_dir).glob("*.txt")
        for filename in text_files:
            label = filename.stem
            labels_set.add(label)
            with filename.open(encoding="utf-8") as file:
                lines = file.read().strip().split("\n")
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(line_to_tensor(name))
                self.labels.append(label)

        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = tensor([self.labels_uniq.index(self.labels[idx])], dtype=long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self) -> int:
        """Calculate and return the number of elements in the data structure.

        :return: The number of elements in the data structure.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str, str]:
        """Retrieve a specific entry from the dataset using the given index.

        :param idx: Index of the dataset entry to retrieve.
        :return: A tuple containing the label tensor, data tensor, label string,
            and data string at the specified index.
        """
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item
