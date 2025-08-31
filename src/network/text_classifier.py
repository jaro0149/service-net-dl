import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Subset

from utils.torch_utils import IterableSubset


class TextClassifier:
    """A forecasting utility that uses RNN to predict class indices based on textual data."""

    def __init__(
            self,
            rnn: Module,
            classes: list[str],
    ) -> None:
        """Initialize the class with a recurrent neural network (RNN) module and a list of class labels.

        :param rnn: A trained RNN-based `torch.nn.Module` model for making predictions.
        :param classes: A list of strings representing the class labels.
        """
        self.rnn = rnn
        self.classes = classes

    def classify_testing_data(
            self,
            testing_data: Subset,
    ) -> tuple[list[int], list[int]]:
        """Generate forecasts and corresponding target indices by running a given model on the provided testing dataset.

        This function iterates over the testing data
        subset, processes each data sample, and predicts class indices corresponding
        to the forecasting results. The predicted forecasts and actual target indices
        are compiled into separate lists and returned as output.

        :param testing_data: A subset of testing data containing labeled text samples to
            be evaluated against the model.

        :return: A tuple containing two lists:
            - The first list includes predicted class indices (forecasts).
            - The second list contains actual target class indices from the testing data.
        """
        all_forecasts: list[int] = []
        all_targets: list[int] = []

        self.rnn.eval()
        with torch.no_grad():
            for (label_tensor, text_tensor) in IterableSubset(testing_data):
                output = self.rnn(text_tensor)
                guess_idx = self.label_idx_from_output(output)
                label_idx = label_tensor.item()

                all_forecasts.append(guess_idx)
                all_targets.append(label_idx)

        return all_forecasts, all_targets

    @staticmethod
    def label_idx_from_output(output: Tensor) -> int:
        """Convert the output of a model to the index of the label with the highest score.

        :param output: Tensor containing the model's output scores for all labels
        :return: Index of the label with the highest score from the output tensor
        """
        _, top_i = output.topk(1)
        idx = top_i[0].item()
        if not isinstance(idx, int):
            msg = f"Expected int, but got {type(idx)}"
            raise TypeError(msg)
        return idx
