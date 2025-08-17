import torch
from torch.nn import Module
from torch.utils.data import Subset

from src.output_transforms import label_idx_from_output
from src.torch_utils import IterableSubset


class Forecaster:
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

    def forecast_on_testing_data(
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
            for (_, text_tensor, label, _) in IterableSubset(testing_data):
                output = self.rnn(text_tensor)
                guess_idx = label_idx_from_output(output)
                label_idx = self.classes.index(label)

                all_forecasts.append(guess_idx)
                all_targets.append(label_idx)

        return all_forecasts, all_targets
