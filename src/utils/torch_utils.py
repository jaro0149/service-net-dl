import io
import logging
import random
import sys
from collections.abc import Iterator

import numpy as np
import torch
from torch import device
from torch.nn import Module
from torch.utils.data.dataset import Subset, _T_co
from torchinfo import summary

logger = logging.getLogger(__name__)


class IterableSubset(Subset[_T_co]):
    """A wrapper for a data subset, enabling iterable behavior."""

    def __init__(self, subset: Subset[_T_co], batch_indices: np.ndarray | None = None) -> None:
        """Initialize an instance of the class with a subset and optionally specified batch indices.

        The initializer allows defining a subset of data and an array of batch indices
        to specify the selected data within that subset. If no batch indices
        are provided, the initializer defaults to using the entire indices
        of the subset.

        :param subset: The subset of data to be considered.
        :param batch_indices: Optional array specifying the indices of batches
                              within the subset. Defaults to None, in which case
                              the subset's indices are used directly.
        """
        if batch_indices is None:
            super().__init__(subset.dataset, subset.indices)
        else:
            super().__init__(subset.dataset, batch_indices.tolist())

    def __iter__(self) -> Iterator[_T_co]:
        """Iterate over the elements of the dataset using specified indices.

        :return: An iterator that yields elements of type `_T_co`.
        """
        for idx in self.indices:
            yield self.dataset[idx]


def get_device() -> device:
    """Determine the optimal device for computation based on hardware availability.

    This function checks the availability of different devices in a priority order:
    CUDA/ROCM, XPU, MPS, and finally falls back to CPU if none of the specialized hardware
    is available.

    :return: The most suitable `torch.device` object for computations.
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda:0")
    if torch.xpu.is_available():
        logger.info("Using XPU device: %s", torch.xpu.get_device_name(0))
        return torch.device("xpu:0")
    if torch.mps.is_available():
        logger.info("Using MPS device")
        return torch.device("mps")
    logger.info("Using CPU")
    return torch.device("cpu")


def log_model_info(
        rnn: Module,
        input_size: tuple[int, ...],
) -> None:
    """Log the detailed summary of the provided PyTorch model.

    :param rnn: The PyTorch model whose summary needs to be logged
    :param input_size: A tuple defining the size of the input tensor for the model
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        summary(rnn, input_size=input_size)
        summary_text = captured_output.getvalue()
        logger.info("Model summary:\n%s", summary_text)
    finally:
        sys.stdout = old_stdout


def set_seed(seed: int) -> None:
    """Set the seed for random number generation in PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
