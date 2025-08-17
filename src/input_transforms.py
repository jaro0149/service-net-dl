import string

import torch
from torch import Tensor

ALLOWED_CHARS: str = string.ascii_letters + " .,;'" + "_"
N_LETTERS: int = len(ALLOWED_CHARS)


def line_to_tensor(line: str) -> Tensor:
    """Convert a given string into a tensor representation with one-hot encoding.

    Each letter in the input string is represented as a one-hot encoded vector at its respective position in
    the resulting tensor.

    :param line: The input string to be converted into a tensor.
    :return: A tensor with shape (len(line), 1, N_LETTERS), where each letter in the string
        is one-hot encoded.
    """
    output_tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        output_tensor[li][0][_letter_to_index(letter)] = 1
    return output_tensor


def _letter_to_index(letter: str) -> int:
    """Convert a single character to its corresponding index value based on the `ALLOWED_CHARS` set.

    If the character is not present in the `ALLOWED_CHARS`, it maps the character to the index of
    the underscore symbol `_`.

    :param letter: A single character to be converted to its index.
    :return: The index of the character in the `ALLOWED_CHARS` or the index of `_` if the character is not found.
    """
    if letter not in ALLOWED_CHARS:
        return ALLOWED_CHARS.find("_")
    return ALLOWED_CHARS.find(letter)
