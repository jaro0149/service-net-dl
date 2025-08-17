from torch import Tensor


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
