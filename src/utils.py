import torch

from typing import Union, Callable

from src.metrics import accuracy
from src.data_handler import read_label_file 


def dict_to_device(d: dict, device: Union[torch.device, str]) -> dict:
    return {k:v.to(device) for k,v in d.items()}


def get_metrics(num_labels: int) -> dict:
    if num_labels == 1:
        pred_fn = lambda x: (x > 0).long()
    else:
        pred_fn = lambda x: torch.argmax(x, dim=1)

    return {
        "acc": lambda x, y: accuracy(pred_fn(x), y),
        "balanced_acc": lambda x, y: accuracy(pred_fn(x), y, balanced=True)
    }


def get_loss_fn(num_labels: int) -> Callable:
    if num_labels == 1:
        return lambda x, y: torch.nn.BCEWithLogitsLoss()(x.flatten(), y.float())
    else:
        return torch.nn.CrossEntropyLoss()


def get_num_labels(label_file: int) -> int:
    num_labels = len(read_label_file(label_file))
    return 1 if num_labels==2 else num_labels
