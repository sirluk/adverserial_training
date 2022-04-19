import math
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.utils import dict_to_device

from typing import Callable, Dict, Union


@torch.no_grad()
def evaulate_adv_attack(
    trainer: nn.Module,
    adv_head: nn.Module,
    val_loader: DataLoader,
    loss_fn: Callable,
    metrics: Dict[str, Callable],
    device: Union[str, torch.device]
):
    encoder.eval()
    adv_head.eval()

    output_list = []
    val_iterator = tqdm(val_loader, desc=f"evaluating", leave=False, position=1)
    for i, batch in enumerate(val_iterator):

        inputs, labels = batch[0], batch[2]
        inputs = dict_to_device(inputs, device)
        hidden = trainer._forward(**inputs)
        outputs = adv_head(hidden)
        if isinstance(outputs, list):
            labels = labels.repeat(len(outputs))
            outputs = torch.cat(outputs, dim=0)         
        output_list.append((
            outputs.cpu(),
            labels
        ))

    p, l = list(zip(*output_list))
    predictions = torch.cat(p, dim=0)
    labels = torch.cat(l, dim=0)

    eval_loss = loss_fn(predictions, labels).item()

    result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
    result["loss"] = eval_loss
    
    return result


def adv_attack(
    trainer: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainLogger,
    loss_fn: Callable,
    metrics: Dict[str, Callable],
    num_labels: int,
    adv_count: int,
    adv_dropout: int,
    num_epochs: int,
    lr: float
):   
    device = trainer.device
    
    adv_head = AdvHead(
        adv_count,
        hid_sizes=trainer.out_size,
        num_labels=num_labels,
        dropout=bool(adv_dropout),
        dropout_prob=adv_dropout
    )
    adv_head.to(device)
    
    optimizer = AdamW(adv_head.parameters(), lr=lr)
    
    global_step = 0
    train_str = "Epoch {}, {}"
    result_str = lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()])
    train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
    for epoch in train_iterator:
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            trainer.train()
            adv_head.train()
            
            inputs, labels = batch[0], batch[2]
            inputs = dict_to_device(inputs, device)

            with torch.no_grad():
                hidden = trainer._forward(**inputs)
            outputs = adv_head(hidden)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
                
            loss = 0.
            for output in outputs:
                loss += loss_fn(output, labels.to(device)) / len(outputs)
            
            loss.backward()
            optimizer.step()  
            adv_head.zero_grad()
            
            logger.step_loss(global_step, loss.item(), lr, "adv_attack")
            
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            
        result = evaulate_adv_attack(trainer, adv_head, val_loader, loss_fn, metrics, device)
        logger.validation_loss(epoch, result, "adv_attack")
        
        train_iterator.set_description(
            train_str.format(epoch, result_str(result)), refresh=True
        )
               
    print("Final results after " +  train_str.format(epoch, result_str(result)))
