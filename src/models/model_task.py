import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead
from src.models.model_base import BaseModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class TaskModel(BaseModel):

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = .3,
        n_hidden: int = 0,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.num_labels = num_labels
        self.dropout = dropout
        self.n_hidden = n_hidden

        self.task_head = ClfHead([self.hidden_size]*(n_hidden+1), num_labels, dropout=dropout)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        learning_rate: float,
        learning_rate_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0

        train_steps = len(train_loader) * num_epochs
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_head,
            optimizer_warmup_steps
        )

        self.zero_grad()

        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])

        train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:

            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )

            logger.validation_loss(epoch, result, "task")

            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )

            cpt = self.save_checkpoint(Path(output_dir), seed)

        print("Final result after " + train_str.format(epoch, str_suffix(result)))
        
        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable]
    ) -> dict:
        self.eval()

        forward_fn = lambda x: self(**x)

        return self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics)

    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.zero_grad()

            logger.step_loss(self.global_step, loss.item())

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_head: float,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_head
            }
        ]
        self.optimizer = AdamW(optimizer_params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        seed: Optional[int] = None
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout,
            "n_hidden": self.n_hidden,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict()
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        seed_str = f"-seed{seed}" if seed is not None else ""
        filename = f"{self.model_name.split('/')[-1]}-task_baseline{seed_str}.pt"
        filepath = output_dir / filename
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels'],
            info_dict['dropout'],
            info_dict['n_hidden']
        )
        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.eval()

        return cls_instance
