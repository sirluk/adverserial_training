import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BaseModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class AdvModel(BaseModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.task_dropout = task_dropout
        self.task_n_hidden = task_n_hidden
        self.adv_dropout = adv_dropout
        self.adv_n_hidden = adv_n_hidden
        self.adv_count = adv_count

        # heads
        self.task_head = ClfHead([self.hidden_size]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        self.adv_head = AdvHead(adv_count, hid_sizes=[self.hidden_size]*(adv_n_hidden+1), num_labels=num_labels_protected, dropout=adv_dropout)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def forward_protected(self, **x) -> torch.Tensor:
        return self.adv_head(self._forward(**x)) 

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        pred_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs: int,
        num_epochs_warmup: int,
        adv_lambda: float,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total

        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            optimizer_warmup_steps
        )

        self.zero_grad()

        train_str = "Epoch {}, {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        train_iterator = trange(num_epochs_total, desc=train_str.format(0, "", ""), leave=False, position=0)
        for epoch in train_iterator:
            if epoch<num_epochs_warmup:
                _adv_lambda = 0.
            else:
                _adv_lambda = adv_lambda

            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
                loss_fn_protected,
                _adv_lambda
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )
            logger.validation_loss(epoch, result, suffix="task")
            
            result_protected = self.evaluate(
                val_loader,
                loss_fn_protected,
                pred_fn_protected,
                metrics_protected,
                predict_prot=True
            )
            logger.validation_loss(epoch, result_protected, suffix="protected")

            result_str = train_str.format(
                epoch, str_suffix(result, "_task"), str_suffix(result_protected, "_protected")
            )

            train_iterator.set_description(result_str, refresh=True)

            cpt = self.save_checkpoint(Path(output_dir), seed)

        print("Final result after " + result_str)

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        predict_prot: bool = False
    ) -> dict:
        self.eval()

        if predict_prot:
            desc = "protected attribute"
            label_idx = 2
            forward_fn = lambda x: self.forward_protected(**x)
        else:
            desc = "task"
            label_idx = 1
            forward_fn = lambda x: self(**x)

        return self._evaluate(val_loader, forward_fn, loss_fn, pred_fn, metrics, label_idx, desc)


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float,
        loss_fn_protected: Callable,
        adv_lambda: float
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            loss = 0.

            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)
            hidden = self._forward(**inputs)
            outputs_task = self.task_head(hidden)
            loss_task = loss_fn(outputs_task, labels_task.to(self.device))
            loss += loss_task

            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda = adv_lambda)
            loss_protected = self._get_mean_loss(outputs_protected, labels_protected.to(self.device), loss_fn_protected)
            loss += loss_protected

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.zero_grad()

            losses_dict = {
                "total": loss.item(),
                "task": loss_task.item(),
                "protected": loss_protected.item()
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        num_warmup_steps: int = 0
    ) -> None:

        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_task_head
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adv_head
            }
        ]

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)

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
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "task_dropout": self.task_dropout,
            "task_n_hidden": self.task_n_hidden,
            "adv_dropout": self.adv_dropout,
            "adv_n_hidden": self.adv_n_hidden,
            "adv_count": self.adv_count,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_str = f"-seed{seed}" if seed is not None else ""
        filename = f"{self.model_name.split('/')[-1]}-adv_baseline{seed_str}.pt"
        filepath = output_dir / filename
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], map_location: Union[str, torch.device] = torch.device('cpu')) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['task_dropout'],
            info_dict['task_n_hidden'],
            info_dict['adv_dropout'],
            info_dict['adv_n_hidden'],
            info_dict['adv_count']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.eval()

        return cls_instance
