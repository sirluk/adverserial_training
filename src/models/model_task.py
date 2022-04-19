import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict

from src.models.model_base import BaseModel
from src.models.model_heads import ClfHead
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class TaskModel(BaseModel):
        
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        **kwargs
    ): 
        super().__init__(model_name, **kwargs)
        
        self.num_labels = num_labels
        self.task_head = ClfHead(self.out_size, num_labels)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        learning_rate: float,
        warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        
        self.global_step = 0
        train_steps = len(train_loader) * num_epochs
        
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            warmup_steps
        )
        
        self.zero_grad()
        
        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs, desc=train_str.format(0, {}), leave=False, position=0)
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
                metrics
            )
            
            logger.validation_loss(epoch, result, "task")
            
            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )
            
            if logger.is_best(result):
                cpt = self.save_checkpoint(Path(output_dir))
        
        print("Final results after " + train_str.format(epoch, str_suffix(result)))
        return cpt
        
        
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable]
    ) -> dict: 
        self.eval()   

        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            logits = self(**inputs)    
            output_list.append((
                logits.cpu(),
                labels
            ))
                    
        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)
        
        eval_loss = loss_fn(predictions, labels).item()
        
        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result            
            
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
                
            logger.step_loss(self.global_step, loss, self.scheduler.get_last_lr()[0])
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            
            self.global_step += 1
 
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        num_warmup_steps: int = 0
    ) -> None:
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict()
        }          

        filename = f"{self.model_name.split('/')[-1]}-task.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath
        
        
    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], map_location=torch.device('cpu')) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)
    
        task_network = TaskModel(
            info_dict['model_name'],
            info_dict['num_labels']
        )
        task_network.encoder.load_state_dict(info_dict['encoder_state_dict'])
        task_network.task_head.load_state_dict(info_dict['task_head_state_dict'])
        
        task_network.eval()
        
        return task_network
