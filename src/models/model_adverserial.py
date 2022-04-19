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
from src.models.model_heads import ClfHead, AdvHead
from src.training_logger import TrainLogger
from src.utils import dict_to_device


class AdverserialModel(BaseModel):
    
    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: int,
        adv_count: int,
        **kwargs
    ): 
        super().__init__(model_name, **kwargs)
        
        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.adv_count = adv_count
        
        self.task_head = ClfHead(self.out_size, num_labels_task)
        self.adv_head = AdvHead(adv_count, hid_sizes=self.out_size, num_labels=num_labels_protected)
            
    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Callable,
        metrics_protected: Dict[str, Callable],
        num_epochs: int,
        num_epochs_warmup: int,
        learning_rate: float,
        learning_rate_adverserial: float,
        warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        
        self.global_step = 0
        num_epochs_warmup = max(0,num_epochs_warmup) # ensure values is >= 0
        num_epochs_total = num_epochs + num_epochs_warmup
        train_steps = len(train_loader) * num_epochs_total
        
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_adverserial,
            warmup_steps
        )
        
        self.zero_grad()
        
        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])
        
        train_iterator = trange(num_epochs_total, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:
            warmup = (epoch<num_epochs_warmup)
               
            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm,
                loss_fn_protected,
                warmup=warmup
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            logger.validation_loss(epoch, result, "task")
            
            if warmup:
                result_protected = {k:None for k in result.keys()}
            else:
                result_protected = self.evaluate(
                    val_loader, 
                    loss_fn_protected,
                    metrics_protected,
                    predict_prot=True
                ) 
                logger.validation_loss(epoch, result_protected, suffix="protected")
            
            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result_protected)), refresh=True
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
        metrics: Dict[str, Callable],
        predict_prot: bool = False
    ) -> dict: 
        self.eval()

        if predict_prot:
            desc = "protected attribute"
            label_idx = 2
            forward_fn = lambda x: self.adv_head(self._forward(**x))
        else:
            desc = "task"
            label_idx = 1
            forward_fn = lambda x: self(**x)        

        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)

        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[label_idx]
            inputs = dict_to_device(inputs, self.device)
            logits = forward_fn(inputs)
            if isinstance(logits, list):
                labels = labels.repeat(len(logits))
                logits = torch.cat(logits, dim=0)         
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
        max_grad_norm: float,
        loss_fn_protected: Callable,
        warmup: bool
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels_task, labels_protected = batch
            inputs = dict_to_device(inputs, self.device)
            hidden = self._forward(**inputs)   
            outputs_task = self.task_head(hidden)
            loss = loss_fn(outputs_task, labels_task.to(self.device))
            
            outputs_protected = self.adv_head.forward_reverse(hidden, lmbda=int(not warmup))
            if isinstance(outputs_protected, torch.Tensor):
                outputs_protected = [outputs_protected]
            losses_protected = []
            for output in outputs_protected:
                losses_protected.append(loss_fn_protected(output, labels_protected.to(self.device)))
            loss_protected = torch.stack(losses_protected).mean()
            loss += loss_protected
            loss.backward()
     

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()          
            self.zero_grad()    
                
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"] # self.scheduler.get_last_lr()[0]
            logger.step_loss(self.global_step, loss, lr) 
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            
            self.global_step += 1
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_adverserial: float,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "lr": learning_rate
            },         
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adverserial
            }
        ]

        self.optimizer = AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-08)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        
    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        info_dict = {
            "model_name": self.model_name,
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "adv_count": self.adv_count,
            "encoder_state_dict": self.encoder.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict()
        }          

        filename = f"{self.model_name.split('/')[-1]}-adverserial.pt"
        filepath = Path(output_dir) / filename
        torch.save(info_dict, filepath)
        return filepath
        
        
    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], map_location=torch.device('cpu')) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)
    
        adv_network = AdverserialModel(
            info_dict['model_name'],
            info_dict['num_labels_task'],
            info_dict['num_labels_protected'],
            info_dict['adv_count']
        )
        adv_network.encoder.load_state_dict(info_dict['encoder_state_dict'])
        adv_network.adv_head.load_state_dict(info_dict['adv_head_state_dict'])
        adv_network.task_head.load_state_dict(info_dict['task_head_state_dict'])
        
        adv_network.eval()
        
        return adv_network
