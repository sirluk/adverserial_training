import os
import torch
from transformers import AutoModel

from typing import Union


class BaseModel(torch.nn.Module):

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def model_type(self) -> str:
        return self.encoder_module.config.model_type

    @property
    def model_name(self) -> str:
        return self.encoder_module.config._name_or_path

    @property
    def hidden_size(self) -> int:
        return self.encoder_module.embeddings.word_embeddings.embedding_dim

    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        cfg = self.encoder_module.config
        for k in possible_keys:
            if k in cfg.__dict__:
                return getattr(cfg, k) + 1 # +1 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)

    def _forward(self, **x) -> torch.Tensor:
        return self.encoder(**x)[0][:,0]

    @torch.no_grad()
    def _evaluate(
        self,
        val_loader: DataLoader,
        forward_fn: Callable,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 1,
        desc: str = "",
        **kwargs
        ) -> dict:

        self.eval()

        eval_loss = 0.
        output_list = []
        val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch[0], batch[label_idx]
            if isinstance(inputs, dict):
                inputs = dict_to_device(inputs, self.device)
            else:
                inputs = inputs.to(self.device)
            logits = forward_fn(inputs, **kwargs)
            if isinstance(logits, list):
                eval_loss += torch.stack([loss_fn(x.cpu(), labels) for x in logits]).mean().item()
                preds, _ = torch.mode(torch.stack([pred_fn(x.cpu()) for x in logits]), dim=0)
            else:
                eval_loss += loss_fn(logits.cpu(), labels).item()
                preds = pred_fn(logits.cpu())
            output_list.append((
                preds,
                labels
            ))

        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)

        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss / (i+1)

        return result

    def _get_mean_loss(self, outputs: Union[torch.Tensor, List[torch.Tensor]], labels: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        losses = []
        for output in outputs:
            losses.append(loss_fn(output, labels))
        return torch.stack(losses).mean()        

    def forward(self, **x) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def _step(self, *args, **kwargs):
        raise NotImplementedError

    def _init_optimizer_and_schedule(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, output_dir: Union[str, os.PathLike], *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, os.PathLike], *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError