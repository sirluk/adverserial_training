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
        return self.encoder.config.model_type
    
    @property
    def model_name(self) -> str:
        return self.encoder.config._name_or_path
    
    @property
    def out_size(self) -> int:
        return self.encoder.embeddings.word_embeddings.embedding_dim
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
                
    def _forward(self, **x) -> torch.Tensor:
        return self.encoder(**x)[0][:,0]

    def forward(self, **x) -> torch.Tensor:
        raise NotImplementedError
        
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
        
    def _step(self, *args, **kwargs):
        raise NotImplementedError
        
    def _init_optimizer_and_schedule(self, *args, **kwargs):
        raise NotImplementedError
        
    def save_checkpoint(self, output_dir: Union[str, os.PathLike]) -> None:
        raise NotImplementedError
     
    @staticmethod
    def load_checkpoint(filepath: Union[str, os.PathLike], map_location=torch.device('cpu')) -> torch.nn.Module:
        raise NotImplementedError