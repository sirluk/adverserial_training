# import IPython; IPython.embed(); exit(1)

import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_handler import get_data_loader, get_data_loader_norec 
from src.model import DiffNetwork
from src.model_task_diff import DiffNetworkTask
from src.model_adverserial import AdverserialNetwork
from src.model_task import TaskNetwork
from src.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.utils import get_metrics, get_loss_fn, get_num_labels
from src.adv_attack import adv_attack

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(torch.nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.base_encoder = trainer.encoder
        self.bottleneck = trainer.bottleneck
        
        self.emb_dim = trainer.embeddings.word_embeddings.embedding_dim
        self.bottleneck_dim = trainer.bottlenck_dim

    def forward(self, **x):
        x_ = self.base_encoder(**x)[0][:,0]
        return self.bottleneck(x_)


def get_data(base_args, args_train):
    Path(args_train.output_dir).mkdir(parents=True, exist_ok=True)

    num_labels = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)

    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    train_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args_train.train_pkl, 
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=args_train.labels_protected_path,
        batch_size=args_train.batch_size, 
        max_length=200,
        raw=base_args.raw,
        debug=base_args.debug
    )
    
    eval_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args_train.val_pkl, 
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=args_train.labels_protected_path,
        batch_size=args_train.batch_size, 
        max_length=200,
        raw=base_args.raw,
        shuffle=False,
        debug=base_args.debug
    )
    
    return train_loader, eval_loader, num_labels, num_labels_protected


def get_logger(base_name, args_train):
    logger_name = "_".join([
        base_name,
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        str(args_train.adv_rev_ratio)
    ])
    train_logger = TrainLogger(
        log_dir = Path("logs"),
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )
    return train_logger


# def run_adv_attack(trainer, train_loader, eval_loader, train_logger, num_labels_protected, args_attack):



# #     encoder = Encoder(trainer.encoder, trainer.bottleneck)

#     adv_head = AdvHead(
#         args_attack.adv_count,
#         hid_sizes=[trainer.encoder_out_dim, trainer.encoder.embeddings.word_embeddings.embedding_dim],
#         num_labels=num_labels_protected,
#         dropout_prob=args_attack.adv_dropout
#     )
#     adv_head.to(DEVICE)   

#     adv_attack(
#         encoder = encoder,
#         adv_head = adv_head,
#         train_loader = train_loader,
#         val_loader = eval_loader,
#         logger = train_logger,
#         loss_fn = get_loss_fn(num_labels_protected),
#         metrics = get_metrics(num_labels_protected),
#         num_epochs = args_attack.num_epochs,
#         lr = args_attack.learning_rate
#     )

def main_diff_pruning(base_args, args_train, args_attack):    
    
    train_loader, eval_loader, num_labels, num_labels_protected = get_data(base_args, args_train)

    method = "adverserial_diff_pruning" if args_train.adverserial else "task_diff_pruning"
    
    train_logger = get_logger(method, args_train)

    if args_train.adverserial:   
        trainer = DiffNetwork(
            args_train.model_name,
            num_labels,
            num_labels_protected,
            adv_count=args_train.adv_count,
            adv_rev_ratio=args_train.adv_rev_ratio
        )
        trainer.to(DEVICE)
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            get_loss_fn(num_labels_protected),
            get_metrics(num_labels_protected),
            args_train.num_epochs_finetune,
            args_train.num_epochs_fixmask,
            args_train.alpha_init,
            args_train.concrete_lower,
            args_train.concrete_upper,
            args_train.structured_diff_pruning,
            args_train.sparsity_pen,
            args_train.fixmask_pct,
            args_train.learning_rate,
            args_train.learning_rate_alpha,
            args_train.learning_rate_adverserial,
            args_train.optimizer_warmup_steps,
            args_train.weight_decay,
            args_train.adam_epsilon,
            args_train.max_grad_norm,
            args_train.output_dir
        )
    else:
        trainer = DiffNetworkTask(
            args_train.model_name,
            num_labels
        )   
        trainer.to(DEVICE)
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            args_train.num_epochs_finetune,
            args_train.num_epochs_fixmask,
            args_train.alpha_init,
            args_train.concrete_lower,
            args_train.concrete_upper,
            args_train.structured_diff_pruning,
            args_train.sparsity_pen,
            args_train.fixmask_pct,
            args_train.weight_decay,
            args_train.learning_rate,
            args_train.learning_rate_alpha,
            args_train.adam_epsilon,
            args_train.optimizer_warmup_steps,
            args_train.max_grad_norm,
            args_train.output_dir
        )
        
    run_adv_attack(trainer, train_loader, eval_loader, train_logger, num_labels_protected, args_attack)


def main_baseline(base_args, args_train, args_attack):    
    
    train_loader, eval_loader, num_labels, num_labels_protected = get_data(base_args, args_train)
    
    method = "adverserial" if args_train.adverserial else "task"
    if args_train.bitfit:
        method = method + "_bitfit"
    else:
        method = method + "_baseline"
    
    train_logger = get_logger(method, args_train)

    if args_train.adverserial:   
        trainer = AdverserialNetwork(
            args_train.model_name,
            num_labels,
            num_labels_protected,
            adv_count=args_train.adv_count
        )  
        trainer.to(DEVICE)
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            get_loss_fn(num_labels_protected),
            get_metrics(num_labels_protected),
            args_train.num_epochs,
            args_train.num_epochs_warmup,
            args_train.learning_rate,
            args_train.learning_rate_adverserial,
            args_train.warmup_steps,
            args_train.max_grad_norm,
            args_train.output_dir,
            args_train.bitfit
        )
    else:
        trainer = TaskNetwork(
            args_train.model_name,
            num_labels
        ) 
        trainer.to(DEVICE)
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            args_train.num_epochs,
            args_train.learning_rate,
            args_train.warmup_steps,
            args_train.max_grad_norm,
            args_train.output_dir,
            args_train.bitfit
        )
        
    # run_adv_attack(trainer, train_loader, eval_loader, train_logger, num_labels_protected, args_attack)
    
    #encoder = Encoder(trainer.encoder, trainer.bottleneck)
    
    adv_attack(
        trainer = trainer,
        train_loader = train_loader,
        val_loader = eval_loader,
        logger = train_logger,
        loss_fn = get_loss_fn(num_labels_protected),
        metrics = get_metrics(num_labels_protected),
        num_labels = num_labels_protected,
        adv_count = args_attack.adv_count,
        adv_dropout = args_attack.adv_dropout,        
        num_epochs = args_attack.num_epochs,
        lr = args_attack.learning_rate
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--baseline", type=bool, default=False, help="")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    train_cfg = "train_config_baseline" if base_args.baseline else "train_config_diff_pruning"
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)    
    args_train = argparse.Namespace(**{**cfg[train_cfg], **cfg["data_config"], **cfg["model_config"]})
    args_attack = argparse.Namespace(**cfg["adv_attack"])
    
    print(f"Running {'baseline' if base_args.baseline else 'diff-pruning'}{' (debug)' if base_args.debug else ''}")
    
    if base_args.baseline:
        main_baseline(base_args, args_train, args_attack)
    else:
        main_diff_pruning(base_args, args_train, args_attack)


