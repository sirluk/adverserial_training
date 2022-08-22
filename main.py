import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.adv_attack import adv_attack
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_logger,
    get_callables
)


def train_task(device, train_loader, val_loader, num_labels, train_logger, args_train, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    trainer = TaskModel(
        model_name = args_train.model_name,
        num_labels = num_labels,
        dropout = args_train.task_dropout,
        n_hidden = args_train.task_n_hidden
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_head = args_train.learning_rate_task_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        seed = seed
    )
    trainer = TaskModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def train_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train, seed = None):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = AdvModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected,
        pred_fn_protected = pred_fn_protected,
        metrics_protected = metrics_protected,
        num_epochs = args_train.num_epochs,
        num_epochs_warmup = args_train.num_epochs_warmup,
        adv_lambda = args_train.adv_lambda,
        learning_rate = args_train.learning_rate,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir,
        seed = seed
    )
    trainer = AdvModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--adv", action="store_true", help="Whether to run adverserial training")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--no_adv_attack", action="store_true", help="Set if you do not want to run adverserial attack after training")
    base_args = parser.parse_args()

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)

    args_train = argparse.Namespace(**cfg["train_config"], **cfg["data_config"], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    if base_args.debug:
        args_train = set_num_epochs_debug(args_train)
        args_attack = set_num_epochs_debug(args_attack)
        args_train = set_dir_debug(args_train)

    device = get_device(not base_args.cpu, base_args.gpu_id)

    train_loader, val_loader, num_labels, num_labels_protected = get_data(args_train, debug=base_args.debug)
    
    train_logger = get_logger(args_train, base_args.adv, base_args.debug, base_args.seed)

    print(f"Running {train_logger.logger_name}")

    if base_args.adv:
        trainer = train_adv(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train, base_args.seed)
    else:
        trainer = train_task(device, train_loader, val_loader, num_labels, train_logger, args_train, base_args.seed)

    if not base_args.no_adv_attack:
        loss_fn, pred_fn, metrics = get_callables(num_labels_protected)
        adv_attack(
            trainer = trainer,
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_protected,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            batch_size = args_attack.attack_batch_size
        )


if __name__ == "__main__":

    main()

