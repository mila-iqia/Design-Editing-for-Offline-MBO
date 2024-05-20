import json
import os

from pprint import pprint

import configargparse

from contextlib import contextmanager, redirect_stderr, redirect_stdout


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    import design_bench
    from register_dataset.register_levy import LevyDataset, LevyOracle
    design_bench.register('Levy-Exact-v0', LevyDataset, LevyOracle,
                          # keyword arguments for building the dataset
                          dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                          # keyword arguments for building the exact oracle
                          oracle_kwargs=dict(noise_std=0.0)
                          )


import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

from nets import DiffusionTest, DiffusionScore
from util import TASKNAME2TASK, configure_gpu, set_seed, get_weights

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "sde-flow"


class RvSDataset(Dataset):

    def __init__(self, task, x, y, w=None, device=None, mode='train'):
        self.task = task
        self.device = device
        self.mode = mode
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        if self.w is not None:
            w = torch.tensor(self.w[idx])
        else:
            w = None
        '''
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
            if w is not None:
                w = w.to(self.device)
        '''
        if w is None:
            return x, y
        else:
            return x, y, w


def temp_get_super_y(task):
    y = task.y.reshape(-1)
    sorted_y_idx = np.argsort(y)

    super_task = design_bench.make(TASKNAME2TASK["superconductor"])
    super_task.map_normalize_y()

    super_y = super_task.y.reshape(-1)
    super_y = np.sort(super_y)
    super_y = super_y[sorted_y_idx]

    return super_y


def split_dataset(task, val_frac=None, device=None, temp=None):
    length = task.y.shape[0]
    shuffle_idx = np.arange(length)
    shuffle_idx = np.random.shuffle(shuffle_idx)

    if task.is_discrete:
        task.map_to_logits()
        x = task.x[shuffle_idx]
        x = x.reshape(x.shape[1:])
        x = x.reshape(x.shape[0], -1)
    else:
        x = task.x[shuffle_idx]

    # y = temp_get_super_y(task)
    y = task.y
    y = y[shuffle_idx]
    if not task.is_discrete:
        x = x.reshape(-1, task.x.shape[-1])
    y = y.reshape(-1, 1)
    # w = get_weights(y, base_temp=0.03 * length)
    # w = get_weights(y, base_temp=0.1)
    w = get_weights(y, temp=temp)

    # TODO: Modify
    # full_ds = DKittyMorphologyDataset()
    # y = (y - full_ds.y.min()) / (full_ds.y.max() - full_ds.y.min())

    print(w.shape)

    if val_frac is None:
        val_frac = 0

    val_length = int(length * val_frac)
    train_length = length - val_length

    train_dataset = RvSDataset(
        task,
        x[:train_length],
        y[:train_length],
        # None,
        w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        # None,
        w[train_length:],
        device,
        mode='val')

    return train_dataset, val_dataset


def split_dataset_based_on_top_candidates(task, size, val_frac=None, device=None, temp=None, is_target=False):
    length = task.y.shape[0]
    shuffle_idx = np.arange(length)
    np.random.shuffle(shuffle_idx)

    if task.is_discrete:
        task.map_to_logits()
        x = task.x[shuffle_idx]
        # x = x.reshape(x.shape[1:])
        x = x.reshape(x.shape[0], -1)
    else:
        x = task.x[shuffle_idx]

    # y = temp_get_super_y(task)
    y = task.y
    y = y[shuffle_idx]
    if not task.is_discrete:
        x = x.reshape(-1, task.x.shape[-1])
    y = y.reshape(-1, 1)
    w = get_weights(y, temp=temp)

    # Sort y in descending order and select the top 'size' instances
    if size is None:
        size = length
    else:
        size = length // 3

    sorted_indices = np.argsort(-y, axis=0).flatten()
    top_indices = sorted_indices[:size]
    x = x[top_indices]
    y = y[top_indices]
    w = w[top_indices]
    size = x.shape[0]
    shuffle_idx = np.arange(size)
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    w = w[shuffle_idx]

    if is_target:
        target_xy = np.load(f"experiments/{args.task}/{TASKNAME2TASK[args.task]}_pseudo_target_123.npy",
                            allow_pickle=True).item()
        x = np.array(target_xy["x"])
        y = np.array(target_xy["pred_y"])[:, np.newaxis]
        w = get_weights(y, temp=temp)
        size = x.shape[0] // 3
        shuffle_idx = np.arange(size)
        np.random.shuffle(shuffle_idx)
        x = x[shuffle_idx]
        y = y[shuffle_idx]
        w = w[shuffle_idx]

    print(x.shape)

    if val_frac is None:
        val_frac = 0

    val_length = int(size * val_frac)
    train_length = size - val_length

    train_dataset = RvSDataset(
        task,
        x[:train_length],
        y[:train_length],
        # None,
        w[:train_length],
        device,
        mode='train')
    val_dataset = RvSDataset(
        task,
        x[train_length:],
        y[train_length:],
        # None,
        w[train_length:],
        device,
        mode='val')

    return train_dataset, val_dataset


class RvSDataModule(pl.LightningDataModule):

    def __init__(self, task, batch_size, num_workers, val_frac, device, temp, top_candidates_size=None, is_target=False):
        super().__init__()

        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.device = device
        self.train_dataset = None
        self.val_dataset = None
        self.temp = temp
        self.top_candidates_size = top_candidates_size
        self.is_target = is_target

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = split_dataset_based_on_top_candidates(
            self.task, size=self.top_candidates_size, val_frac=self.val_frac, device=self.device, temp=self.temp, is_target=self.is_target)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=self.num_workers,
                                  batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size)
        return val_loader


def log_args(
        args: configargparse.Namespace,
        wandb_logger: pl.loggers.wandb.WandbLogger,
) -> None:
    """Log arguments to a file in the wandb directory."""
    wandb_logger.log_hyperparams(args)

    args.wandb_entity = wandb_logger.experiment.entity
    args.wandb_project = wandb_logger.experiment.project
    args.wandb_run_id = wandb_logger.experiment.id
    args.wandb_path = wandb_logger.experiment.path

    out_directory = wandb_logger.experiment.dir
    pprint(f"out_directory: {out_directory}")
    args_file = os.path.join(out_directory, args_filename)
    with open(args_file, "w") as f:
        try:
            json.dump(args.__dict__, f)
        except AttributeError:
            json.dump(args, f)


def run_training(
        taskname: str,
        seed: int,
        wandb_logger: pl.loggers.wandb.WandbLogger,
        args,
        device=None,
):
    epochs = args.epochs
    max_steps = args.max_steps
    train_time = args.train_time
    hidden_size = args.hidden_size
    depth = args.depth
    learning_rate = args.learning_rate
    auto_tune_lr = args.auto_tune_lr
    dropout_p = args.dropout_p
    checkpoint_every_n_epochs = args.checkpoint_every_n_epochs
    checkpoint_every_n_steps = args.checkpoint_every_n_steps
    checkpoint_time_interval = args.checkpoint_time_interval
    batch_size = args.batch_size
    val_frac = args.val_frac
    use_gpu = args.use_gpu
    device = device
    num_workers = args.num_workers
    vtype = args.vtype
    T0 = args.T0
    normalise_x = args.normalise_x
    normalise_y = args.normalise_y
    debias = args.debias
    score_matching = args.score_matching

    set_seed(seed)
    if taskname != 'tf-bind-10':
        task = design_bench.make(TASKNAME2TASK[taskname])
    else:
        task = design_bench.make(TASKNAME2TASK[taskname],
                                 dataset_kwargs={"max_samples": 30000})

    if task.is_discrete:
        task.map_to_logits()
        # print("X shape:", task.x.shape)
    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()

    print("TASK NAME: ", taskname)

    if not score_matching:
        model = DiffusionTest(taskname=taskname,
                              task=task,
                              learning_rate=learning_rate,
                              hidden_size=hidden_size,
                              vtype=vtype,
                              beta_min=args.beta_min,
                              beta_max=args.beta_max,
                              simple_clip=args.simple_clip,
                              T0=T0,
                              debias=debias,
                              dropout_p=dropout_p)
    else:
        print("Score matching loss")
        model = DiffusionScore(taskname=taskname,
                               task=task,
                               learning_rate=learning_rate,
                               hidden_size=hidden_size,
                               vtype=vtype,
                               beta_min=args.beta_min,
                               beta_max=args.beta_max,
                               simple_clip=args.simple_clip,
                               T0=T0,
                               debias=debias,
                               dropout_p=dropout_p)

    # monitor = "val_loss" if val_frac > 0 else "train_loss"
    monitor = "elbo_estimator" if val_frac > 0 else "train_loss"
    checkpoint_dirpath = os.path.join(wandb_logger.experiment.dir,
                                      checkpoint_dir)
    checkpoint_filename = f"{taskname}_{seed}-" + "-{epoch:03d}-{" + f"{monitor}" + ":.4e}"
    periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename=checkpoint_filename,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=checkpoint_every_n_epochs,
        every_n_train_steps=checkpoint_every_n_steps,
        train_time_interval=pd.Timedelta(checkpoint_time_interval).
        to_pytimedelta() if checkpoint_time_interval is not None else None,
    )
    val_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor=monitor,
        filename=checkpoint_filename,
        save_last=True,  # save latest model
        save_top_k=1,  # save top model based on monitored loss
    )
    trainer = pl.Trainer(
        devices=1,
        # auto_lr_find=auto_tune_lr,
        max_epochs=epochs,
        # max_steps=max_steps,
        max_time=train_time,
        logger=wandb_logger,
        # progress_bar_refresh_rate=20,
        callbacks=[periodic_checkpoint_callback, val_checkpoint_callback],
        # track_grad_norm=2,  # logs the 2-norm of gradients
        limit_val_batches=1.0 if val_frac > 0 else 0,
        limit_test_batches=0,
    )

    # train_dataset, val_dataset = split_dataset(task=task,
    #                                            val_frac=val_frac,
    #                                            device=device)
    # train_data_module = DataLoader(train_dataset)  #, num_workers=num_workers)
    # val_data_module = DataLoader(val_dataset)  #, num_workers=num_workers)

    # trainer.fit(model, train_data_module, val_data_module)
    data_module = RvSDataModule(task=task,
                                val_frac=val_frac,
                                device=device,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                temp=args.temp,
                                top_candidates_size=args.top_candidates_size,
                                is_target=args.is_target)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        )
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys()),
                        default='ant',
                        )
    # reproducibility
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help=
        "sets the random seed; if this is not specified, it is chosen randomly",
    )
    parser.add_argument("--condition", default=0.0, type=float)
    parser.add_argument("--lamda", default=0.0, type=float)
    parser.add_argument("--temp", default='90', type=str)
    parser.add_argument("--suffix", type=str, default="")
    # experiment tracking
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--score_matching", action='store_true', default=False)
    # training
    train_time_group = parser.add_mutually_exclusive_group(required=False)
    train_time_group.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="the number of training epochs.",
    )
    train_time_group.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help=
        "the number of training gradient steps per bootstrap iteration. ignored "
        "if --train_time is set",
    )
    train_time_group.add_argument(
        "--train_time",
        default="00:01:00:00",
        type=str,
        help="how long to train, specified as a DD:HH:MM:SS str",
    )
    parser.add_argument("--num_workers",
                        default=1,
                        type=int,
                        help="Number of workers")
    checkpoint_frequency_group = parser.add_mutually_exclusive_group(
        required=False)
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        help="the period of training epochs for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        help="the period of training gradient steps for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_time_interval",
        type=str,
        help="how long between saving checkpoints, specified as a HH:MM:SS str",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="place networks and data on the GPU",
    )
    parser.add_argument('--simple_clip', action="store_true", default=False)
    parser.add_argument("--which_gpu",
                        default=0,
                        type=int,
                        help="which GPU to use")
    parser.add_argument(
        "--normalise_x",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalise_y",
        action="store_true",
        default=False,
    )

    # i/o
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'cifar'],
                        default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0',
                        type=float,
                        default=1.0,
                        help='integration time')
    parser.add_argument('--vtype',
                        type=str,
                        choices=['rademacher', 'gaussian'],
                        default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=1.)

    # model
    parser.add_argument(
        '--real',
        type=eval,
        choices=[True, False],
        default=True,
        help=
        'transforming the data from [0,1] to the real space using the logit function'
    )
    parser.add_argument(
        '--debias',
        action="store_true",
        default=False,
        help=
        'using non-uniform sampling to debias the denoising score matching loss'
    )

    # TODO: remove
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="learning rate for each gradient step",
    )
    parser.add_argument(
        "--auto_tune_lr",
        action="store_true",
        default=False,
        help=
        "have PyTorch Lightning try to automatically find the best learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        help="dropout probability",
        default=0,
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        required=False,
        default=20.0,
    )
    parser.add_argument(
        "--top_candidates_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--is_target",
        type=eval,
        choices=[True, False],
        default=False,
    )
    args = parser.parse_args()

    wandb_project = "score-matching " if args.score_matching else "sde-flow"

    args.seed = np.random.randint(2 ** 31 - 1) if args.seed is None else args.seed
    set_seed(args.seed + 1)
    device = configure_gpu(args.use_gpu, args.which_gpu)

    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    if args.mode == 'train':
        if not os.path.exists(expt_save_path):
            os.makedirs(expt_save_path)
        wandb_logger = pl.loggers.wandb.WandbLogger(
            project=wandb_project,
            name=f"{args.name}_task={args.task}_{args.seed}",
            save_dir=expt_save_path)
        log_args(args, wandb_logger)
        run_training(
            taskname=args.task,
            seed=args.seed,
            wandb_logger=wandb_logger,
            args=args,
            device=device,
        )
    else:
        raise NotImplementedError
