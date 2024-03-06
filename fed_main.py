"""
Federated training main logic
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import time
import copy
import random
import math
import logging
import pandas as pd
import pickle as pkl
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import monai.transforms as monai_transforms

from nets.models import (
    DenseNet,
    UNet,
)

from fed.global_trainer import FedTrainner
from utils.util import setup_logger, get_timestamp, setup_parser
from utils.datasets import split_df, split_dataset, balance_split_dataset
from utils.workflow import prepare_workflow, prepare_args



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    prepare_args(args)

    SAVE_PATH = args.save_path

    # Set up logging
    lg = setup_logger(
        f"{args.mode}-{get_timestamp()}",
        args.log_path,
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    lg = logging.getLogger(f"{args.mode}-{get_timestamp()}")
    
    lg.info(args)

    generalize_sites = None
    (
        server_model,
        loss_fun,
        train_sites,
        val_sites,
        train_sets,
        val_sets,
        test_sets,
        train_loaders,
        val_loaders,
        test_loaders,
    ) = prepare_workflow(args, lg)

    assert (
        int(args.clients) == len(train_sites)
    ), f"Client num {args.clients}, train site num {len(train_sites)} do not match."
    assert len(val_loaders) == len(val_sites)  # == int(args.clients)
    train_total_len = sum([len(train_loaders[idx]) for idx in train_sites])
    client_weights = (
        [len(train_loaders[idx]) / train_total_len for idx in train_sites]
        if args.weighted_avg
        else [ 1.0 / float(args.clients) ] * int(args.clients)
    )
    lg.info("Client Weights: " + str(client_weights))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    if torch.cuda.device_count() > 1:
        device = "cuda"
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    lg.info(f"Device: {device}")
    print(f"Device: {device}")

    
    from torch.utils.tensorboard import SummaryWriter

    args.writer = SummaryWriter(args.log_path)

    trainer = FedTrainner(
        args,
        lg,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=client_weights,
        generalize_sites=generalize_sites,
    )

    trainer.best_changed = False
    trainer.early_stop = 20

    trainer.client_steps = [torch.tensor(len(train_loader)) for train_loader in train_loaders]
    print("Client steps:", trainer.client_steps)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        trainer.server_model.load_state_dict(checkpoint["server_model"])
        if args.local_bn:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(
                    checkpoint["model_{}".format(client_idx)]
                )
        else:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(checkpoint["server_model"])
        trainer.best_epoch, trainer.best_acc = checkpoint["best_epoch"], checkpoint["best_acc"]
        trainer.start_iter = int(checkpoint["a_iter"]) + 1

        print("Resume training from epoch {}".format(trainer.start_iter))
    else:
        # log the best for each model on all datasets
        trainer.best_epoch = 0
        trainer.best_acc = 0.0
        trainer.start_iter = 0

    if args.test:
        trainer.inference(args.ckpt, test_loaders, loss_fun, val_sites, process=True)
    else:
        try:
            trainer.start(
                train_loaders, val_loaders, test_loaders, loss_fun, SAVE_PATH
            )
        except NotImplementedError:
            print("private finish")


    logging.shutdown()