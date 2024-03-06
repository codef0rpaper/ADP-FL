import logging
import os
import time
import argparse
from scipy import ndimage
from medpy import metric
import numpy as np
import torch


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}.log".format(logger_name))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime

def setup_parser():
    parser = argparse.ArgumentParser()
    # Federated training settings
    parser.add_argument("-N", "--clients", help="The number of participants", type=int, default=2)
    parser.add_argument(
        "-VN", "--virtual_clients", help="The number of virtual clients", type=int, default=1
    )
    parser.add_argument("--lr", type=float, default=0, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=-1, help="learning rate decay for scheduler"
    )
    parser.add_argument(
        "--early", action="store_true", help="early stop w/o improvement over 20 epochs"
    )
    parser.add_argument("--batch", type=int, default=0, help="batch size")
    parser.add_argument("--rounds", type=int, default=100, help="iterations for communication")
    parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs")
    parser.add_argument("--mode", type=str, default="fedavg", help="different FL algorithms")
    parser.add_argument(
        "--adp_noise", action="store_true", help="add adaptive noise"
    )
    parser.add_argument(
        "--adp_round", action="store_true", help="adjust round adaptive"
    )
    parser.add_argument(
        "--pretrain", action="store_true", help="Use Alexnet/ResNet pretrained on Imagenet"
    )
    # Experiment settings
    parser.add_argument("--exp", type=str, default=None, help="exp name")
    parser.add_argument(
        "--save_path", type=str, default="", help="path to save the checkpoint"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from the save path checkpoint"
    )
    parser.add_argument("--gpu", type=str, default="0", help='gpu device number e.g., "0,1,2"')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    # Data settings
    parser.add_argument(
        "--data", type=str, default="RSNA-ICH", help="Different dataset: cifar10, cifar10c"
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=float, default=1, help="Sample rate at each round"
    )
    parser.add_argument("--leave", type=str, default="no", help="leave one domain/client out")
    parser.add_argument("--merge", action="store_true", help="Use a global val/test set")
    parser.add_argument("--balance", action="store_true", help="Do not balance training data")
    parser.add_argument(
        "--weighted_avg",
        action="store_true",
        help="Use weighted average, default is pure avg, i.e., 1/N",
    )
    # CIFAR-10 Split
    parser.add_argument(
        "--split",
        type=str,
        default="UNI",
        choices=["UNI", "POW", "LabelNonIID", "FeatureNonIID"],
        help="Data distribution setting",
    )
    # Method settings
    parser.add_argument(
        "--local_bn", action="store_true", help="Do not aggregate BN during communication"
    )
    parser.add_argument("--generalize", action="store_true", help="Generalization setting")
    parser.add_argument("--gn", action="store_true", help="use groupnorm")
    parser.add_argument("--selu", action="store_true", help="use_selu")
    parser.add_argument(
        "--comb",
        type=str,
        default="times",
        choices=["times", "plus"],
        help="Combination mode, cos+lval or cosxlval",
    )
    parser.add_argument(
        "--free",
        type=int,
        default=-1,
        help="Set a client as free rider (always providing repeating data)",
    )
    parser.add_argument("--noisy", type=int, default=-1, help="Set a client as a noisy client")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="The hyper parameter for tune loss for DC"
    )
    parser.add_argument(
        "--adaclip",
        action="store_true",
        help="use adaptive clip (meadian norm as norm clip bound)",
    )
    parser.add_argument(
        "--noclip",
        action="store_true",
        help="Do not clip gradients, only for ideal trial",
    )

    parser.add_argument("--ema", type=float, default=0.0, help="the rate for keeping history")

    # FL algorithm hyper parameters
    parser.add_argument("--mu", type=float, default=1e-3, help="The hyper parameter for fedprox")
    parser.add_argument("--S", type=float, default=10, help="The hyper parameter for dp")
    # parser.add_argument("--sigma", type=float, default=1e-4, help="The hyper parameter for dp")

    parser.add_argument("--epsilon", type=float, default=None, help="The budget for dp")
    parser.add_argument("--noise_multiplier", type=float, default=None, help="The budget for dp")
    parser.add_argument("--delta", type=float, default=1e-3, help="The budget for dp")
    parser.add_argument("--accountant", type=str, default="prv", help="The dp accountant")
    parser.add_argument(
        "--dp_mode",
        type=str,
        default="overhead",
        choices=["overhead", "bounded"],
        help="Using which mode to do private training. Options: overhead, bounded.",
    )
    parser.add_argument(
        "--balance_split", action="store_true", help="Activate balance virtual client splitting."
    )
    parser.add_argument("--test", action="store_true", help="Running test mode.")
    parser.add_argument("--ckpt", type=str, default="None", help="Path for the testing ckpt")
    parser.add_argument(
        "--adam_lr", type=float, default=0.1, help="Global learning rate for FedAdam."
    )
    parser.add_argument("--dp2_interval", type=int, default=3, help="Interval for DP2-RMSProp")
    parser.add_argument(
        "--rmsprop_lr", type=float, default=1, help="Global learning rate for DP2-RMSProp."
    )
    parser.add_argument(
        "--ada_vn",
        action="store_true",
        help="Running adaptive virtual client splitting. VN = ceil(vn*self.virtual_clients), vn may vary in range [-2,2]",
    )
    parser.add_argument(
        "--init_vn",
        action="store_true",
        help="Esitimating virtual client splitting using first round results.",
    )
    parser.add_argument(
        "--ada_stable",
        action="store_true",
        help="VN = ceil(vn)*self.virtual_clients, vn should be unchanged",
    )
    parser.add_argument(
        "--ada_prog",
        action="store_true",
        help="VN = ceil(vn/2 * self.virtual_clients), progressively reaching optimal VN",
    )
    parser.add_argument(
        "--data_path",
        type=str, default="",
        help="dataset root path",
    )
    parser.add_argument("--clip", type=float, default=0, help="Gradient clip")
    parser.add_argument(
        "--center",
        action="store_true",
        help="center training",
    )
    parser.add_argument("--round_factor", type=float, default=0.99, help="round factor")
    parser.add_argument("--round_threshold", type=float, default=0.0001, help="round threshold")
    parser.add_argument("--folder", type=str, default=None, help="folder name")
    parser.add_argument("--no_dp", action="store_true", help="no dp")
    return parser

def _eval_haus(pred, gt):
    """
    :param pred: whole brain prediction
    :param gt: whole
    :param detail:
    :return: a list, indicating Dice of each class for one case
    """
    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().numpy()
    pred[pred >= 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    gt = gt.detach().cpu().numpy()
    # haus = []

    """ During the first several epochs, prediction may be all zero, which will throw an error. """
    if pred.sum() == 0:
        hd = torch.tensor(1000.0)
    else:
        hd = metric.binary.hd95(pred, gt)
    # hd = metric.binary.hd(gt, pred)

    return hd


def _eval_iou(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5, smooth=1e-5):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = (outputs > threshold).long()
    labels = labels.long()
    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return iou.mean()


def cos_sim(a, b):
    from numpy import dot
    from numpy.linalg import norm

    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def metric_log_print(metric_dict, cur_metric):
    if "AUC" in list(cur_metric.keys()):
        clients_accs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Acc" in k]
        )
        metric_dict = dict_append("mean_Acc", clients_accs_avg, metric_dict)

        clients_aucs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "AUC" in k]
        )
        metric_dict = dict_append("mean_AUC", clients_aucs_avg, metric_dict)

        clients_sens_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Sen" in k]
        )
        metric_dict = dict_append("mean_Sen", clients_sens_avg, metric_dict)

        clients_spes_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Spe" in k]
        )
        metric_dict = dict_append("mean_Spe", clients_spes_avg, metric_dict)

        clients_f1_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "F1" in k]
        )
        metric_dict = dict_append("mean_F1", clients_f1_avg, metric_dict)

        out_str = f" | {'AUC'}: {clients_aucs_avg:.4f} | {'Acc'}: {clients_accs_avg:.4f} | {'Sen'}: {clients_sens_avg:.4f} | {'Spe'}: {clients_spes_avg:.4f} | {'F1'}: {clients_f1_avg:.4f}"
    elif "Dice" in list(cur_metric.keys()):
        clients_dice_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Dice" in k]
        )
        metric_dict = dict_append("mean_Dice", clients_dice_avg, metric_dict)

        clients_hd_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "HD" in k]
        )
        metric_dict = dict_append("mean_HD", clients_hd_avg, metric_dict)

        clients_iou_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "IoU" in k]
        )
        metric_dict = dict_append("mean_IoU", clients_iou_avg, metric_dict)

        out_str = f" | {'Dice'}: {clients_dice_avg:.4f} | {'HD'}: {clients_hd_avg:.4f} | {'IoU'}: {clients_iou_avg:.4f}"
    else:
        raise NotImplementedError

    return metric_dict, out_str

def dict_append(key, value, dict_):
    """
    dict_[key] = list()
    """
    if key not in dict_:
        dict_[key] = [value]
    else:
        dict_[key].append(value)
    return dict_
