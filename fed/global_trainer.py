import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import copy
import time
import random
import math
import logging
import pandas as pd
from sklearn import metrics
from utils.loss import DiceLoss
from utils.util import _eval_haus, _eval_iou, dict_append, metric_log_print
from dataset.dataset import DatasetSplit
from utils.nova_utils import SimpleFedNova4Adam

from fed.local_trainer import LocalUpdateDP


class FedTrainner(object):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs,
    ) -> None:
        self.args = args
        self.logging = logging
        self.device = device
        self.lr_decay = args.lr_decay > 0
        self.server_model = server_model
        self.train_sites = train_sites
        self.val_sites = val_sites
        self.client_num_train = len(train_sites)
        self.client_num = len(val_sites)
        self.sample_rate = args.sample_rate
        assert self.sample_rate > 0 and self.sample_rate <= 1
        self.aggregation_idxs = None
        self.aggregation_client_num = max(int(self.client_num * self.sample_rate), 1)
        self.client_weights = (
            [1 / self.aggregation_client_num] * len(self.aggregation_client_num) # pure avg
            if client_weights is None
            else client_weights
        )
        self.current_iter = 0
        self.client_models = [copy.deepcopy(server_model) for idx in range(self.client_num)]
        self.client_grads = [None for i in range(self.client_num)]
        (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
        ) = ({}, {}, {}, {}, {}, {})

        self.generalize_sites = (
            kwargs["generalize_sites"] if "generalize_sites" in kwargs.keys() else None
        )

        self.train_loss["mean"] = []
        self.val_loss["mean"] = []
        self.test_loss["mean"] = []

        self.sigma = []
        self.used_sigma_reciprocal = 0

        self.best_changed = False
        # TODO 根据mode 选择聚合函数
        aggregation_dict = {
            "fedsgd": self.FedWeightAvg,
            "fedadam": self.FedAdamAggregation,
        }
        self.aggregation = aggregation_dict[args.mode]
        if args.mode == "fedadam":
            self.mt = None
            self.vt = None
            # momentum param for Adam optim
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.adam_tau = 1e-9

            self.adam_lr = args.adam_lr
        elif args.mode == "fedrmsprop":
            assert self.sample_rate == 1, "assume all clients join the training"
            self.interval = 3
            self.Gt = None
            self.At = None
            self.init_At = False
            self.rmsprop_gamma = 0.9
            self.rmsprop_epsilon = 1e-7
            self.rmsprop_lr = self.args.rmsprop_lr
            self.count = 0

    def save_metrics(self):
        metrics_pd = pd.DataFrame.from_dict(self.val_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.val_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_acc.csv"))

        metrics_pd = pd.DataFrame.from_dict(self.test_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.test_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_acc.csv"))
    
    def start(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
    ):
        self.clients = [
            LocalUpdateDP(
                args=self.args, 
                train_loader=train_loaders[idx], 
                val_loader=val_loaders[idx],
                test_loader=test_loaders[idx],
                loss_fun=loss_fun, 
                model=self.client_models[idx],
                device=self.device,
                logging=self.logging, 
                idx=idx
            ) for idx in range(len(train_loaders))
        ]
        self.logging.info("=====================FL Start=====================") 
        for iter in range(self.args.rounds):
            self.current_iter = iter
            if iter >= self.args.rounds:
                break
            self.logging.info("------------ Round({:^5d}/{:^5d}) Train ------------".format(iter, self.args.rounds))
            t_start = time.time()
            if not self.args.no_dp:
                sigma = self._calculate_sigma(
                    epsilon=self.args.epsilon, 
                    delta=self.args.delta, 
                    iter=iter, 
                    rounds=self.args.rounds,
                    sensitiviy=2*self.args.C
                )
                self.logging.info(f"sigma: {sigma}")
            else:
                sigma = None
            w_locals, loss_locals = [], []
            if self.sample_rate < 1:
                self.aggregation_idxs = random.sample(
                    list(range(self.client_num_train)), self.aggregation_client_num
                )
            else:
                self.aggregation_client_num = len(self.client_models)
                self.aggregation_idxs = list(range(len(self.client_models)))
            for idx in self.train_sites:
                local = self.clients[idx]
                w, loss = local.train(sigma=sigma)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            
            # update global weights
            w_glob = self.aggregation(w_locals, self.client_weights)
            # copy weight to net_glob
            self.server_model.load_state_dict(w_glob)
            # update client model
            for index in self.aggregation_idxs:
                self.clients[index].update_model(self.server_model.to("cpu").state_dict())

            # validation
            self.logging.info("------------ Validation ------------")
            with torch.no_grad():
                assert len(self.val_sites) == len(val_loaders)
                for client_idx in self.aggregation_idxs:
                    local = self.clients[client_idx]
                    val_loss, val_acc = local.validation_model()
                    self.record(iter, client_idx, val_loss, val_acc, "val")

                clients_loss_avg = np.mean(
                    [v[-1] for k, v in self.val_loss.items() if "mean" not in k]
                )
                self.val_loss["mean"].append(clients_loss_avg)
                self.val_acc, out_str = metric_log_print(self.val_acc, val_acc)

                self.args.writer.add_scalar(f"Loss/val", clients_loss_avg, iter)

                mean_val_acc_ = (
                    self.val_acc["mean_Acc"][-1]
                    if "mean_Acc" in list(self.val_acc.keys())
                    else self.val_acc["mean_Dice"][-1]
                )
                self.logging.info(
                    " Site-Average | Val Loss: {:.4f}{}".format(
                        clients_loss_avg, out_str
                    )
                )

                if mean_val_acc_ > self.best_acc:
                    self.best_acc = mean_val_acc_
                    self.best_epoch = iter
                    self.best_changed = True
                    self.logging.info(
                        " Best Epoch:{} | Avg Val Acc: {:.4f}".format(
                            self.best_epoch, np.mean(mean_val_acc_)
                        )
                    )
                # save and test
                if self.best_changed:
                    model_dicts = self.prepare_ckpt(iter)
                    self.logging.info(
                        " Saving the local and server checkpoint to {}...".format(
                            SAVE_PATH + f"/model_best_{iter}"
                        )
                    )
                    torch.save(model_dicts, SAVE_PATH + f"/model_best_{iter}")
                    self.best_changed = False
                    test_sites = self.val_sites
                    for index in test_sites:
                        client = self.clients[index]
                        test_loss, test_acc = client.test_ckpt(SAVE_PATH + f"/model_best_{iter}")
                        self.record(iter, index, test_loss, test_acc, "test")
                    
                    clients_loss_avg = np.mean(
                        [v[-1] for k, v in self.test_loss.items() if "mean" not in k]
                    )
                    self.test_loss["mean"].append(clients_loss_avg)

                    self.test_acc, out_str = metric_log_print(self.test_acc, test_acc)

                    self.logging.info(
                        " Site-Average | Test Loss: {:.4f}{}".format(clients_loss_avg, out_str)
                    )
                elif iter % 10 == 0:
                    model_dicts = self.prepare_ckpt(iter)
                    self.logging.info(
                        " Saving the local and server checkpoint to {}...".format(
                            SAVE_PATH + f"/model_{iter}"
                        )
                    )
                    torch.save(model_dicts, SAVE_PATH + f"/model_{iter}")

                self.save_metrics()

            t_end = time.time()
            self.logging.info("Round {:3d}, Time:  {:.2f}s".format(iter, t_end - t_start))
            if self.args.adp_round and iter > 1:
                self.adaptive_rounds(iter)
            if iter == 3 and self.args.debug:
                break

        self.logging.info("=====================FL completed=====================")

    def record(self, iter, client_idx, loss, acc, mode):
        assert mode in ["val", "test"]
        loss_dict = getattr(self, f"{mode}_loss")
        acc_dict = getattr(self, f"{mode}_acc")
        loss_dict = dict_append(
            f"client_{self.val_sites[client_idx]}", loss, loss_dict
        )
        self.args.writer.add_scalar(
            f"Loss/val_{self.val_sites[client_idx]}", loss, iter
        )
        if isinstance(acc, dict):
            out_str = ""
            for k, v in acc.items():
                out_str += " | Val {}: {:.4f}".format(k, v)
                acc_dict = dict_append(
                    f"client{self.val_sites[client_idx]}_" + k, v, acc_dict
                )
                self.args.writer.add_scalar(
                    f"Performance/val_client{self.val_sites[client_idx]}_{k}", v, iter,
                )

            self.logging.info(
                " Site-{:<10s}| Val Loss: {:.4f}{}".format(
                    str(self.val_sites[client_idx]), loss, out_str
                )
            )
        else:
            acc_dict = dict_append(
                f"client_{self.val_sites[client_idx]}",
                round(acc, 4),
                acc_dict,
            )
            self.logging.info(
                " Site-{:<10s}| Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                    str(self.val_sites[client_idx]), loss, acc
                )
            )
            self.args.writer.add_scalar(
                f"Accuracy/val_{self.val_sites[client_idx]}", acc, iter
            )


    def _calculate_sigma(self, epsilon, delta, iter, rounds, sensitiviy):
        # \sigma^2=\frac{T-t}{\frac{\epsilon^2}{2\ln(1/\delta)}-\sum_{i=1}^{t}\frac{1}{\sigma_i^2}}
        sigma_sqr = (rounds - iter) / (epsilon**2 / (2 * sensitiviy**2 * np.log(1 / delta)) - self.used_sigma_reciprocal)
        self.sigma.append(sigma_sqr**0.5)
        self.used_sigma_reciprocal += 1 / sigma_sqr
        if iter == 1:
            self.logging.info(f"sigma_0: {self.sigma[0]} , sigma_1: {sigma_sqr**0.5}")
        return sigma_sqr**0.5
    
    def _compute_update(self, old_param, new_param):
        return [(new_param[key] - old_param[key]) for key in new_param.keys()]
    
    def FedWeightAvg(self, w, size):
        totalSize = sum(size)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w[0][k]*size[0]
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * size[i]
            # print(w_avg[k])
            w_avg[k] = torch.div(w_avg[k], totalSize)
        return w_avg

    def FedRMSPROPAgggegation(self, w, weight):
        branch = self.branch_func(self.interval)
        with torch.no_grad():
            server_model = copy.deepcopy(self.server_model).to("cpu")
            clients_grads = [self._compute_update(server_model.state_dict(), client_model) for client_model in w]
            if branch == 2:
                for i in range(len(clients_grads[0])):
                    denom = torch.sqrt(self.At[i]).add(self.rmsprop_epsilon)
                    for idx_client in range(self.client_num):
                        clients_grads[idx_client][i].div_(denom)
                lr = self.rmsprop_lr

            aggregated_grads = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]

            for i in range(self.client_num):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + clients_grads[i][idx] * weight[i]
                    )

            if self.Gt is None:
                self.Gt = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]

            if branch <= 1:
                for i in range(len(aggregated_grads)):
                    self.Gt[i].add_(aggregated_grads[i])
                self.count += 1
                lr = 1

            if branch == 1:
                Gt_avg_sq = copy.deepcopy(self.Gt)
                for i in range(len(aggregated_grads)):
                    Gt_avg_sq[i] = torch.pow(Gt_avg_sq[i].div(self.count), 2)

                    if self.init_At:
                        self.At[i].mul_(self.rmsprop_gamma).add_(
                            Gt_avg_sq[i], alpha=1 - self.rmsprop_gamma
                        )

                if not self.init_At:
                    self.At = copy.deepcopy(Gt_avg_sq)
                    self.init_At = True

                self.Gt = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]
                self.count = 0
            assert len(self.server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(w[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.add_(aggregated_grads[idx])
            return server_model.state_dict()
                
    def FedAdamAggregation(self, w, weight):
        with torch.no_grad():
            server_model = copy.deepcopy(self.server_model).to("cpu")
            clients_grads = [self._compute_update(server_model.state_dict(), client_model) for client_model in w]
            aggregated_grads = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]
            for i in range(len(weight)):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + clients_grads[i][idx] * weight[i]
                    )
            if self.mt is None:
                self.mt = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]

            if self.vt is None:
                self.vt = [torch.zeros_like(grad_term) for grad_term in clients_grads[0]]
            for idx, key in enumerate(server_model.state_dict().keys()):
                assert self.mt[idx].shape == aggregated_grads[idx].shape
                if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                    continue

                self.mt[idx].mul_(self.beta1).add_(aggregated_grads[idx], alpha=1 - self.beta1)
                self.vt[idx].mul_(self.beta2).add_(
                    torch.pow(aggregated_grads[idx], 2), alpha=1 - self.beta2
                )

                mt_h = self.mt[idx]
                denom = torch.sqrt(self.vt[idx]) + self.adam_tau

                aggregated_grads[idx].copy_(mt_h.mul(self.adam_lr).div(denom))
            assert len(self.server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(w[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.add_(aggregated_grads[idx])

        return server_model.state_dict()
        
    def prepare_ckpt(self, iter):
        if self.args.local_bn:
            model_dicts = {
                "server_model": self.server_model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "iter": iter,
            }
            for model_idx, model in enumerate(self.client_models):
                model_dicts["model_{}".format(model_idx)] = model.state_dict()
        else:
            model_dicts = {
                "server_model": self.server_model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "iter": iter,
            }
        return model_dicts
    
    def adaptive_rounds(self, iter):
        # TODO adaptive rounds
        factor = self.args.round_factor
        threshold = self.args.round_threshold
        if self.val_loss["mean"][iter-1] - self.val_loss["mean"][iter] < threshold:
            self.args.rounds = iter + int((self.args.rounds - iter) * factor)
    
    def branch_func(self, interval):
        i = self.current_iter

        s1 = interval
        s2 = interval
        on_interval = (i + 1 + s2) % (s1 + s2) == 0
        use_adaptive = (i // interval) % (1 + 1) > 0
        return 2 if use_adaptive else (1 if on_interval else 0)