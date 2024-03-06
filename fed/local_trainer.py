import torch
from torch import nn, autograd
# from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import random
from sklearn import metrics
import copy
from utils.loss import DiceLoss
from utils.util import _eval_haus, _eval_iou

def metric_calc(gt, pred, score):
    tn, fp, fn, tp = metrics.confusion_matrix(gt, pred).ravel()
    acc = metrics.accuracy_score(gt, pred)
    try:
        auc = metrics.roc_auc_score(gt, score)
    except ValueError:
        auc = 0
    sen = metrics.recall_score(gt, pred)  # recall = sensitivity = TP/TP+FN
    spe = tn / (tn + fp)  # specificity = TN / (TN+FP)
    f1 = metrics.f1_score(gt, pred)
    return [tn, fp, fn, tp], auc, acc, sen, spe, f1

class LocalUpdateDP(object):
    def __init__(self, args, train_loader, val_loader, test_loader, loss_fun, model ,device, logging, idx) -> None:
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fun = loss_fun
        self.lr = args.lr
        self.model = model
        # if args.mode == "fedsgd":
        #     self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.args.lr, momentum=0.9)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=self.args.rounds
                    ) if args.lr_decay else None
        self.device = device
        self.logging = logging
        self.idx = idx

    def update_model(self, model):
        self.model.load_state_dict(model)

    def train(self, sigma=None):
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #                 optimizer, T_max=self.args.rounds
        #             )
        optimizer = self.optimizer
        loss_all = 0
        segmentation = "UNet" in self.model.__class__.__name__
        train_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        num_sample_test = 0
        origin_model = copy.deepcopy(self.model).to("cpu")
        w_k_2 = None
        w_k_1 = None

        for step, data in enumerate(self.train_loader):
            if step == len(self.train_loader) - 2:
                w_k_2 = copy.deepcopy(self.model).to("cpu")
            if step == len(self.train_loader) - 1:
                w_k_1 = copy.deepcopy(self.model).to("cpu")
                
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)
            
            self.model.to(self.device)
            self.model.train()
            optimizer.zero_grad()
            inp = inp.to(self.device)
            output = self.model(inp)
            
            if self.args.data.startswith("prostate"):
                loss = self.loss_fun(output[:, 0, :, :], target)
            else:
                loss = self.loss_fun(output, target)

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        train_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        train_acc["HD"] = 0.0
                    else:
                        train_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        train_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            train_acc["HD"] += hd
                            num_sample_test += 1
                else:
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        train_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            loss_all += loss.item()

        loss = loss_all / len(self.train_loader)
        # print(f"Site-{self.idx} | Train Loss: {loss_all}")
        if segmentation:
            acc = {
                "Dice": train_acc["Dice"] / len(self.train_loader),
                "IoU": train_acc["IoU"] / len(self.train_loader),
                "HD": train_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }
        
        # self.lr = scheduler.get_last_lr()[0]

        out_str = ""
        for k, v in acc.items():
            out_str += " | Train {}: {:.4f} ".format(k, v)
        self.logging.info(
            "Site-{:<5s} rounds:{:<2d} | Train Loss: {:.4f}{}".format(
                str(self.idx), len(self.train_loader), loss, out_str
            )
        )

        g_k_1 = self._compute_gradiant(old_model=w_k_2, new_model=w_k_1)
        model_estimate = self._compute_model(old_model=w_k_1, gradient=g_k_1)
        gradient_estimate = self._compute_gradiant(old_model=origin_model, new_model=model_estimate)

        # C_for_clip = self._compute_norm_l2_model_dict(gradient_estimate)
        g_all = self._compute_gradiant(old_model=origin_model, new_model=copy.deepcopy(self.model).to("cpu"))
        C_for_clip = self.args.C
        # TODO : Adaptive C 
        beta_clip_fact = self._compute_beta(w_k_1, g_k_1, C_for_clip)
        
        if self.args.debug:
            self.logging.info("Site-{:<5s} | before add noise model norm: {:.8f}".format(
                str(self.idx), 
                self._compute_norm_l2_model(self.model)
            ))
        
            self.logging.info("Site-{:<5s} | before clip gradiant norm: {:.8f}".format(
                str(self.idx), 
                self._compute_norm_l2_model_dict(g_all)
            ))
        
        if not self.args.no_dp and sigma != None:
            # clip gradients and add noises
            if self.args.adp_noise:
                sensitivity_params = self.clip_gradients(self.model, beta_clip_fact, origin_model, model_estimate)
            else:
                self.clip_gradients_normal_l2(self.model, origin_model, C_for_clip)
            
            g_all = self._compute_gradiant(old_model=origin_model, new_model=copy.deepcopy(self.model).to("cpu"))
            if self.args.debug:
                self.logging.info("Site-{:<5s} | after clip gradiant norm: {:.8f}".format(
                    str(self.idx), 
                    self._compute_norm_l2_model_dict(g_all)
                ))
                self.logging.info("Site-{:<5s} | after clip model norm: {:.8f}".format(
                    str(self.idx), 
                    self._compute_norm_l2_model(self.model)
                ))

            if self.args.adp_noise:
                self.add_noise_per_param(self.model, sensitivity_params, sigma)
            # self.add_noise_per_param_on_g(model, sensitivity_params, sigma, g_all, origin_model)
            else:
                self.add_noise(self.model, sigma)
            
            norm = 0
            for _, param in self.model.named_parameters():
                norm += torch.norm(param, 2).item()**2
            self.logging.info("Site-{:<5s} | after add noise norm: {:.8f}".format(str(self.idx), norm**0.5))
        return self.model.to("cpu").state_dict(), loss
    
    def _test(self, model, mode):
        assert mode in ["val", "test"]
        data_loader = self.val_loader if mode == "val" else self.test_loader
        model.to(self.device)
        model.eval()
        loss_all = 0
        num_sample_test = 0

        segmentation = "UNet" in model.__class__.__name__
        test_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        for _, data in enumerate(data_loader):
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)

            inp = inp.to(self.device)
            output = model(inp)

            if self.args.data.startswith("prostate"):
                loss = self.loss_fun(output[:, 0, :, :], target)
            else:
                loss = self.loss_fun(output, target)

            loss_all += loss.item()

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        test_acc["HD"] = 0.0
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            test_acc["HD"] += hd
                            num_sample_test += 1
                else:
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

        loader_length = len(data_loader)
        loss = loss_all / loader_length
        # acc = test_acc/ len(data_loader) if segmentation else correct/total
        if segmentation:
            acc = {
                "Dice": test_acc["Dice"] / loader_length,
                "IoU": test_acc["IoU"] / loader_length,
                "HD": test_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }
        model.to("cpu")
        return loss, acc
    
    def validation_model(self):
        return self._test(self.model, "val")
    
    def test_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        test_model = copy.deepcopy(self.model).to(self.device)
        test_model.load_state_dict(checkpoint["server_model"])
        return self._test(test_model, "test")

    
    def clip_gradients(self, model, beta, origin_model, model_estimate):
        sensitivity_params = {}
        # get dict of model
        origin_model_dict = origin_model.state_dict()
        model_estimate_dict = model_estimate.to("cpu").state_dict()
        model.to("cpu")
        # each param in model.parameters(), clip_m = beta/2 * |model_weights - gradient|, param = min(max(param, param_orgin_model - clip_m), param_orgin_model + clip_m)
        for name, param in model.named_parameters():
            if name in origin_model_dict and name in model_estimate_dict:
                # clip_m = beta/2 * |model_weights - gradient|
                clip_m = beta / 2 * torch.norm(model_estimate_dict[name], 2)
                sensitivity_params[name] = 2 * clip_m
                distance = torch.norm(origin_model_dict[name] - param, 2)
                param.data = origin_model_dict[name] - (clip_m / distance if clip_m < distance else 1) * (origin_model_dict[name] - param.data)
        model.to(self.device)
        return sensitivity_params

    def clip_gradients_normal_l2(self, model, origin_model, C):
        model.to("cpu")
        grad_norm = []
        for name, param in model.named_parameters():
            if name in origin_model.state_dict():
                grad_norm.append(torch.norm(origin_model.state_dict()[name] - param, 2))
                # if distance > C:
                #     param.data = origin_model.state_dict()[name] + (param - origin_model.state_dict()[name]) * (C / distance if distance > C else 1)
        grad_norm = torch.asarray(grad_norm)
        global_norm = torch.norm(grad_norm, 2)
        norm_factor = torch.minimum(C / (global_norm + 1e-15), torch.tensor(1.0))
        for name, param in model.named_parameters():
            if name in origin_model.state_dict():
                param.data = origin_model.state_dict()[name] + (param - origin_model.state_dict()[name]) * norm_factor
        model.to(self.device)
        return model

    def add_noise(self, model, sigma):
        for p in model.parameters():
            # add normal noise with std = sigma
            p.data += torch.normal(mean=0, std=sigma, size=p.size()).to(p.device)

        return model
    
    def add_noise_per_param(self, model, sensitivity_params, sigma):
        # get number of parameters
        params_num = len(model.state_dict())
        
        for name, param in model.named_parameters():
            if name in sensitivity_params:
                sigma_m = params_num**0.5 * sigma * sensitivity_params[name]
                noise = torch.normal(mean=0, std=sigma_m, size=param.size()).to(param.device)
                param.data += noise

        return model
    
    def add_noise_per_param_on_g(self, model, sensitivity_params, sigma, g_all, origin_model):
        # get number of parameters
        params_num = len(model.state_dict())
        
        for name in g_all.keys():
            if name in sensitivity_params:
                sigma_m = params_num**0.5 * sigma * sensitivity_params[name]
                noise = torch.normal(mean=0, std=sigma_m, size=g_all[name].size()).to("cpu")
                g_all[name].data += noise

        model_dict = origin_model.state_dict()
        for name in model_dict.keys():
            if name in g_all.keys():
                model_dict[name] += g_all[name]

        model.load_state_dict(model_dict)

        return model
    
    def get_grads(self, model):
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)
    
    def get_params(self, model):
        params = []
        for p in model.parameters():
            params.append(p.view(-1))
        return torch.cat(params)
    
    def _compute_gradiant(self, old_model, new_model):
        old_param = old_model.to("cpu").state_dict()
        new_param = new_model.to("cpu").state_dict()
        gradients = {}
        for name in old_param.keys():
            gradients[name] = old_param[name] - new_param[name]
        return gradients
    
    def _compute_beta(self, model, gradient, C):
        beta_clip_fact = 0
        model_weights = model.to("cpu").state_dict()
        for name in model_weights.keys():
            param_diff = model_weights[name] - gradient[name]  # w_k_1 - g_k_1
            beta_clip_fact += torch.norm(param_diff, 2).item()**2
        beta_clip_fact = 2 * C / beta_clip_fact**0.5  # Taking the square root to get the L2 norm
        self.logging.info(f"beta_clip_fact: {beta_clip_fact}")
        return beta_clip_fact
    
    def _compute_norm_l2_model(self, model):
        return sum([torch.norm(param, 2).item()**2 for _, param in model.named_parameters()])**0.5
    
    def _compute_norm_l2_model_dict(self, model_dict):
        return sum([torch.norm(param, 2).item()**2 for _, param in model_dict.items()])**0.5
    
    def _compute_model(self, old_model, gradient):
        model_copy = copy.deepcopy(old_model).to("cpu")
        for name, param in model_copy.named_parameters():
            if name in gradient:
                param.data -= gradient[name]
        
        return model_copy
