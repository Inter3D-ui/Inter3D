import torch.optim.lr_scheduler
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from .base_regularization import Distortion, compute_plane_smoothness, NeDepth
from typing import Sequence
from einops import rearrange, repeat


class CompositeLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * (torch.pow(results['rgb'] - batch['rgb'], 2)).mean()


class DepthLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * torch.abs(results['depth'] - batch['depth']).mean()


class SemanticLoss:
    def __init__(self, weight=1e-2):
        self.weight = weight
        self.loss = nn.CrossEntropyLoss()

    def apply(self, results, batch):
        sem = results['semantic']
        # probabilities = F.softmax(sem, dim=1)
        # loss = torch.zeros_like(sem).float()
        # for i in range(sem.size(0)):
        #     label_one_hot = torch.zeros(sem.size(1), device=sem.device, dtype=torch.int64). \
        #         scatter_(0, batch['semantic'][i].unsqueeze(0), 1)
        #     loss[i] = -torch.log(probabilities[i] + 1e-9) * label_one_hot
        # loss = loss.sum() / sem.size(0)
        # return self.weight * loss
        return self.weight * (self.loss(sem, batch['semantic']))


class DistortionLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results):
        loss = (self.weight * Distortion.apply(results['ws'], results['deltas'],
                                               results['ts'], results['rays_a'])).mean()
        return loss


class OpacityLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results):
        o = results['opacity'].clamp(1e-5, 1 - 1e-5)
        return (self.weight * -(o * torch.log(o))).mean()


class TimeSmoothness:
    def __init__(self, weight=1e-4):
        self.weight = weight

    def apply(self, multi_res_grids, stage):
        total = 0
        for grids in multi_res_grids:
            grid = grids[stage]
            # for grid in grids:
            total += compute_plane_smoothness(grid)
        return torch.as_tensor(total) * self.weight


class L1TimePlanes:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, multi_res_grids, stage):
        total = 0.0
        for grids in multi_res_grids:
            grid = grids[stage]
            # for grid in grids:
            total += torch.abs(1 - grid).mean()
        return torch.as_tensor(total) * self.weight


class DensityLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results, results_stage, batch):
        loss = 0

        if batch['stage_num'] == 0:
            mask = torch.where(
                (batch['semantic'] != results_stage['stage_num']) &
                (batch['semantic'] != 0)
            )[0]
            loss += self.weight * (torch.abs(
                results_stage['rgb'][mask] - results['rgb'][mask])).mean()
        else:
            mask = torch.where(
                (batch['semantic'] != results_stage['stage_num']) &
                (batch['semantic'] != batch['stage_num']) &
                (batch['semantic'] != 0)
            )[0]
            loss += self.weight * (torch.abs(
                results_stage['rgb'][mask] - results['rgb'][mask])).mean()

        loss += (1e-3 * Distortion.apply(results_stage['ws'], results_stage['deltas'],
                                         results_stage['ts'], results_stage['rays_a'])[mask]).mean()
        o = results_stage['opacity'][mask].clamp(1e-5, 1 - 1e-5)
        loss += (1e-3 * -(o * torch.log(o))).mean()
        return loss


class BDCLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results, batch):
        mask = torch.where(
            (batch['semantic'] != 0)
        )[0]
        ne_depth = NeDepth.apply(results['sigmas'], results['deltas'], results['ts'],
                                 results['rays_a'],
                                 results['vr_samples'])
        loss = (1e-3 * torch.abs(ne_depth[mask] - results['depth'][mask])).mean()
        return loss
