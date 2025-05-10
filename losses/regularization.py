import torch.optim.lr_scheduler
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from .base_regularization import Distortion, compute_plane_smoothness
from typing import Sequence
from einops import rearrange, repeat


class CompositeLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def apply(self, results, batch):
        return self.weight * (torch.pow(results['rgb'] - batch['rgb'], 2)).mean()



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
            for grid in grids:
                grid_ = grid[:, :, :, stage]
                total += compute_plane_smoothness(grid_)
        return torch.as_tensor(total) * self.weight


class L1TimePlanes:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, multi_res_grids, stage):
        total = 0.0
        for grids in multi_res_grids:
            for grid in grids:
                grid_ = grid[:, :, :, stage]
                total += torch.abs(1 - grid_).mean()
        return torch.as_tensor(total) * self.weight


class DensityLoss:
    def __init__(self, weight=1e-3):
        self.weight = weight

    def apply(self, results, results_stage, batch):
        loss = 0

        if batch['stage_num'] == 0:
            mask = torch.where(
                (batch['semantic'] != results_stage['stage_num'])
                & (batch['semantic'] != 0)
            )[0]
            loss += self.weight * (torch.abs(
                results_stage['rgb'][mask] - results['rgb'][mask])).mean()
            loss += 1e-1 * (torch.abs(
                results_stage['depth'][mask] - results['depth'][mask])).mean()
        else:
            mask = torch.where(
                # (batch['semantic'] != results_stage['stage_num']) &
                # (batch['semantic'] != batch['stage_num']) &
                (batch['semantic'] != 0)
            )[0]
            loss += self.weight * (torch.abs(
                results_stage['rgb'][mask] - results['rgb'][mask])).mean()
            loss += 1e-1 * (torch.abs(
                results_stage['depth'][mask] - results['depth'][mask])).mean()

        loss += (1e-3 * Distortion.apply(results_stage['ws'], results_stage['deltas'],
                                         results_stage['ts'], results_stage['rays_a'])[mask]).mean()
        o = results_stage['opacity'][mask].clamp(1e-5, 1 - 1e-5)
        loss += (1e-3 * -(o * torch.log(o))).mean()
        return loss

