import itertools

import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.nn import functional as F
from einops import repeat, rearrange, reduce

def init_grid_param(
        out_dim,  # 32
        reso,
        uniform=False,
        a=0.1,
        b=0.5):  # [64,64,64,5]
    grid_coefs = nn.ParameterList()
    for res in reso[:-1]:
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [res, reso[-1]]
        ))
        if uniform:  # Initialize time planes to 1
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        else:
            nn.init.ones_(new_grid_coef)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts,
                            ms_grids,
                            concat_features
                            ):
    # if pts.shape[-1] == 3:
    #     coo_combs = list(itertools.combinations(
    #         range(pts.shape[-1]), 2)
    #     )
    # else:
    coo_combs = [(dimension, pts.shape[-1] - 1) for dimension in range(pts.shape[-1]) if dimension != pts.shape[-1] - 1]
    multi_scale_interp = [] if concat_features else 0
    grid: nn.ParameterList
    for grid in ms_grids:
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


def grid_sample_wrapper(grid, coords, align_corners=False):
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))

    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        # mode='bilinear',
        mode='nearest',
        padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


