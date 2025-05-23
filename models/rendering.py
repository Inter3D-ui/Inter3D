import numpy as np
import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor == 0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive == 0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        sigmas[valid_mask], _rgbs = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        rgbs[valid_mask] = _rgbs.float()

        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)


        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples  # total samples for all rays

    rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg * rearrange(1 - opacity, 'n -> n 1')

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    stage = int(kwargs.get('stage'))
    share_grid = kwargs.get("share_grid", True)
    if share_grid:
        (rays_a, xyzs, dirs,
         results['deltas'], results['ts'], results['rm_samples']) = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    else:
        (rays_a, xyzs, dirs,
         results['deltas'], results['ts'], results['rm_samples']) = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield_lists[stage],
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)

    # for k, v in kwargs.items():  # supply additional inputs, repeated per ray
    #     if isinstance(v, torch.Tensor):
    #         kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    sigmas, rgbs = model(xyzs, dirs, **kwargs)

    (results['vr_samples'], results['opacity'],
     results['depth'], results['rgb'], results['ws']) = \
        VolumeRenderer.apply(sigmas, rgbs.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))

    results['rays_a'] = rays_a
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['sigmas'] = sigmas
    results['rgbs'] = rgbs.contiguous()
    results['xyzs'] = xyzs
    results['dirs'] = dirs

    if kwargs.get('random_bg', False):
        rgb_bg = torch.rand(3, device=rays_o.device)
    else:
        rgb_bg = torch.zeros(3, device=rays_o.device)
    results['rgb'] = results['rgb'] + \
                     rgb_bg * rearrange(1 - results['opacity'], 'n -> n 1')

    return results


def stage_render_rays_train(results, **kwargs):
    (results['vr_samples'], results['opacity'],
     results['depth'], results['rgb'], results['ws']) = \
        VolumeRenderer.apply(results['sigmas'], results['rgbs'].contiguous(),
                             results['deltas'], results['ts'],
                             results['rays_a'], kwargs.get('T_threshold', 1e-4))

    if kwargs.get('random_bg', False):
        rgb_bg = torch.rand(3, device=results['rays_a'].device)
    else:
        rgb_bg = torch.zeros(3, device=results['rays_a'].device)
    results['rgb'] = results['rgb'] + \
                     rgb_bg * rearrange(1 - results['opacity'], 'n -> n 1')
    return results
