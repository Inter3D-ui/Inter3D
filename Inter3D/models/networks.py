import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange, repeat
from .custom_functions import TruncExp
import numpy as np
from .rendering import NEAR_DISTANCE
from models.field import init_grid_param, interpolate_ms_features


class NGP(nn.Module):
    def __init__(self, scale, stage_num, sem_num, share_grid):
        super().__init__()
        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.stage_num = stage_num
        self.sem_num = sem_num
        self.share_grid = share_grid
        if self.share_grid:
            self.register_buffer('density_bitfield',
                                 torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))
        else:
            self.register_buffer('density_bitfield_lists',
                                 torch.tensor(
                                     [torch.zeros(self.cascades * self.grid_size ** 3 // 8,
                                                  dtype=torch.uint8).numpy()] * self.stage_num))
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        # base model
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                }
            )
        self.sigma_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=16,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )
        # if self.sem_num != 0:
        # sem dir rgb
        self.semantic_net = \
            tcnn.Network(
                n_input_dims=16, n_output_dims=self.sem_num,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        self.concat_features = False
        self.out_dim = 32

        self.grids = nn.ModuleList()
        resolution = [64, 64, 64, self.stage_num - 1]
        # for res in [8]:
        resolution_res = [r * 8 for r in resolution[:3]] + resolution[3:]
        gp = init_grid_param(
            out_dim=self.out_dim,
            reso=resolution_res,
            uniform=False
        )
        self.grids.append(gp)
        self.grids_grad(False)

    def grids_grad(self, grad=False):
        for grid in self.grids:
            for param in grid.parameters():
                param.requires_grad = grad

    def density2grid(self, x, stages):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        feature = self.xyz_encoder(x)

        sigma_list = []
        h_list = []
        for stage in stages:
            if stage == 0:
                stage_feature = feature
            else:
                stage = stage - 1
                stage = stage / (self.stage_num - 2)
                stage_x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device) * stage], -1)
                stage_x = stage_x * 2 - 1
                stage_feature = interpolate_ms_features(pts=stage_x, ms_grids=self.grids,
                                                        concat_features=self.concat_features) * feature

            h = self.sigma_net(stage_feature)
            sigmas = TruncExp.apply(h[:, 0])
            sigmas = torch.where(sigmas == torch.inf,
                                 torch.tensor([6666], device=x.device, dtype=sigmas.dtype),
                                 sigmas)
            sigma_list.append(sigmas)
            h_list.append(h)

        sigma_list = torch.stack(sigma_list).to(x.device)
        sigma_list = rearrange(sigma_list, "n m->m n")

        if len(stages) == 1:
            max_index = torch.ones(x.shape[0], device=x.device, dtype=torch.int64) * stages[0]
        else:
            max_index = torch.argmax(sigma_list, dim=-1)
        sigmas = torch.gather(sigma_list, -1, max_index.unsqueeze(-1)).squeeze(-1)
        return sigmas

    def density_all(self, x, return_feat=False, stages=[0]):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        feature = self.xyz_encoder(x)
        h = self.sigma_net(feature)
        sigmas = TruncExp.apply(h[:, 0])
        sigma_list = [sigmas]
        h_list = [h]
        for stage in stages:
            if stage == 0:
                continue
            else:
                stage = stage - 1
                stage = stage / (self.stage_num - 2)
                stage_x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device) * stage], -1)
                stage_x = stage_x * 2 - 1
                stage_feature = interpolate_ms_features(pts=stage_x, ms_grids=self.grids,
                                                        concat_features=self.concat_features) * feature

            h = self.sigma_net(stage_feature)
            sigmas = TruncExp.apply(h[:, 0])
            sigmas = torch.where(sigmas == torch.inf,
                                 torch.tensor([6666], device=x.device, dtype=sigmas.dtype),
                                 sigmas)
            sigma_list.append(sigmas)
            h_list.append(h)

        sigma_list = torch.stack(sigma_list).to(x.device)
        delta_sigma = rearrange(torch.abs(sigma_list[1:] - sigma_list[0]), "c n->n c")
        max_index = torch.argmax(delta_sigma, dim=-1) + 1

        sigma_list = rearrange(sigma_list, "n m->m n")
        sigmas = torch.gather(sigma_list, -1, max_index.unsqueeze(-1)).squeeze(-1)
        if return_feat:
            h_list = rearrange(torch.stack(h_list).to(x.device), "n m c->m c n")
            h = torch.gather(h_list, -1, max_index.unsqueeze(-1).unsqueeze(-1).expand(-1, h_list.shape[1], -1)).squeeze(
                -1)
            return sigmas, h
        else:
            return sigmas

    def density(self, x, stage=0, return_feat=False):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        if int(stage) == 0:
            feature = self.xyz_encoder(x)
        else:
            base_feature = self.xyz_encoder(x)

            stage = stage - 1
            stage = stage / (self.stage_num - 2)
            stage_x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device) * stage], -1)
            stage_x = stage_x * 2 - 1
            feature = interpolate_ms_features(pts=stage_x, ms_grids=self.grids,
                                              concat_features=self.concat_features) * base_feature

        h = self.sigma_net(feature)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat:
            return sigmas, h
        return sigmas

    def forward(self, x, d, **kwargs):
        stage = kwargs.get("stage", 0)
        stages = kwargs.get("stages", None)
        if stages and len(stages) == 1:
            stage = stages[0]
            stages = None
        if stages:
            sigmas, h = self.density_all(x, return_feat=True, stages=stages)
        else:
            sigmas, h = self.density(x, stage=stage, return_feat=True)
        semantics = self.semantic_net(h)
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))
        return sigmas, rgbs, semantics

    @torch.no_grad()
    def get_all_cells(self):
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades
        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold, stage):
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.grid_coords.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            if self.share_grid:
                indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            else:
                indices2 = torch.nonzero(self.density_grid_lists[stage, c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.grid_coords.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64 ** 3, stage=0):
        N_cams = poses.shape[0]
        if self.share_grid:
            self.count_grid = torch.zeros_like(self.density_grid)
        else:
            self.count_grid = torch.zeros_like(self.density_grid_lists[stage])
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2 ** (c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2] >= 0) & \
                           (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) & \
                           (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
                covered_by_cam = (uvd[:, 2] >= NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i + chunk]] = \
                    count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (uvd[:, 2] < NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                if self.share_grid:
                    self.density_grid[c, indices[i:i + chunk]] = \
                        torch.where(valid_mask, 0.0, -1.0)
                else:
                    self.density_grid_lists[stage, c, indices[i:i + chunk]] = \
                        torch.where(valid_mask, 0.0, -1.0)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False, stage=0, stages=[0]):
        if self.share_grid:
            density_grid_tmp = torch.zeros_like(self.density_grid)
        else:
            density_grid_tmp = torch.zeros_like(self.density_grid_lists[stage])
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                           density_threshold, stage)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            if self.share_grid:
                density_grid_tmp[c, indices] = self.density2grid(xyzs_w, stages=stages)
            else:
                density_grid_tmp[c, indices] = self.density(xyzs_w, stage=stage, return_feat=False)

        # if erode:
        #     # My own logic. decay more the cells that are visible to few cameras
        #     decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)

        if self.share_grid:
            self.density_grid = \
                torch.where(self.density_grid < 0,
                            self.density_grid,
                            torch.maximum(self.density_grid * decay, density_grid_tmp))

            mean_density = self.density_grid[self.density_grid > 0].mean().item()

            vren.packbits(self.density_grid, min(mean_density, density_threshold),
                          self.density_bitfield)
        else:
            self.density_grid_lists[stage] = \
                torch.where(self.density_grid_lists[stage] < 0,
                            self.density_grid_lists[stage],
                            torch.maximum(self.density_grid_lists[stage] * decay, density_grid_tmp))

            mean_density = self.density_grid_lists[stage, self.density_grid_lists[stage] > 0].mean().item()

            vren.packbits(self.density_grid_lists[stage], min(mean_density, density_threshold),
                          self.density_bitfield_lists[stage])

    @torch.no_grad()
    def init_density_grid(self, stages):
        if len(stages) == 1:
            self.density_bitfield = self.density_bitfield_lists[int(stages[0])]
        else:
            density_bitfield_lists = self.density_bitfield_lists
            delta_bitfield = rearrange(torch.abs(density_bitfield_lists[stages] - density_bitfield_lists[0]),
                                       "c n->n c")
            max_index = torch.argmax(delta_bitfield, dim=-1)
            density_bitfield_lists = rearrange(density_bitfield_lists[stages], "n m->m n")
            self.density_bitfield = torch.gather(density_bitfield_lists, -1, max_index.unsqueeze(-1)).squeeze(-1)

    def update_delta_density_grid(self, density_threshold, stage):
        if self.share_grid:
            density_grid_tmp = repeat(torch.zeros_like(self.density_grid), "n c->m n c", m=2)
        else:
            density_grid_tmp = repeat(torch.zeros_like(self.density_grid_lists[stage]), "n c->m n c", m=2)

        # infer sigmas
        for c in range(self.cascades):
            if self.share_grid:
                indices = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            else:
                indices = torch.nonzero(self.density_grid_lists[stage, c] > density_threshold)[:, 0]
            if len(indices) > 0:
                rand_idx = torch.randint(len(indices), (self.grid_size ** 3 // 4,),
                                         device=self.grid_coords.device)
                indices = indices[rand_idx]
                coords = vren.morton3D_invert(indices.int())
            else:
                coords = torch.randint(self.grid_size, (self.grid_size ** 3 // 4, 3), dtype=torch.int32,
                                       device=self.grid_coords.device)
                indices = vren.morton3D(coords).long()

            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size

            density_grid_tmp[0, c, indices] = self.density(xyzs_w, stage=0, return_feat=False)
            density_grid_tmp[1, c, indices] = self.density(xyzs_w, stage=stage, return_feat=False)

        delta = torch.abs(density_grid_tmp[1] - density_grid_tmp[0])
        sum_delta = torch.sum(delta, -1).unsqueeze(-1)
        p_density = delta / (sum_delta + 1e-10)

        sum_density = density_grid_tmp.sum(-1).unsqueeze(-1) + 1e-10
        q_density = density_grid_tmp / (sum_density + 1e-10)

        return (torch.mean(-1 * p_density * torch.log(p_density + 1e-10)),
                torch.mean(q_density * torch.log(q_density + 1e-10) + 0.16))
