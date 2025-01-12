import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange, repeat, reduce

# data
from torch.utils.data import DataLoader
from datasets.colmap import ColmapDataset
from utils.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES, stage_render_rays_train

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses.regularization import CompositeLoss, DistortionLoss, OpacityLoss, SemanticLoss, L1TimePlanes, \
    TimeSmoothness, DensityLoss, BDCLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils.utils import slim_ckpt, load_ckpt
import imgviz
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import copy

import warnings

warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)

        self.model = NGP(scale=hparams.scale, stage_num=hparams.stage_num,
                         sem_num=hparams.sem_num,
                         # sem_num=hparams.sem_num if hparams.stage_end_epoch == hparams.num_epochs else 0,
                         share_grid=hparams.share_grid)
        G = self.model.grid_size
        if hparams.share_grid:
            self.model.register_buffer('density_grid',
                                       torch.zeros(self.model.cascades, G ** 3))
        else:
            self.model.register_buffer('density_grid_lists',
                                       torch.tensor(
                                           [torch.zeros(self.model.cascades, G ** 3).numpy()] * self.hparams.stage_num))

        self.model.register_buffer('grid_coords',
                                   create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split, stage, stages):
        poses = batch['pose']
        directions = batch["direction"]
        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split != 'train',
                  'share_grid': self.hparams.share_grid,
                  'random_bg': self.hparams.random_bg,
                  'stage': stage,
                  'stages': stages}

        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        self.CompositeLoss = CompositeLoss(self.hparams.composite_weight)
        self.OpacityLoss = OpacityLoss(self.hparams.opacity_weight)
        self.DistortionLoss = DistortionLoss(self.hparams.distortion_weight)
        self.SemanticLoss = SemanticLoss(self.hparams.semantic_weight)
        self.L1TimePlanesLoss = L1TimePlanes(self.hparams.l1TimePlanes_weight)
        self.TimeSmoothnessLoss = TimeSmoothness(self.hparams.timeSmoothness_weight)
        self.DensityLoss = DensityLoss(self.hparams.density_weight)
        self.BDCLoss = BDCLoss(self.hparams.bdc_weight)

        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'dir_name': self.hparams.dir_name,
                  'stage_num': self.hparams.stage_num
                  }
        self.train_dataset = ColmapDataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        self.train_dataset.init_data(stage=[0])

        self.test_dataset = ColmapDataset(split='test', **kwargs)
        self.test_dataset.init_data(stage=[0])

    def configure_optimizers(self):
        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]

        return opts

    def train_dataloader(self):
        if self.current_epoch // self.hparams.stage_end_epoch == 1:
            self.model.grids_grad(True)
            self.train_dataset.init_data(stage=[i for i in range(self.hparams.stage_num)])
            if not self.hparams.share_grid:
                for i in range(self.hparams.stage_num):
                    # index = torch.where(self.train_dataset.stages == int(i))
                    # poses = self.train_dataset.poses[index].to(self.device)
                    # self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                    #                                 poses,
                    #                                 self.train_dataset.img_wh,
                    #                                 stage=int(i))
                    self.model.density_bitfield_lists[int(i)] = torch.where(
                        self.model.density_bitfield_lists[0] > 0,
                        self.model.density_bitfield_lists[0],
                        self.model.density_bitfield_lists[int(i)])
                    self.model.density_grid_lists[int(i)] = torch.where(self.model.density_grid_lists[int(i)] == 0,
                                                                        self.model.density_grid_lists[0],
                                                                        self.model.density_grid_lists[int(i)])
        return DataLoader(self.train_dataset,
                          num_workers=1,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=1,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        # if self.hparams.share_grid:
        #     self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                     self.train_dataset.poses_all.to(self.device),
        #                                     self.train_dataset.img_wh)
        # else:
        #     self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                     self.train_dataset.poses.to(self.device),
        #                                     self.train_dataset.img_wh,
        #                                     stage=int(0))
        pass

    def training_step(self, batch, batch_nb, *args):
        loss = 0
        # unique_stage, num_stage = torch.unique(batch['stage'], sorted=True, return_counts=True)
        if self.current_epoch >= self.hparams.stage_end_epoch:
            if self.hparams.share_grid:
                if self.global_step % self.update_interval == 0:
                    stages = [i for i in range(self.hparams.stage_num)]
                    self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                                   warmup=False,
                                                   decay=1,
                                                   erode=True,
                                                   stages=stages
                                                   )
            else:
                for i in range(self.hparams.stage_num):
                    if self.global_step % (self.update_interval + i) == 0:
                        self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                                       warmup=False,
                                                       decay=0.95,
                                                       erode=True,
                                                       stage=int(i)
                                                       )
        else:
            if self.global_step % self.update_interval == 0:
                self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                               warmup=self.global_step < self.warmup_steps,
                                               decay=0.95,
                                               erode=True,
                                               stage=int(0),
                                               stages=[int(0)],
                                               )

        # ------------------bg------------------
        results = self.forward(batch, split='train', stage=batch['stage_num'], stages=None)
        loss += self.CompositeLoss.apply(results, batch)
        loss += self.DistortionLoss.apply(results)
        loss += self.OpacityLoss.apply(results)
        # loss += self.SemanticLoss.apply(results, batch)
        # loss += self.BDCLoss.apply(results, batch)

        if self.current_epoch // self.hparams.stage_end_epoch >= 1:
            base_num = batch['stage_num']
            kwargs = {'test_time': False,
                      'share_grid': self.hparams.share_grid,
                      'random_bg': self.hparams.random_bg}
            if base_num == 0:
                stage_num = int(np.random.choice([i for i in range(1, self.hparams.stage_num)], 1)[0])
            else:
                stage_num = 0
            kwargs['stage'] = stage_num
            sigmas, rgbs, semantics = self.model.forward(results['xyzs'].detach(), results['dirs'].detach(),
                                                         **kwargs)

            results_stage = {
                'deltas': results["deltas"].detach(),
                'ts': results["ts"].detach(),
                'rays_a': results['rays_a'].detach(),
                'sigmas': sigmas,
                'rgbs': rgbs,
                'semantics': semantics
            }
            results_stage = stage_render_rays_train(results=results_stage, **kwargs)
            results_stage['stage_num'] = stage_num
            loss += self.DensityLoss.apply(results, results_stage, batch)

            # ------------------stage grid------------------
            loss += self.L1TimePlanesLoss.apply(self.model.grids, batch['stage_num'] - 1)
            loss += self.TimeSmoothnessLoss.apply(self.model.grids, batch['stage_num'] - 1)


        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples'] / len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples'].sum() / len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        self.col = imgviz.label_colormap()
        self.val_dir = f'results/{self.hparams.exp_name}'
        os.makedirs(self.val_dir, exist_ok=True)
        if not self.hparams.share_grid:
            stages = [i for i in range(self.hparams.stage_num)]
            # stages = [1]
            self.model.init_density_grid(stages)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        stages = [i for i in range(self.hparams.stage_num)]
        # stages = [1]
        results = self.forward(batch, split='test', stage=None, stages=stages)

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.test_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()

        idx = batch['img_idxs']
        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        rgb_pred = (rgb_pred * 255).astype(np.uint8)
        depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
        imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        # if self.hparams.stage_end_epoch == self.hparams.num_epochs:
        # semantic_pred = rearrange(results['semantic'].cpu(), '(h w) c-> h w c', h=h)
        # semantic_pred = torch.argmax(semantic_pred, dim=-1).numpy()
        # semantic_pred = Image.fromarray(semantic_pred.astype(np.uint8), 'P')
        # semantic_pred.putpalette(self.col.flatten())
        # semantic_pred.save(os.path.join(self.val_dir, f'{idx:03d}_s.png'))

        if idx == 0:
            plt.imshow(rgb_pred)
            plt.show()
        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim, True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False)
                      if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      reload_dataloaders_every_n_epochs=hparams.stage_end_epoch,  # 每5个epoch 重新加载dataload
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only:  # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs - 1}.ckpt')
        torch.save(ckpt_, f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs - 1}_slim.ckpt')

    # imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
    # imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
    #                 [imageio.imread(img) for img in imgs[::2]],
    #                 fps=30, macro_block_size=1)
    # imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
    #                 [imageio.imread(img) for img in imgs[1::2]],
    #                 fps=30, macro_block_size=1)

    imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
    rgb_list = []
    depth_list = []
    sem_list = []
    for img_path in imgs:
        img = imageio.imread(img_path)
        if '_d.png' in img_path:
            depth_list.append(img)
        elif '_s.png' in img_path:
            sem_list.append(img)
        else:
            rgb_list.append(img)
    imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'), rgb_list, fps=30, macro_block_size=1)
    imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'), depth_list, fps=30, macro_block_size=1)
    # if hparams.stage_end_epoch == hparams.num_epochs:
    # imageio.mimsave(os.path.join(system.val_dir, 'sem.mp4'), sem_list, fps=30, macro_block_size=1)
