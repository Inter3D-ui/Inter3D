import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import random


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.stage_one = True

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            if self.stages[-1] == self.stage_num - 1:
                stage_num = torch.randint(0, self.stage_num, (1,)).item()
            else:
                stage_num = 0

            index = torch.where(self.stages == int(stage_num))[0]
            img_idxs = np.random.choice(index, self.batch_size)


            pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
            stage = self.stages[img_idxs]
            rays = self.rays[img_idxs, pix_idxs]
            semantics = self.semantics[img_idxs, pix_idxs]
            depth = self.depth[img_idxs, pix_idxs]
            poses = self.poses[img_idxs]
            directions = self.directions[pix_idxs]

            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3],
                      'semantic': semantics,
                      'depth': depth,
                      "pose": poses,
                      "direction": directions,
                      'stage': stage,
                      "stage_num": stage_num
                      }
        else:
            stage = self.stages[idx]
            sample = {'pose': self.poses[idx], 'img_idxs': idx, "direction": self.directions, 'stage': stage}
            if len(self.rays) > 0:  # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
        return sample
