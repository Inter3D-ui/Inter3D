import os

import torch
import torch.nn.functional as F
import cv2

from tqdm import tqdm

from utils.ray_utils import *
from utils.color_utils import read_image
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import PIL.Image as Image

from datasets.base import BaseDataset
from einops import rearrange, repeat
import matplotlib.pyplot as plt

class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.stage_num = kwargs.get("stage_num", 0)
        self.read_intrinsics()
        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        intrinsics_path = os.path.join(self.root_dir, f'sparse/0/cameras.bin')
        camdata = read_cameras_binary(intrinsics_path)

        h = int(camdata[1].height * self.downsample)
        w = int(camdata[1].width * self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            cx = camdata[1].params[2] * self.downsample
            cy = camdata[1].params[3] * self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample < 1:  # mipnerf360 data
            folder = f'images_{int(1 / self.downsample)}'
        else:
            folder = 'images'

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]  # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses_all, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses_all[..., 3], axis=-1).min()
        self.poses_all[..., 3] /= scale
        self.pts3d /= scale

        self.rays_all = []
        self.semantics_all = []
        self.depth_all = []
        self.stages_all = []

        # self.zero_semantics = torch.tensor([])
        # self.zero_stages = torch.tensor([])
        # self.zero_rays = torch.tensor([])
        # self.zero_depth = torch.tensor([])
        # self.zero_poses = torch.tensor([])
        # self.zero_directions = torch.tensor([])

        if split == 'test':
            new_img_paths = []
            new_pose = []
            for pose, img_path in zip(self.poses_all, img_paths):
                if os.path.basename(img_path).startswith(str(int(0))):
                    # if not os.path.basename(img_path).startswith(str(int(0))):
                    new_img_paths.append(img_path)
                    new_pose.append(pose)
            self.poses_all = new_pose
            img_paths = new_img_paths

        self.poses_all = torch.FloatTensor(self.poses_all)  # (N_images, 3, 4)

        print(f'Loading {len(img_paths)} {split} images ...')
        stages_all = set()
        for i, img_path in enumerate(tqdm(img_paths)):
            buf = []  # buffer for ray attributes: rgb, etc
            semantic_buf = []
            depth_buf = []

            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]

            self.rays_all += [torch.cat(buf, 1)]

            sem_path = img_path.replace("images", "sem").replace("jpg", 'png')
            if os.path.exists(sem_path):
                semantic = Image.open(sem_path)
                semantic = np.array(semantic)
                semantic = cv2.resize(semantic, self.img_wh)
                semantic = rearrange(torch.LongTensor(semantic), "h w->(h w) 1")

            semantic_buf += [semantic]
            self.semantics_all += [torch.cat(semantic_buf, 1)]

            depth_path = os.path.join(self.root_dir, "depth", os.path.basename(img_path).split(".")[0] + ".png")
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                depth = np.array(depth)
                depth = cv2.resize(depth, self.img_wh)
                depth = rearrange(torch.LongTensor(depth), "h w->(h w) 1") / 255
            else:
                depth = torch.ones(self.img_wh[0] * self.img_wh[1], 1, device=img.device, dtype=torch.int64)

            depth_buf += [depth]
            self.depth_all += [torch.cat(depth_buf, 1)]

            path_stage = int(os.path.basename(img_path)[0])
            stages_all.add(path_stage)
            self.stages_all.append(len(stages_all) - 1)

            # if split != 'test' and path_stage == 0:
            #     pix_index = torch.where(semantic == 0)[0]
            #     self.zero_semantics = torch.cat([self.zero_semantics, semantic[pix_index]])
            #     self.zero_rays = torch.cat([self.zero_rays, img[pix_index]])
            #     self.zero_depth = torch.cat([self.zero_depth, depth[pix_index]])
            #     self.zero_poses = torch.cat(
            #         [self.zero_poses, repeat(self.poses_all[i], "h w->n h w", n=pix_index.shape[0])])
            #     self.zero_directions = torch.cat([self.zero_directions, self.directions[pix_index]])

        self.stages_all = torch.FloatTensor(self.stages_all)
        self.rays_all = torch.stack(self.rays_all)  # (N_images, hw, ?)
        self.semantics_all = torch.stack(self.semantics_all).squeeze(-1)  # (N_images, hw)
        self.depth_all = torch.stack(self.depth_all).squeeze(-1)  # (N_images, hw)

        # self.zero_semantics = self.zero_semantics.to(self.semantics_all.dtype)
        # self.zero_stages = self.zero_stages.to(self.stages_all.dtype)
        # self.zero_rays = self.zero_rays.to(self.rays_all.dtype)
        # self.zero_depth = self.zero_depth.to(self.depth_all.dtype)
        # self.zero_poses = self.zero_poses.to(self.poses_all.dtype)
        # self.zero_directions = self.zero_directions.to(self.directions.dtype)

        if split != "test":
            unique_stage, num_stage = torch.unique(self.stages_all, sorted=True, return_counts=True)
            self.unique_stage = unique_stage
            self.num_stage = num_stage

    def init_data(self, stage=[0]):
        index = torch.tensor([], dtype=torch.int64)
        for s in stage:
            stage_index = torch.where(self.stages_all == int(s))[0]
            index = torch.cat((index, stage_index), 0)

        self.stages = self.stages_all[index]
        self.rays = self.rays_all[index]
        self.semantics = self.semantics_all[index]
        self.depth = self.depth_all[index]
        self.poses = self.poses_all[index]



