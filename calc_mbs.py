# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils
from torch.nn import functional as F


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------
def get_mask(model, batch, cid):
    normalized_batch = transforms.functional.normalize(
        batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    # cid = sem_class_to_idx['car']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    boolean_car_masks = (normalized_masks.argmax(1) == cid)
    return boolean_car_masks.float()

def norm_ip(img, low, high):
    img_ = img.clamp(min=low, max=high)
    img_.sub_(low).div_(max(high - low, 1e-5))
    return img_


def norm_range(t, value_range=(-1, 1)):
    if value_range is not None:
        return norm_ip(t, value_range[0], value_range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))

#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--pose_cond', type=int, help='pose_cond angle', default=90, show_default=True)

def generate_images(
    network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float,
    reload_modules: bool,
    pose_cond: int,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:1')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new


    pose_cond_rad = pose_cond/180*np.pi
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    
    # load segmentation net
    seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to(device)
    seg_net.requires_grad_(False)
    seg_net.eval()

    mse_total = 0

    n_sample = 64
    batch = 32

    n_sample = n_sample // batch * batch
    batch_li = n_sample // batch * [batch]
    pose_cond_rad = pose_cond/180*np.pi

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    cam_pivot = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    c = conditioning_params.repeat(batch, 1)

    for batch in tqdm(batch_li):
        # z and w
        z0 = torch.from_numpy(np.random.randn(batch, G.z_dim)).to(device)
        z1 = torch.from_numpy(np.random.randn(batch, G.z_dim)).to(device)


        ws0 = G.mapping(z0, c, truncation_psi=0.7, truncation_cutoff=None)
        ws1 = G.mapping(z1, c, truncation_psi=0.7, truncation_cutoff=None)

        c0 = c.clone()
        c1 = c.clone()

        img0 = G.synthesis(ws0, c0, ws_bcg = ws0.clone())['image']
        img0 = norm_range(img0)
        img1 = G.synthesis(ws0, c1, ws_bcg = ws1.clone())['image']
        img1 = norm_range(img1)
        # 15 means human mask
        mask0 = get_mask(seg_net, img0, 15).unsqueeze(1)
        mask1 = get_mask(seg_net, img1, 15).unsqueeze(1)
        
        diff = torch.abs(mask0-mask1)
        mse = F.mse_loss(mask0, mask1)

        # mutual_bg_mask = (1-mask0) * (1-mask1)

        # diff = F.l1_loss(mutual_bg_mask*img1, mutual_bg_mask*img0, reduction='none')
        # diff = torch.where(diff < 1/255, torch.zeros_like(diff), torch.ones_like(diff))
        # diff = torch.sum(diff, dim=1)
        # diff = torch.where(diff < 1, torch.zeros_like(diff), torch.ones_like(diff))
        utils.save_image(
            # (1-mask1)*img1,
            mask1,
            f'alphamse/changebg/mask1.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            mask0,
            f'alphamse/changebg/mask0.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            img0,
            f'alphamse/changebg/img0.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            img1,
            f'alphamse/changebg/img1.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            diff,
            f'alphamse/changebg/diff.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        # sys.exit()

        # change_fg_score += torch.sum(torch.sum(diff, dim=(1,2)) / (torch.sum(mutual_bg_mask, dim=(1,2,3))+1e-8))
        mse_total += mse.cpu().detach().numpy()
        
    print(f'mse_final: {mse_total/len(batch_li)}')



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

