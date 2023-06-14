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
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def generate_images(
    network_pkl: str,
    outdir: str,
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float,
    reload_modules: bool,
    pose_cond: int,
):
    """Generate interpolation images using pretrained network pickle.

    Examples:

    \b
    python gen_interpolation.py --network models/easy-khair-180-gpc0.8-trans10-025000.pkl\
          --trunc 0.7 --outdir interpolation_out
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

    network_pkl = os.path.basename(network_pkl)
    outdir = os.path.join(outdir, os.path.splitext(network_pkl)[0] + '_' + str(pose_cond))
    os.makedirs(outdir, exist_ok=True)

    pose_cond_rad = pose_cond/180*np.pi
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)



    # n_sample = 64
    # batch = 32

    # n_sample = n_sample // batch * batch
    # batch_li = n_sample // batch * [batch]
    pose_cond_rad = pose_cond/180*np.pi

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    cam_pivot = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    conditioning_cam2world_pose_back = LookAtPoseSampler.sample(-pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params_back = torch.cat([conditioning_cam2world_pose_back.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

   
    conditioning_cam2world_pose_side = LookAtPoseSampler.sample(45/180*np.pi, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params_side = torch.cat([conditioning_cam2world_pose_side.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) 

    # set two random seeds for interpolation 
    seed1, seed2 = 521, 329
    z0 = torch.from_numpy(np.random.RandomState(seed1).randn(G.z_dim).reshape(1,G.z_dim)).to(device)
    z1 = torch.from_numpy(np.random.RandomState(seed2).randn(G.z_dim).reshape(1,G.z_dim)).to(device)   
    c = conditioning_params

    ws0 = G.mapping(z0, c, truncation_psi=0.7, truncation_cutoff=None)
    ws1 = G.mapping(z1, c, truncation_psi=0.7, truncation_cutoff=None)

    image_final = []
    for c in [conditioning_params, conditioning_params_side, conditioning_params_back]:
        img0 = G.synthesis(ws0, c)['image']
        img0 = norm_range(img0)
        img1 = G.synthesis(ws1, c)['image']
        img1 = norm_range(img1)





        img_list = []
        for interpolation_idx in [0,2,3,4,6,8]:
        # for interpolation_idx in range(0,14,1):
            # interpolation_idx = 8
            ws_new = ws0.clone()
            ws_new[:, interpolation_idx:, :] = ws1[:, interpolation_idx:, :]
            img_new = G.synthesis(ws_new, c)['image']
            img_new = norm_range(img_new)
            img_list.append(img_new)

        img_list.append(img0)
        
        img_new = torch.cat(img_list, dim=0)
        image_final.append(img_new)

    image_final = torch.cat(image_final, dim=2)
    
    utils.save_image(
        image_final,
        os.path.join(outdir, f'img_interpolation_seed{seed1}_{seed2}.png'),
        # nrow=8,
        normalize=True,
        range=(0, 1),
        padding=0,
    )
        # utils.save_image(
        #     diff,
        #     f'alphamse/changebg/diff.png',
        #     nrow=8,
        #     normalize=True,
        #     range=(0, 1),
        #     padding=0,
        # )
        # sys.exit()


        





#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

