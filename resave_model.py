# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import pickle

import click
import torch
import torch.nn.functional as F

import dnnlib
import legacy

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--output', 'output_pkl', help='Network pickle filename', required=True)
def main(
    network_pkl: str,
    output_pkl: str,
):
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        data = legacy.load_network_pkl(fp)
        data_new = {}
        for name in data.keys():
            module = data.get(name, None)
            if module is not None and isinstance(module, torch.nn.Module):
                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
            data_new[name] = module
            del module # conserve memory
        with open(output_pkl, 'wb') as f:
            pickle.dump(data_new, f)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------