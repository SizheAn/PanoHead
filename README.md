## PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°<br>
<a href="https://arxiv.org/abs/2303.13071"><img src="https://img.shields.io/badge/arXiv-2303.13071-b31b1b" height=22.5></a>
<a href="https://creativecommons.org/licenses/by/4.0"><img src="https://img.shields.io/badge/LICENSE-CC--BY--4.0-yellow" height=22.5></a>
<a href="https://www.youtube.com/watch?v=Y8NXiBOEWoE"><img src="https://img.shields.io/static/v1?label=CVPR 2023&message=8 Minute Video&color=red" height=22.5></a>  



![Teaser image](./misc/teaser.png)

**PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°**<br>
Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y. Ogras, Linjie Luo
<br>https://sizhean.github.io/panohead<br>

Abstract: *Synthesis and reconstruction of 3D human head has gained increasing interests in computer vision and computer graphics recently. Existing state-of-the-art 3D generative adversarial networks (GANs) for 3D human head synthesis are either limited to near-frontal views or hard to preserve 3D consistency in large view angles. We propose PanoHead, the first 3D-aware generative model that enables high-quality view-consistent image synthesis of full heads in 360° with diverse appearance and detailed geometry using only in-the-wild unstructured images for training. At its core, we lift up the representation power of recent 3D GANs and bridge the data alignment gap when training from in-the-wild images with widely distributed views. Specifically, we propose a novel two-stage self-adaptive image alignment for robust 3D GAN training. We further introduce a tri-grid neural volume representation that effectively addresses front-face and back-head feature entanglement rooted in the widely-adopted tri-plane formulation. Our method instills prior knowledge of 2D image segmentation in adversarial learning of 3D neural scene structures, enabling compositable head synthesis in diverse backgrounds. Benefiting from these designs, our method significantly outperforms previous 3D GANs, generating high-quality 3D heads with accurate geometry and diverse appearances, even with long wavy and afro hairstyles, renderable from arbitrary poses. Furthermore, we show that our system can reconstruct full 3D heads from single input images for personalized realistic 3D avatars.*


## Requirements

* We recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later.  (Why is a separate CUDA toolkit installation required?  We use the custom CUDA extensions from the StyleGAN3 repo. Please see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd PanoHead`
  - `conda env create -f environment.yml`
  - `conda activate panohead`


## Getting started

Download the whole `models` folder from [link](https://drive.google.com/drive/folders/1m517-F1NCTGA159dePs5R5qj02svtX1_?usp=sharing) and put it under the root dir.

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames.


## Generating results

```.bash
# Generate videos using pre-trained model

python gen_videos.py --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
--seeds 0-3 --grid 2x2 --outdir=out --cfg Head --trunc 0.7

```

```.bash
# Generate images and shapes (as .mrc files) using pre-trained model

python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 \
    --network models/easy-khair-180-gpc0.8-trans10-025000.pkl
```

## Applications
```.bash
# Generate full head reconstruction from a single RGB image.
# Please refer to ./gen_pti_script.sh
# For this application we need to specify dataset folder instead of zip files.
# Segmentation files are not necessary for PTI inversion.

./gen_pti_script.sh
```

```.bash
# Generate full head interpolation from two seeds.
# Please refer to ./gen_interpolation.py for the implementation

python gen_interpolation.py --network models/easy-khair-180-gpc0.8-trans10-025000.pkl\
        --trunc 0.7 --outdir interpolation_out
```



## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('*.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # camera parameters
img = G(z, c)['image']                           # NCHW, float32, dynamic range [-1, +1], no truncation
mask = G(z, c)['image_mask']                    # NHW, int8, [0,255]
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.



## Datasets

FFHQ-F(ullhead) consists of [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset), [K-Hairstyle dataset](https://psh01087.github.io/K-Hairstyle/), and an in-house human head dataset. For head pose estimation, we use [WHENet](https://arxiv.org/abs/2005.10353).

Due to the license issue, we are not able to release FFHQ-F dataset that we used to train the model. [test_data_img](./dataset/testdata_img/) and [test_data_seg](./dataset/testdata_seg/) are just an example for showing the dataset struture. For the camera pose convention, please refer to [EG3D](https://github.com/NVlabs/eg3d). 


## Datasets format
For training purpose, we can use either zip files or normal folder for image dataset and segmentation dataset. For PTI, we need to use folder.

To compress dataset folder to zip file, we can use [dataset_tool_seg](./dataset_tool_seg.py). 

For example:
```.bash
python dataset_tool_seg.py --img_source dataset/testdata_img --seg_source  dataset/testdata_seg --img_dest dataset/testdata_img.zip --seg_dest dataset/testdata_seg.zip --resolution 512x512
```

## Obtaining camera pose and cropping the images
Please follow the [guide](3DDFA_V2_cropping/cropping_guide.md)

## Obtaining segmentation masks
You can try using deeplabv3 or other off-the-shelf tool to generate the masks. For example, using deeplabv3: [misc/segmentation_example.py](misc/segmentation_example.py)




## Training

Examples of training using `train.py`:

```
# Train with StyleGAN2 backbone from scratch with raw neural rendering resolution=64, using 8 GPUs.
# with segmentation mask, trigrid_depth@3, self-adaptive camera pose loss regularizer@10

python train.py --outdir training-runs  --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7


# Second stage finetuning to 128 neural rendering resolution (optional).

python train.py --outdir results --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--resume=~/training-runs/experiment_dir/network-snapshot-025000.pkl\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7\\
--neural_rendering_resolution_final=128 --resume_kimg 1000
```

## Metrics



```.bash
./get_metrics.sh
```
There are three evaluation modes: all, front, and back as we mentioned in the paper. Please refer to [cal_metrics.py](./calc_metrics.py) for the implementation.


## Citation

If you find our repo helpful, please cite our paper using the following bib:

```
@InProceedings{An_2023_CVPR,
    author    = {An, Sizhe and Xu, Hongyi and Shi, Yichun and Song, Guoxian and Ogras, Umit Y. and Luo, Linjie},
    title     = {PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360deg},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20950-20959}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

We thank Shuhong Chen for the discussion during Sizhe's internship.

This repo is heavily based off the [NVlabs/eg3d](https://github.com/NVlabs/eg3d) repo; Huge thanks to the EG3D authors for releasing their code!