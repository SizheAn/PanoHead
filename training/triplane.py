''' Updated tri-plane (Tri-grid) generator
Code adapted from following paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."
See LICENSES/LICENSE_EG3D for original license.
'''

import math
import torch
from torch_utils import persistence
from training.networks_stylegan2 import ToRGBLayer, SynthesisNetwork, MappingNetwork
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        bcg_synthesis_kwargs = synthesis_kwargs.copy()
        bcg_synthesis_kwargs["channel_base"] = bcg_synthesis_kwargs["channel_base"] // 2
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=rendering_kwargs['triplane_size'], img_channels=32*3*rendering_kwargs['triplane_depth'], mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'decoder_activation': rendering_kwargs['decoder_activation']})
        self.torgb = ToRGBLayer(32, 3, w_dim) if rendering_kwargs.get('use_torgb_raw', False) else None
        self.bcg_synthesis = SynthesisNetwork(w_dim, img_resolution=self.superresolution.input_resolution, img_channels=32, **bcg_synthesis_kwargs) if rendering_kwargs.get('use_background', False) else None
        # New mapping network for self-adaptive camera pose, dim = 3
        self.t_mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=3, num_ws=1, last_activation='linear', lr_multiplier=1.0, **mapping_kwargs)
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    # camera pose self_adaptation mapping, return c after added t_vector
    def apply_delta_c(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        '''
        Input:
        z: latent code z
        c: latent code c

        Output:
        c_new: latent code c after adding the delta
        delta_c: delta_c from z and c_cam
        '''
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        delta_c = self.t_mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        delta_c = torch.squeeze(delta_c, 1)
        # delta_c = delta_c * 0.1 # scale for better initialization
        # index 3, 7, 11 are the translation index in extrinsic
        c_new = c.clone()
        c_new[:,3] += delta_c[:,0]
        c_new[:,7] += delta_c[:,1]
        c_new[:,11] += delta_c[:,2]

        return c_new, delta_c

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, 
            ws_bcg=None, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three D*32-channel planes, where D=self.rendering_kwargs['triplane_depth'], defines the depth of the tri-grid
        planes = planes.view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        if self.decoder.activation == "sigmoid":
            feature_image = feature_image * 2 - 1 # Scale to (-1, 1), taken from ray marcher
        # Generate Background
        if self.bcg_synthesis:
            ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]
            if ws_bcg.size(1) < self.bcg_synthesis.num_ws:
                ws_bcg = torch.cat([ws_bcg, ws_bcg[:,-1:].repeat(1,self.bcg_synthesis.num_ws-ws_bcg.size(1),1)], 1)
            bcg_image = self.bcg_synthesis(ws_bcg, update_emas=update_emas, **synthesis_kwargs)
            bcg_image = torch.nn.functional.interpolate(bcg_image, size=feature_image.shape[2:],
                    mode='bilinear', align_corners=False, antialias=self.rendering_kwargs['sr_antialias'])
            feature_image = feature_image + (1-weights_samples) * bcg_image

        # Generate Raw image
        if self.torgb:
            rgb_image = self.torgb(feature_image, ws[:,-1], fused_modconv=False)
            rgb_image = rgb_image.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        else:
            rgb_image = feature_image[:, :3]
        # Run superresolution to get final image
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        mask_image = weights_samples*(1 + 2*0.001) - 0.001
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, "image_mask": mask_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # Obtain the new c with learned offset
        c, delta_c = self.apply_delta_c(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

@persistence.persistent_class
class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        self.activation = options['decoder_activation']
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = x[..., 1:]
        sigma = x[..., 0:1]
        if self.activation == "sigmoid":
            # Original EG3D
            rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style, use with toRGB
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2, inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}
