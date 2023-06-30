""" Projecting input images into latent spaces. """

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import pickle

import dnnlib
import legacy

from camera_utils import LookAtPoseSampler

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    optimize_noise             = False,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples,1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt-w_avg).square().mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss +  reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f} w_norm: {w_norm_loss:<4.2f}  noise: {float(reg_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out


def project_pti(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    w_pivot: torch.Tensor,
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    for step in range(num_steps):
        # Learning rate schedule.
        # t = step / num_steps
        # lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        # lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        # lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        # lr = initial_learning_rate * lr_ramp
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # Synth images from opt_w.
        synth_images = G.synthesis(w_pivot, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        loss = 0.1 * mse_loss + perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f}')

        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())

    return out_params

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname',       help='Target image file to project to', required=True, metavar='FILE|DIR')
@click.option('--idx',                    help='index from dataset', type=int, default=0,  metavar='FILE|DIR')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--num-steps-pti',          help='Number of optimization steps for pivot tuning', type=int, default=350, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps',                    help='Frames per second of final video', default=30, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    idx: int,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    num_steps_pti: int,
    fps: int,
):
    """Project given image to the latent space of pretrained network pickle.
    Examples:
    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    G.rendering_kwargs["ray_start"] = 2.35

    if target_fname is not None:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=target_fname, use_labels=True, max_size=None, xflip=False)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        target_fname = dataset._path + "/" + dataset._image_fnames[idx]
        c = torch.from_numpy(dataset._get_raw_labels()[idx:idx+1]).to(device)
        print(f"projecting: [{idx}] {target_fname}")
        print(f"camera matrix: {c.shape}")
    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        c=c,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
    G_steps = project_pti(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        w_pivot=projected_w_steps[-1:],
        c=c,
        num_steps=num_steps_pti,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps[::2]:
            synth_image = G.synthesis(projected_w.unsqueeze(0).to(device), c=c, noise_mode='const')['image']
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        for G_new in G_steps:
            G_new.to(device)
            synth_image = G_new.synthesis(projected_w_steps[-1].unsqueeze(0).to(device), c=c, noise_mode='const')['image']
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            G_new.cpu()
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    G_final = G_steps[-1].to(device)
    synth_image = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c, noise_mode='const')['image']
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
        network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
        pickle.dump(network_data, f)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------