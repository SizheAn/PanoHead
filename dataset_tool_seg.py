''' Tool for creating ZIP/PNG based datasets of image/mask pairs.
Code adapted from following paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."
See LICENSES/LICENSE_EG3D for original license.
'''

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(img_source_dir, seg_source_dir, *, use_basename: bool, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(img_source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Build path dictionary for segmentation masks
    seg_input_images = [str(f) for f in sorted(Path(seg_source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    seg_images_dict = {}
    for fname in seg_input_images:
        arch_fname = os.path.relpath(fname, seg_source_dir)
        arch_fname = arch_fname.replace('\\', '/')
        if use_basename:
            arch_fname = os.path.basename(arch_fname)
        arch_fname = os.path.splitext(arch_fname)[0] # ignore extension
        seg_images_dict[arch_fname] = fname

    # Load labels.
    labels = {}
    meta_fname = os.path.join(img_source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}
        print("original labels:", len(labels))

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        idx = 0 
        for _, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, img_source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            if use_basename:
                arch_fname = os.path.basename(arch_fname)
            img = np.array(PIL.Image.open(fname))
            if len(labels) > 0 and not arch_fname in labels:
                print("Label not found:", arch_fname, labels.get(arch_fname))
                continue # Ignore images without label
            if not os.path.splitext(arch_fname)[0] in seg_images_dict:
                print("Segmentation not found:", os.path.splitext(arch_fname)[0])
                continue # Ignore images without segmentation label
            else:
                seg = np.array(PIL.Image.open(seg_images_dict[os.path.splitext(arch_fname)[0]]))
            yield dict(img=img, seg=seg, label=labels.get(arch_fname), arch_fname=arch_fname)
            idx += 1
            if idx >= max_idx:
                break
    return max_idx, iterate_images()


def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resample: Optional[int],
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--img_source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--seg_source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--img_dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--seg_dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
@click.option('--use_basename', help='Use basename as identifier for labels and masks', default=False, metavar="BOOL")
@click.option('--ext', help='Output format', type=click.Choice(['png', 'jpg']), default="png")
def convert_dataset(
    ctx: click.Context,
    img_source: str,
    seg_source: str,
    img_dest: str,
    seg_dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    use_basename: Optional[bool],
    ext: Optional[str],
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    PIL.Image.init() # type: ignore

    if img_dest == '' or seg_dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_image_folder(img_source, seg_source, use_basename=use_basename, max_images=max_images)
    archive_root_dir_img, save_bytes_img, close_dest_img = open_dest(img_dest)
    archive_root_dir_seg, save_bytes_seg, close_dest_seg = open_dest(seg_dest)

    if resolution is None: resolution = (None, None)
    transform_img = make_transform(transform, *resolution, resample=PIL.Image.LANCZOS)
    transform_seg = make_transform(transform, *resolution, resample=PIL.Image.LANCZOS)

    dataset_attrs = None

    labels = []
    fnames = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        img_archive_fname = f'{idx_str[:5]}/img{idx_str}.{ext}'
        seg_archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_img(image['img'])
        seg = transform_seg(image['seg'])

        # Transform may drop images.
        if img is None or seg is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        img_channels = img.shape[2] if img.ndim == 3 else 1
        seg_channels = seg.shape[2] if seg.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'img_channels': img_channels,
            'seg_channels': seg_channels,
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['img_channels'] not in [1, 3, 4]:
                error('Input images must be stored as RGB or grayscale')
            if dataset_attrs['seg_channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {img_archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an target format
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB', 4: 'RGBA'}[img_channels])
        if img_channels == 4: img = img.convert('RGB')
        image_bits = io.BytesIO()
        if ext == "png":
            img.save(image_bits, format='png', compress_level=0, optimize=False)
        else:
            img.save(image_bits, format='jpeg', quality=100)
        save_bytes_img(os.path.join(archive_root_dir_img, img_archive_fname), image_bits.getbuffer())

        # Save the segmentation as an uncompressed PNG.
        seg = PIL.Image.fromarray(seg, { 1: 'L', 3: 'RGB'}[seg_channels])
        # if seg_channels == 4: seg = seg.convert('RGB')
        image_bits = io.BytesIO()
        seg.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes_seg(os.path.join(archive_root_dir_seg, seg_archive_fname), image_bits.getbuffer())

        # Append condition label
        labels.append([img_archive_fname, image['label']] if image['label'] is not None else None)
        fnames.append([img_archive_fname, image['arch_fname']])

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None,
        'fnames': fnames if all(x is not None for x in fnames) else None
    }
    save_bytes_img(os.path.join(archive_root_dir_img, 'dataset.json'), json.dumps(metadata))
    close_dest_img()
    close_dest_seg()
    print("# images: ", len(labels))
    print('labeled: ', metadata['labels'] is not None)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
