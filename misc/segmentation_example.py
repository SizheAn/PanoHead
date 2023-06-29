import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils


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

image = Image.open('image.jpg')
# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Apply the transformation to the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to('cuda:0')

# load segmentation net
seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to('cuda:0')
seg_net.requires_grad_(False)
seg_net.eval()

# 15 means human mask
mask0 = get_mask(seg_net, input_batch, 15).unsqueeze(1)

# Squeeze the tensor to remove unnecessary dimensions and convert to PIL Image
mask_squeezed = torch.squeeze(mask0)
mask_image = ToPILImage()(mask_squeezed)

# Save as PNG
mask_image.save("mask.png")
