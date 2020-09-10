import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np


def tensor_to_image(t, tensor_out = False):

    t = t.squeeze(dim=0)
    t = t.detach()
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    t = t.permute(1, 2, 0)
    mean = torch.reshape(mean, (1, 1, 3))
    std = torch.reshape(std, (1, 1, 3))
    t = t * std + mean
    t = (t - t.min()) / (t.max() - t.min())
    image = t.numpy()*255
    
    return Image.fromarray(image.astype(np.uint8))

def load_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.Lambda(lambda t: t.unsqueeze(0))
    ])
    return preprocess(image)