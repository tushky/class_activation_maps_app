import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def read_image(path):
    #image = Image.open(path).convert('RGB')
    return preprocess(path)

def preprocess(image):

    '''
    Accepts PIL image and convert it into [B, C, H, W] shaped tensor normalized with imagenet stats
    '''
    transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    # Reshape to size 224 X 224
                    transforms.CenterCrop(224),
                    # Convert to torch tensor
                    transforms.ToTensor(),
                    # Normalize with imagenet stats
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    transforms.Lambda(lambda x: x.unsqueeze(0))
                ]
            )
    return transform(image)

def postprocess(t):

    '''
    accepts tensor of shape [B, C, H, W] processed with imagenet stats and convert it to PIL image
    '''

    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            # Remove normalization using imagenet stats
            transforms.Lambda(lambda t : t*std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)),
            # Remove batch dimention
            #transforms.Lambda(lambda  x: (x - x.min())/(x.max() - x.min() + 1e-06)),
            transforms.Lambda(lambda t: t.squeeze(0)),
            # Convert to PIL image
            transforms.ToPILImage()
        ]
    )
    return transform(t)