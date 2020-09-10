from torchvision import models
import torch
import torch.nn as nn
from utils import tensor_to_image
from PIL import Image
import numpy as np


class SaliencyMap(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()
        #for param in self.cnn.parameters() : param.requires_grad=False
    
    def forward(self, t):

        t.requires_grad = True
        out = self.cnn(t)
        target = torch.zeros_like(out)
        target[0, out.argmax(dim=1).item()] = 1
        loss = torch.sum(out * target)
        print(loss)
        loss.backward()
        t_grad = t.grad.detach()
        t_grad = t_grad.abs().max(0).values
        t_grad = t_grad[0,:,:].numpy()
        t_grad = (t_grad - t_grad.min()) / (t_grad.max() - t_grad.min())
        t_grad *= 255
        return Image.fromarray(t_grad.astype(np.uint8))

if __name__ == "__main__":
    img = torch.randn((1, 3, 224, 224))
    smap = SaliencyMap()
    out = smap(img)
    out.show()