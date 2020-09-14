import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import os
import io

class Hook:
    
    def __init__(self, name, layer, backward=False):
        
        self.name = name
        
        if backward:
            #print('backward pass')
            self.handle = layer.register_backward_hook(self.hook_fn)
        else:
            self.handle = layer.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.module = module
        print(f'{self.name} output shape : {output.shape}')
    
    def remove(self):
        self.handle.remove()
        print(f'hook on layer {self.name} removed')


class CAM(nn.Module):

    def __init__(self, cnn, name='inception5b'):
        
        super().__init__()
        self.cnn = cnn
        self.cnn.eval()
        self.name = name

        self.recursive_hook(self.cnn, self.name)

    
    def get_cam(self, image):

        pred = self.cnn(image)

        cam = self.hook.output.data
        self.hook.remove()
        #cam = F.interpolate(cam, scale_factor=32, mode='bicubic', align_corners=False)
        #cam = cam.squeeze(0)
        cam = cam.permute(0, 2, 3, 1)

        weight = list(self.cnn.named_children())[-1][1].weight.data
        weight = weight.t()

        out = torch.matmul(cam, weight)
        out = out.permute(0, 3, 1, 2)
        out = out.max(dim=1, keepdim=True).values
        out = F.interpolate(out, scale_factor=self.img.shape[2]//out.shape[2], mode='bicubic', align_corners=False)

        mean_out = out.view(out.shape[2:]).numpy()

        plt.imshow(mean_out, cmap='jet')
        test_img = self.img.squeeze(0).permute(1, 2, 0).numpy()
        plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)), alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf), pred.argmax().item()

    def recursive_hook(self, module, target_name):
        for name, layer in module.named_children():
            if name == target_name:
                print(f'{name} layer hooked')
                self.hook = Hook(name, layer)
            self.recursive_hook(layer, target_name)

if __name__ == '__main__':

    cnn = torchvision.models.googlenet(pretrained=True)

    cam = CAM(cnn, 'inception5b')
    print(os.getcwd())
    image = Image.open(os.getcwd()+'/test/catdog.jpeg').convert('RGB')

    cam.get_cam(image)
