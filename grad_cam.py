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
from utils import load_image, tensor_to_image


class Hook:
    
    def __init__(self, name, layer, backward=False):
        
        self.name = name
        self.backward = backward
        if self.backward:
            print(f'backward hook set on layer {self.name}')
            self.handle = layer.register_backward_hook(self.hook_fn)
        else:
            print(f'forward hook set on layer {self.name}')
            self.handle = layer.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.module = module
        print(f'{"backward" if self.backward else "forward"} hook executed on layer {self.name}')

    def remove(self):
        self.handle.remove()
        print(f'{"backward" if self.backward else "forward"} hook on layer {self.name} removed')


class SubNet(nn.Module):

    def __init__(self, cnn):

        super().__init__()
        cnn.eval()
        self.avgpool = cnn.avgpool
        self.dropout = cnn.dropout
        self.fc = cnn.fc
        #self.classifier = cnn.classifier
    
    def forward(self, t):
        print(f'input tensor shape is {t.shape}')
        #t = t.mean(3).mean(2)
        #t = torch.flatten(t, 1)
        #return self.classifier(t)
        t = self.avgpool(t)
        t = torch.flatten(t, 1)
        t = self.dropout(t)
        t = self.fc(t)
        return t


class GradCAM():

    
    def __init__(self, cnn, name=None):
        
        self.cnn = cnn
        self.name = name
        self.relu = nn.ReLU()
        self.hook = None
        print(f'hook layer name : {self.name}')
        self.hook_last_conv(self)

    def hook_last_conv(self, name=None):

        model = self.cnn
        print(type(model))
        if isinstance(model, torchvision.models.GoogLeNet) or self.name:
            print(f'searching for layer {self.name} recursivly')
            self._named_hook(model, self.name if self.name else 'inception5b', '', 0)
        else:
            conv = [None, None]
            conv = self._recursive_hook(model, conv, '', 0)
            self.hook = Hook(conv[0], conv[1])

    def _named_hook(self, module, target_name, parent_name , depth):

        '''Recursivly search for "target_name" layer in the model and add hook '''
        for name, layer in module.named_children():
            name = parent_name + '_' + name if parent_name else name
            #print('\t'*depth, name)
            if name == target_name:
                self.hook = Hook(name, layer)
                print(f'{name} layer hooked')
            self._named_hook(layer, target_name, name, depth+1)

    def _recursive_hook(self, module, conv, parent_name, depth):
        
        '''Recursively search for last occuring conv layer in the model and return its name and layer'''
        for name, layer in module.named_children():
            name = parent_name + '_' + name if parent_name else name
            #print('\t'*depth, name)
            if isinstance(layer, nn.Conv2d):
                conv[0], conv[1] = name, layer
            self._recursive_hook(layer, conv, name, depth+1)
        return conv

    def get_gradcam(self, image, model, index=None):

        self.cnn.eval()
        with torch.no_grad():
            original = self.cnn(image)
            print(f'Predicted class index : {original.argmax().item()}')
        
        cam = self.hook.output.data
        cam.requires_grad = True

        #self.hook.remove()

        pred = model(cam)
        index = index if index else pred.argmax().item()

        target = torch.zeros_like(pred)
        target[0, index] = 1
        target.require_grad=True

        loss = torch.sum(pred * target)

        model.zero_grad()

        loss.backward()

        grad = cam.grad
        grad = torch.mean(grad, (2,3), keepdim=True)

        out = self.relu((cam.detach() * grad).sum(1, keepdim=True))
        out = nn.functional.interpolate(out, scale_factor=image.shape[2] // out.shape[2], mode='bilinear', align_corners=False)
        out = out.squeeze(0).squeeze(0)
        out -= out.min()
        out /= out.max()
        plt.axis('off')
        plt.imshow(out, cmap='jet')
        test_img = image.squeeze(0).permute(1, 2, 0).numpy()
        plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)), alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        return Image.open(buf), index

    def get_cam(self, image):

        self.cnn.eval()
        with torch.no_grad():
            pred = self.cnn(image)
            print(f'Predicted class index : {pred.argmax().item()}')
        
        cam = self.hook.output.data
        
        
        cam = cam.permute(0, 2, 3, 1)

        weight = list(self.cnn.named_children())[-1][1].weight.data
        weight = weight.t()

        out = torch.matmul(cam, weight)
        out = out.permute(0, 3, 1, 2)
        out = out.max(dim=1, keepdim=True).values
        out = F.interpolate(out, scale_factor=image.shape[2]//out.shape[2], mode='bicubic', align_corners=False)

        mean_out = out.view(out.shape[2:]).numpy()
        
        plt.axis('off')
        plt.imshow(mean_out, cmap='jet')
        test_img = image.squeeze(0).permute(1, 2, 0).numpy()
        plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)), alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf), pred.argmax().item()

    

if __name__ == '__main__':

    cnn = torchvision.models.resnet18(pretrained=True)
    cam = GradCAM(cnn)
    image = read_image(os.getcwd()+'/test/spider.png')
    cam.show_cam(image)