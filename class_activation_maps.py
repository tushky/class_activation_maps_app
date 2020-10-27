"""
Created by : Tushar Gadhiya
"""

import os
import io
import torch
import torch.nn as nn
from torch.nn.functional import relu
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

from utils import postprocess, read_image


class SubNet(nn.Module):
    '''
        Extract part of the model that requires grad
    '''
    def __init__(self, model):

        super(SubNet, self).__init__()
        self.classifier = model.classifier

    def forward(self, tensor):
        '''
            pass output of features to classifier
        '''
        print(f'input tensor shape is {tensor.shape}')
        tensor = tensor.mean(3).mean(2)
        tensor = torch.flatten(tensor, 1)
        return self.classifier(tensor)


class ClassActivationMaps:

    """
        Generate Class Activation Maps for the input image using one of the following methods.
            - Class Activation Maps (CAM)
            - Gradient weighted Class Activation Maps (Grad-CAM)
            - Gradient weighted Class Activation Maps++ (Grad-CAM++)
            - Score weighted Class Activation Maps (Score-CAM)

    Args:
        model (nn.Module): any pretrained convolutional neural network.
        layer_name (str or None): name of the conv layer you want to visualize.
            If None, the last occuring conv layer in the model will be used.

    Attributes:
        model (nn.Module): the pretrained network
        layer_name(str or none): name of the conv layer
        hooks (list): contains handles for forward and backward hooks
        interractive (bool): determines wether to remove the hooks after obtaining cam.
        methods (list): list of acceptable methods

    Example:
        model = torchvision.models.resnet34(pretrained=True)
        image = read_image('test.img')
        cam = ClassActivationMaps(model)
        cam.show_cam(image, method='gradcam++')
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.methods = ['cam', 'gradcam', 'gradcam++', 'scorecam']


    def cam(self, tensor, classifier, class_index):

        """
            Implimentation of vanilla Class Activation Map. Works on specific type
            of CNN network only. Last two layer of the network must have to be
            globalavgpool layer followed by single fully connected layer. For example
            it will not work on VGG16 or AlexNet. but it will work on ResNet or GoogleNet.

        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """
        # obtain prediction for the given image
        with torch.no_grad():
            class_activation_map = self.model.features(tensor)
            pred = classifier(class_activation_map)
        print(f'predicted class : {pred.argmax().item()}')

        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()

        class_activation_map = class_activation_map.permute(0, 2, 3, 1)

        weight = list(classifier.classifier.named_children())[-1][1].weight.data.t()

        class_activation_map = class_activation_map @ weight[:, [class_index]]
        class_activation_map = class_activation_map.permute(0, 3, 1, 2)

        return class_activation_map, class_index


    def gradcam(self, tensor, classifier, class_index):

        """
            Implimentation of gradient weighted class activation maps (Grad-CAM).
            It generalizes vanilla class activation maps and removes the limitation
            on the structure of the network.
        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """

        # obtain prediction for the given image
        with torch.no_grad():
            features = self.model.features(tensor)
        features.requires_grad = True
        pred = classifier(features)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()
        target[0, class_index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "index"
        loss = torch.sum(pred * target)
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = features.grad.detach()
        # obtain weights for each feature map of last conv layer using gradients of that layer
        grad = torch.mean(grad, (0, 2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        features = features.detach()
        # obtain weighed feature maps, keep possitive influence only
        class_activation_map = relu((features * grad).sum(1, keepdim=True))

        return class_activation_map, class_index


    def gradcamplus(self, tensor, classifier, class_index):

        """
            Implimentation of gradient weighted class activation maps++ (Grad-CAM++).
            It generalizes Grad-CAM and by extention CAM. It produces better visualization
            by considering pixels.
        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """
        a = torch.Tensor([1.1])
        with torch.no_grad():
            features = self.model.features(tensor)
        features.requires_grad = True
        pred = classifier(features)
        #print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()
        target[0, class_index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "index"
        loss = a ** torch.sum(pred * target)
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = features.grad.detach()
        print(grad.shape, 'grad shape')
        # Second order derivative of score of class 'c' with respect to output of last conv layer.
        # Since secod order derivative of relu layer is zero,
        # the formula is simplified to just square of the first order derivative.
        grad_2 = (grad ** 2)
        # Third order derivative of score of class 'c' with respect to output of last conv layer.
        # Since secod and third order derivative of relu layer is zero,
        # the formula is simplified to just cube of the first order derivative.
        grad_3 = (grad ** 3) * torch.log(a)
        #grad *= loss.item()
        # get global average of gradients of each feature map
        grad_3_sum = torch.mean(grad, (2, 3), keepdim=True)
        # prepare for alpha denominator
        grad_3 = grad_3 * grad_3_sum
        # get alpha
        alpha_d = 2 * grad_2 + grad_3
        alpha_d = torch.where(alpha_d != 0.0, alpha_d, torch.Tensor([1.0]))
        alpha = torch.div(grad_2, alpha_d+1e-06)
        alpha_t = torch.where(relu(grad)>0, alpha, torch.Tensor([0.0]))
        alpha_c = torch.sum(alpha_t, dim=(2, 3), keepdim=True)
        alpha_cn = torch.where(alpha_c !=0, alpha_c, torch.Tensor([1.0]))
        alpha /= alpha_cn
        # get final weights of each feature map
        weight = (alpha * relu(grad)).sum((2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        features = features.detach()
        # obtain weighed feature maps, keep possitive influence only
        class_activation_map = relu(features * weight).sum(1, keepdim=True)
        print(class_activation_map.shape)
        return class_activation_map, class_index


    def get_cam(self, tensor, classifier, method, class_index=None):

        """
            return class_activation_map generated by specified method
        """        

        if method == 'cam' :
            cam_map, index = self.cam(tensor, classifier, class_index)
        elif method == 'gradcam' :
            cam_map, index = self.gradcam(tensor, classifier, class_index)
        elif method == 'gradcam++':
            cam_map, index = self.gradcamplus(tensor, classifier, class_index)
        else:
            raise ValueError(f'Invalid method name {method}')

        # resize it to the shape of input image
        cam_map = nn.functional.interpolate(cam_map,
                                            size=(tensor.shape[2], tensor.shape[3]),
                                            mode='bilinear',
                                            align_corners=False
                                            )
        # remove batch and channel dimention
        cam_map = cam_map.squeeze(0).squeeze(0)

        # Normalize
        cam_map -= cam_map.min()
        cam_map /= (cam_map.max() + 1e-05)
        print(cam_map.shape)
        return cam_map, index


    def show_cam(self, tensor, classifier, method, class_index=None):

        """
            display class_activation_map generated by specified method
        """
        # specify size of the image
        plt.gcf().set_size_inches(5, 5)
        plt.axis('off')
        # plot CAM
        class_activation_map, index = self.get_cam(tensor, classifier, method, class_index)
        img = postprocess(tensor.mean(dim=0, keepdim=True))
        # plot input image
        plt.imshow(class_activation_map, cmap='jet')
        plt.imshow(img, alpha=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        return Image.open(buf), index

if __name__ == '__main__':
    cnn = torchvision.models.mobilenet_v2(pretrained=True)
    classifier = SubNet(cnn)
    cam = ClassActivationMaps(cnn)
    FILE = 'spider.png'
    image = read_image(os.getcwd() +'/images/'+ FILE)

    out, index = cam.show_cam(image, classifier, method='gradcam')
    plt.imshow(out)
    plt.show()
