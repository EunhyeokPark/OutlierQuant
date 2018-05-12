import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ol_module import *

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(OL_Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.input_dim = (3,224,224)
        self.features = OL_Sequential(
            OL_Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  #0
            nn.ReLU(inplace=True),  #1
            OL_Active(True, True),
            nn.MaxPool2d(kernel_size=3, stride=2),            #2
            OL_Conv2d(64, 192, kernel_size=5, padding=2), #3
            nn.ReLU(inplace=True),  #4
            OL_Active(True, True),
            nn.MaxPool2d(kernel_size=3, stride=2),            #5
            OL_Conv2d(192, 384, kernel_size=3, padding=1), #6
            nn.ReLU(inplace=True), #7
            OL_Active(True, True),
            OL_Conv2d(384, 256, kernel_size=3, padding=1), #8
            nn.ReLU(inplace=True), #9
            OL_Active(True, True),
            OL_Conv2d(256, 256, kernel_size=3, padding=1), #10
            nn.ReLU(inplace=True), #11
            OL_Active(True, True),
            nn.MaxPool2d(kernel_size=3, stride=2),#12
        )
        self.classifier = OL_Sequential(
            nn.Dropout(), #0
            OL_Linear(256 * 6 * 6, 4096), #1
            nn.ReLU(inplace=True), #2
            OL_Active(True, True),
            nn.Dropout(), #3
            OL_Linear(4096, 4096), #4
            nn.ReLU(inplace=True), #5
            OL_Active(True, True),
            OL_Linear(4096, num_classes), #6
        )

        self.flatten = Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def ol_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model