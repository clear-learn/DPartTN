import torch.nn as nn
import math
import torch


__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):
    def __init__(self, features, image_size=448):
        super(VGG, self).__init__()
        self.grid = 7
        self.features = features
        self.image_size = image_size
        self.classifier = nn.Sequential(
            nn.Linear(512 * self.grid * self.grid, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10*(self.grid**2)),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 10)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_ch = 4
    first_flag = True
    for out_ch in cfg:
        stride = 1
        if out_ch == 64 and first_flag:
            stride = 2
            first_flag = False
        if out_ch == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_ch = out_ch
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(batch_norm=False, use_gpu=False, **kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=batch_norm), **kwargs)
    if use_gpu:
        return model.cuda()
    return model


def vgg13(batch_norm=False, use_gpu=False, **kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=batch_norm), **kwargs)
    if use_gpu:
        return model.cuda()
    return model


def vgg16(batch_norm=False, use_gpu=False, **kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=batch_norm), **kwargs)
    if use_gpu:
        return model.cuda()
    return model


def vgg19(batch_norm=False, use_gpu=False, **kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=batch_norm), **kwargs)
    if use_gpu:
        return model.cuda()
    return model


def test():
    import torch
    from torch.autograd import Variable
    net = vgg11(batch_norm=True, use_gpu=True)
    img = torch.rand(30, 4, 448, 448)
    img = Variable(img).cuda()
    output = net(img).cpu()
    print(output.size())


if __name__ == '__main__':
    test()
