import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)# 24 x 64 x 112 x 112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 24 x 64 x 56 x 56
        self.layer1 = self._make_layer(block, 64, layers[0]) # 24 x 64 x 56 x 56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 24 x 128 x 28 x 28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 24 x 256 x 14 x 14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
       # x3 = self.layer4(x3)
        return x3

    def load_my_state_dict(self, parameter_url):
        param = model_zoo.load_url(parameter_url)
        model_dict = self.state_dict()
        for name, param in param.items():
            if name not in model_dict:
                continue
            else:
                model_dict[name].copy_(param)


class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.inplanes = 512
        # self.conv1 = self.conv_bn(2, 64, stride=4)  # 64 * 112 * 112
        # self.conv2 = self.conv_bn(64, 128, stride=2)  # 128 * 56 * 56
        # self.conv3 = self.conv_bn(128, 256, stride=2)  # 256 * 14 * 14
        self.conv1 = self._make_layer(BasicBlock, 512, 2, stride=2)
        # self.conv2 = self.conv_bn(256, 256, stride=1)
        # self.conv3 = self.conv_bn(256, 256, stride=1)

    def forward(self, x3, m3):

        # f1= torch.cat((x1, m1), dim=1)
        # f1 = self.conv1(f1)

        # f2 = torch.cat((x2, m2), dim=1)
        # f2 = self.conv2(f2)

        # f3 = torch.cat((x3, m3), dim=1)
        # f3 = self.conv3(f3)
        x = torch.cat((x3, m3),dim=1)
        # x=torch.cat((x3,m3),dim=1)
        x = self.conv1(x)

        return x

    def conv_bn(self, in_planes, out_planes, kernel_size=3, stride=1):
        layers = list()
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class PENet(nn.Module):
    def __init__(self):
        super(PENet, self).__init__()
        self.mask_net = MaskNet()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv4 = self.conv_bn(512, 512)
        self.conv5 = self.conv_bn(512, 256)  # 64 * 112 * 112

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self,x3, m3):
        #x = self.res_net(x)
        m = self.mask_net(x3, m3)
        # y = torch.cat((x, m), dim=1)
        y = self.conv4(m)
        y = self.conv5(y)

        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.dropout(y)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        return y

    def conv_bn(self, in_planes, out_planes, kernel_size=1, stride=1):
        layers = list()
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def resnet():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def PE():
    model = PENet()
    return model

class Binary_Accuracy(nn.Module):
    def __init__(self):
        super(Binary_Accuracy, self).__init__()
    def forward(self, pred,label):
        _,pos_out = torch.max(F.sigmoid(pred),1)
        _,label=torch.max(label,1)
        total_acc = torch.div(100*(pos_out == label).sum(),pos_out.shape[0])
        if((label==1).shape[0]==0):
            pos_acc = None
        else:
            pos_acc = torch.div(100*((label==1)&(pos_out == label)).sum(),(label==1).sum())
        if ((label != 1).shape[0] == 0):
            neg_acc = None
        else:
            neg_acc = torch.div(100 * ((label != 1) & (pos_out == label)).sum(), (label!=1).sum())
        '''
        #_,neg_out = torch.max(F.sigmoid(neg),1)
        total_acc = torch.div(pos_out.shape[0])



        total_acc=torch.div(100*(pos_out.sum() + (1-neg_out).sum()), pos_out.shape[0]+neg_out.shape[0])
        pos_acc = torch.div(100*pos_out.sum(),pos_out.shape[0])
        neg_acc = torch.div(100*(1-neg_out).sum(),neg_out.shape[0])'''
        return total_acc,pos_acc,neg_acc

class Custom_loss(nn.Module):
    def __init__(self):
        super(Custom_loss, self).__init__()
    def forward(self, pred_pos,label):
        '''
        loss_bce = nn.BCELoss()
        pos = F.sigmoid(pred_pos)
        neg = F.sigmoid(pred_neg)
        target_pos = torch.zeros_like(pos, dtype=torch.float)
        target_pos[:, 1] = 1
        target_neg = torch.zeros_like(neg, dtype=torch.float)
        target_neg[:, 0] = 1
        '''
        loss_bce = nn.BCELoss()
        pos = F.sigmoid(pred_pos)
        #cat1 = torch.cat((pos, neg))
        #cat2 = torch.cat((target_pos, target_neg))
        loss=loss_bce(pos, label)
        return loss

class Loglike_loss(nn.Module):
    def __init__(self):
        super(Loglike_loss, self).__init__()
    def forward(self, pred_pos,pred_neg):
        pos_loss = -F.log_softmax(pred_pos)[:, 1]
        neg_loss = -F.log_softmax(pred_neg)[:, 0]
        loss = pos_loss.mean() + neg_loss.mean()
        return loss

class NLL_loss(nn.Module):
    def __init__(self):
        super(NLL_loss, self).__init__()
    def forward(self, pred_pos,pred_neg):
        '''
        pos = F.log_softmax(pred_pos)
        neg = F.log_softmax(pred_neg)
        target_pos = torch.zeros_like(pos,dtype=torch.long)
        #target_pos[:,0] = 0.
        target_pos[:,1] = 1
        target_neg = torch.zeros_like(neg,dtype=torch.long)
        target_neg[:,0] = 1
        #target_neg[:,1] = 0.
        #nllloss = nn.NLLLoss()
        #nllloss = F.nll_loss()
        #F.nll_loss(pos,target_pos)
        pos_loss = F.nll_loss(input=pos,target=target_neg)
        neg_loss = F.nll_loss(neg,target_neg)
        #pos_loss = -pos[:,]
        loss = pos_loss+neg_loss
        return loss
        '''

