import time
import os
import network_1by1 as network
from util_dataset_1by1 import *
from util_mask_1by1 import *
from util_train import *

from torch.utils.data import DataLoader
from torch.autograd import Variable

file_root = '/home/dlagkrtns/vkdlf/PE_jiwoo/JPEGImages/'
os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
start_epoch = 0
num_epochs = 400
lr=0.001
weight_decay=1e-4
batch_size=24

##################### Load Train Data #####################
train_dataset = PEDataset(root=file_root, list_file=['voc_20072012_1090_train.txt'], train=True,transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = PEDataset(root=file_root, list_file=['voc_20072012_1090_test.txt'], train=False,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

##################### Set device/Net/Criterion #####################
device = torch.device("cuda:0")
resnet = network.resnet()
penet = network.PE()
train_accuracy = network.Binary_Accuracy()
criterion = network.Custom_loss()
# to device
resnet.to(device)
penet.to(device)
train_accuracy.to(device)
criterion.to(device)

##################### Load Model #####################
load_state(resnet,torch.load('original.pth'))
for param in resnet.parameters():
    param.requires_grad = False
for param in penet.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, penet.parameters()), lr=lr)
print('Model Loaded, GPU : %d' %(device.index))
print('Parameter Fixed')

best_acc = 0

for epoch in range(num_epochs):
        positive = []
        negative = []
        total_loss = 0.
        total_acc=0.
        positive_acc=0.
        negative_acc=0.
        validation_loss = 0.
        vald_acc=0.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # switch to train mode


        penet.train()


        ##################### Train #####################
        for i, (img, cover, part,label) in enumerate(train_loader):
            images = Variable(img)
            part_img = Variable(part)
            cover_img = Variable(cover)
            image = img.cuda(device)
            K1 = part_img.cuda(device)
            K3 = cover_img.cuda(device)
            label = label.cuda(device)

            #K1, K2, K3 = mask_product(K1, K2, image, K3)

            feature1,feature2,feature3 = resnet(K3)
            feature_pos1,feature_pos2,feature_pos3 = resnet(K1)
            # f1 [64 112 112]
            # f2 [128 56 56]
            # f3 [256 14 14]
            output = penet(feature1,feature2,feature3, feature_pos1,feature_pos2,feature_pos3)

            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            t_acc,p_acc,n_acc = train_accuracy(output, label)
            total_acc +=t_acc.item()
            positive_acc = p_acc.item()
            negative_acc = n_acc.item()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Total_Loss {Total_loss:.4f}\t'
                  'Total_ACC {Total_ACC:.4f}\t'
                  'pos_ACC {pos_ACC:.4f}\t'
                  'neg_ACC {neg_ACC:.4f}'.format(
                epoch, i, len(train_loader), loss=loss,
                Total_loss=total_loss/(i+1),Total_ACC=total_acc/(i+1),
                pos_ACC=positive_acc,neg_ACC = negative_acc
            ))








        ##################### Test #####################
        penet.eval()

        for i, (img, cover, part,label) in enumerate(test_loader):
            images = Variable(img)
            part_img = Variable(part)
            cover_img = Variable(cover)
            image = img.cuda(device)
            K1 = part_img.cuda(device)
            K3 = cover_img.cuda(device)
            label = label.cuda(device)

            #K1, K2, K3 = mask_product(K1, K2, image, K3)

            feature1,feature2,feature3 = resnet(K3)
            feature_pos1,feature_pos2,feature_pos3 = resnet(K1)
            # f1 [64 112 112]
            # f2 [128 56 56]
            # f3 [256 14 14]
            output = penet(feature1,feature2,feature3, feature_pos1,feature_pos2,feature_pos3)

            loss = criterion(output, label)
            validation_loss += loss.item()
            t_acc_v,pos_s,neg_s =   train_accuracy(output, label)
            vald_acc+=t_acc_v.item()
            if(i%10==0):
                print('TEST: [{0}][{1}/{2}]\t'
                      'Loss {loss:.4f}\t'
                      'Total_Loss {Total_loss:.4f}\t'
                      'Total_ACC {Total_ACC:.4f}\t'
                      'pos_ACC {pos_ACC:.4f}\t'
                      'neg_ACC {neg_ACC:.4f}'.format(
                    epoch, i, len(test_loader), loss=loss,
                    Total_loss=validation_loss / (i + 1), Total_ACC=vald_acc / (i + 1),
                    pos_ACC=pos_s, neg_ACC=neg_s
                ))
        validation_loss /= len(test_loader)
        vald_acc /= len(test_loader)
        print('Test_Epoch: [{0}]\t'
                 'Total_Loss {Total_loss:.4f}\t'
              'Total_Acc {Total_Acc:.4f}'.format(
                 epoch, Total_loss=validation_loss, Total_Acc=vald_acc))

        file_name = '512_concat.pth'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'penet',
            'state_dict': penet.state_dict(),
            'best_acc1': vald_acc,
            'optimizer': optimizer.state_dict(),
        }, best_acc < vald_acc, filename=file_name)
        if (best_acc < vald_acc):
            best_acc = vald_acc


















