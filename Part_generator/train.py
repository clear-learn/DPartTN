from torch.utils.data import DataLoader
from torch.autograd import Variable

from resnet_yolo import resnet50 as resnet
from yoloLoss_he import yoloLoss
from dataset import *
import numpy as np

os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
file_root = '/home/dlagkrtns/pub/db/COCO2017/'
data_dir1 = 'train2017'
data_dir2 = 'val2017'


learning_rate = 0.001
num_epochs = 200
batch_size = 6
use_resnet = True
if use_resnet:
    net = resnet(pretrained=True)
# print(net)
net.load_state_dict(torch.load('COCO_best.pth'))
# net.load_state_dict(torch.load('yolo.pth'))

criterion = yoloLoss(7, 2, 5, 0.5)

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate*1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]



optimizer = torch.optim.Adam(params, lr=learning_rate)

train_dataset = COCODataset(root=file_root,data_dir= data_dir1, list_file=['train_gt_all_more.txt'], train=True,transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


test_dataset = COCODataset(root=file_root,data_dir= data_dir2, list_file=['val_gt_all_more.txt'], train=True,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % batch_size)

#logfile = open('log.txt', 'w')

num_iter = 0
best_test_loss = np.inf

net = net.cuda()

for epoch in range(num_epochs):
    net.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' %
                  (epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1

    # validation
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    '''if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        torch.save(net.state_dict(), 'COCO_best.pth')
        print('get best test loss %.5f' % validation_loss)
    print('get test loss %.5f' % validation_loss)
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()
    torch.save(net.state_dict(), 'COCO_yolo.pth')'''
