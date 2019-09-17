from ang_dataset import *
from dataset import *
from torch.utils.data import DataLoader
import os
from predict import *
import matplotlib.pyplot as plt
from resnet_yolo import resnet50 as resnet

os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
file_root = '/home/dlagkrtns/pub/db/COCO2017/'
data_dir1 = 'train2017'
data_dir2 = 'val2017'

learning_rate = 0.001
num_epochs = 200
batch_size = 1

if __name__ == '__main__':
    test_dataset = COCODataset(root=file_root, data_dir=data_dir2, list_file=['val_gt_all_more.txt'], train=False,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = resnet()
    model.load_state_dict(torch.load('COCO_best.pth'))
    model.eval().cuda()
    for i, (images,img_ori, newgt) in enumerate(test_loader):
        images = images.cuda()
        pred = model(images)

        preds = decoder(pred, newgt, thresh=False)

        dpi = 80.0
        image_place = img_ori.numpy()
        size = image_place.shape[2:]
        figsize = (size[1] / dpi, size[0] / dpi)
        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(np.moveaxis(image_place[0, [2, 1, 0]], 0, -1))

        rect = plt.Rectangle((newgt[0, 0], newgt[0, 1]), newgt[0, 2], newgt[0, 3], linewidth=1, edgecolor="g", zorder=1,
                             fill=False)
        ax.add_patch(rect)

        color = ["#ff0000", "#FF8000", "#FFA500"]
        for j in range(len(preds)):
            bbox_pred = preds[j]
            rect = plt.Rectangle((bbox_pred[0], bbox_pred[1]), bbox_pred[2], bbox_pred[3],
                                 linewidth=1, edgecolor='r', zorder=1, fill=False)
            ax.add_patch(rect)
        plt.show()
        plt.close()
    print('---start evaluate---')
