from dataset import *
from torch.utils.data import DataLoader
import os
from predict import *
import matplotlib.pyplot as plt
from resnet_yolo import resnet50 as resnet

os.environ["CUDA_DEIVCES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__ == '__main__':
    test_dataset = PGDataset(root='./dataset/', list_data=['jogging'], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = resnet()
    model.load_state_dict(torch.load('yolo.pth'))
    model.eval().cuda()
    dpi = 80.0
    figsize = (300 / dpi, 300 / dpi)
    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi, clear=True)

    for i, (images, _, img_ori, gt) in enumerate(test_loader):
        images = images.cuda()
        pred = model(images)

        preds = decoder(pred,gt, use_nms=False)

        '''dpi = 80.0
        image_place = img_ori.numpy()
        images=image_place[0]
        images = np.transpose(images,(1,2,0))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(images)

        rect = plt.Rectangle((gt[0, 0], gt[0, 1]), gt[0, 2], gt[0, 3], linewidth=1, edgecolor="g", zorder=1,
                             fill=False)
        ax.add_patch(rect)

        color = ["#ff0000", "#FF8000", "#FFA500"]
        for j in range(len(preds)):
            bbox_pred = preds[j]
            rect = plt.Rectangle((bbox_pred[0], bbox_pred[1]), bbox_pred[2], bbox_pred[3],
                                 linewidth=1, edgecolor='r', zorder=1, fill=False)
            ax.add_patch(rect)
        plt.pause(.01)

    print('---start evaluate---')'''
