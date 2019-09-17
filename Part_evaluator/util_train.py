import torch
import shutil
import scipy.io as sio

def load_state(net, param):
    model_dict = net.state_dict()
    for name, param in param.items():
        if name not in model_dict:
            continue
        #if name == 'conv1.weight':
        #    continue
        else:
            model_dict[name].copy_(param)

def save_checkpoint(state, is_best, filename='5_Channel.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_' + filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

### Just for chicking weight
def save_weight(model, num):
    k = model.state_dict()
    for m in k:
        data1 = k[m].data.cpu().numpy()
        # scipy.io.savemat('\\save_tmp\\test.mat', dict(data1))
        data2 = {}
        data2['data'] = data1
        sio.savemat('./Save_weight/' + str(num) + m + '.mat', data2)
        # np.save('./save_tmp/'+str(num)+m+'.npy',data1)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr2 = lr * (0.1 ** (epoch // 30))
    return lr2

def load_equal_print(net, param, param2):
    name_list = []
    model_dict = net.state_dict()
    # param2 = param2['state_dict']
    for name, param in param.items():
        if name not in model_dict:
            continue
        else:
            z = (param2[name].data.cpu().numpy())
            l = param.data.cpu().numpy()
            k = (z == l)
            if (~k.all()):
                name_list.append(name)
            # model_dict[name].copy_(param)
    return name_list