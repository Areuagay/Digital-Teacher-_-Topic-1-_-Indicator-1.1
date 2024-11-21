'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import torch
import shutil
import numpy as np
import sklearn



class AverageMeter(object):
    """Computes and stores the average and current value"""

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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    #print('target', target, 'output', output)    
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        #print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        #print('F1: ', f1)
        return res, f1*100
    #print(res)
    return res


def calculate_accuracy_perclass(output, target, cls, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    output = torch.argmax(output, dim=1)
    output = (output == cls).to(torch.long)
    target = (target == cls).to(torch.long)
    # for i in target:
    #     if target[i] == 0
    acc = torch.sum((output==target).to(torch.long)) * 100 / output.size(0)
    return acc

def calculate_f1_perclass(output, target, cls):
    """Computes the precision@k for the specified values of k"""

    output = torch.argmax(output, dim=1)
    output = (output == cls).to(torch.long)
    target = (target == cls).to(torch.long)
    # for i in target:
    #     if target[i] == 0
    f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()), list(output.cpu().numpy()),zero_division=1, average='weighted')*100
    return f1


# 计算非零列平均值
import numpy as np


def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num / den

def calculate_f1_prec_allclass(output, target):
    output = torch.argmax(output, dim=1)
    y_str = sklearn.metrics.classification_report(list(target.cpu().numpy()), list(output.cpu().numpy()),
                                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # 提取F1分数
    f1_scores = []
    for line in y_str.split('\n')[2:-5]:
        f1_scores.append(float(line.split()[3]))  # 提取每行的F1分数
    # 转换为数组形式
    f1_scores_array = np.array(f1_scores)

    # 提取prec
    prec_scores = []
    for line in y_str.split('\n')[2:-5]:
        prec_scores.append(float(line.split()[1]))  # 提取每行的prec
    # 转换为数组形式
    prec_scores_array = np.array(prec_scores)

    return f1_scores_array, prec_scores_array


def save_checkpoint(state, is_best, opt, fold):
    torch.save(state, '%s/%s_checkpoint'% (opt.result_path, opt.store_name)+str(fold)+'.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name)+str(fold)+'.pth','%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.2 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


def cmd(x1, x2, n_moments):
    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1-mx1
    sx2 = x2-mx2
    dm = matchnorm(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        scms += scm(sx1, sx2, i + 2)
    return scms

# 经验期望向量
def matchnorm(x1, x2):
    power = torch.pow(x1-x2,2)
    summed = torch.sum(power)
    sqrt = summed**(0.5)
    return sqrt
    # return ((x1-x2)**2).sum().sqrt()
# 样本中心矩向量
def scm(sx1, sx2, k):
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return matchnorm(ss1, ss2)