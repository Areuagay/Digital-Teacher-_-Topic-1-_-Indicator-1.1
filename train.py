'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import time

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_perclass, calculate_f1_perclass, cmd
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def text_loader():
    text_label = []
    for n in os.listdir('./label_data'):
        p = os.path.join('./label_data', n)
        with open(p) as file:
            text_label.append([i[:-2] for i in file.readlines()])
    text = [text_label[i][np.random.randint(0, 20)] for i in range(8)]

    return text



def train_epoch_multimodal(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                           epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    print("******************************************")

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    neu = AverageMeter()
    fru = AverageMeter()
    sad = AverageMeter()
    hap = AverageMeter()
    ang = AverageMeter()
    exc = AverageMeter()
    sur = AverageMeter()
    fea = AverageMeter()

    end_time = time.time()
    text = text_loader()
    labels = ['neu','fru','sad','hap','ang','exc','sur','fea']
    # for i, (audio_inputs, visual_inputs, targets, text) in enumerate(data_loader):
    for i,(_, visual_inputs, vlens, audio_inputs, alens, label) in enumerate(data_loader):
        if visual_inputs is None:
            continue
        data_time.update(time.time() - end_time)

        # targets = targets.to(opt.device)
        targets = label
        # for index, value in enumerate(targets):
        #     if value[-3:]=='neu':
        #         targets[index] = 0
        #     elif value[-3:]=='fru':
        #         targets[index] = 1
        #     elif value[-3:]=='sad':
        #         targets[index] = 2
        #     elif value[-3:]=='hap':
        #         targets[index] = 3
        #     elif value[-3:]=='ang':
        #         targets[index] = 4
        #     elif value[-3:]=='exc':
        #         targets[index] = 5
        #     elif value[-3:]=='sur':
        #         targets[index] = 6
        #     elif value[-3:]=='fea':
        #         targets[index] = 7
        #     else:
        #         targets[index] = 8
        for index, value in enumerate(targets):
            if value[-3:]=='ang':
                targets[index] = 0
            elif value[-3:]=='exc':
                targets[index] = 1
            elif value[-3:]=='fea':
                targets[index] = 2
            elif value[-3:]=='fru':
                targets[index] = 3
            elif value[-3:]=='hap':
                targets[index] = 4
            elif value[-3:]=='neu':
                targets[index] = 5
            elif value[-3:]=='sad':
                targets[index] = 6
            elif value[-3:]=='sur':
                targets[index] = 7
            else:
                targets[index] = 8
        targets = torch.tensor(targets,dtype=torch.long).cuda()



        # visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4) #bs, len, dim
        # visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
        #                                       visual_inputs.shape[3], visual_inputs.shape[4])

        audio_inputs = Variable(audio_inputs.cuda())
        visual_inputs = Variable(visual_inputs.cuda())

        targets = Variable(targets)
        # targets = torch.cat([targets,targets],dim=0)
        model = model.to(opt.device)
        feature, text_feature, result, ap1, vp1 = model(audio_inputs, visual_inputs,alens,vlens ,text_loader())
        # feature, text_feature = model(audio_inputs, visual_inputs, text)
        # feature, text_feature, cls_result, audio_feature, video_feature = model(audio_inputs, visual_inputs, text)
        # feature, text_feature, audio_feature, video_feature = model(audio_inputs, visual_inputs, text)
        text_feature = text_feature[:8]


        # ce+mse+cosine
        ce = criterion(result, targets)
        mse = mse_loss(feature, text_feature[targets])
        # cosine = 16*cosine_loss(feature, text_feature[targets],targets)
        cosine = 0
        for j in range(0,16):
            y = torch.full((8,), -1).to(opt.device)
            first_row = feature[j]
            repeated_first_row = first_row.repeat(8, 1).to(opt.device) # 将（16，768）中的第i行向量取出，复制8次
            y1=int(targets[j])
            y[y1] = 1
            cosine1 = cosine_loss(repeated_first_row, text_feature, y)
            cosine += cosine1
        cmd1 =  cmd(ap1, vp1, 2).to(opt.device)
        loss = ce + 16*mse +cosine*0.1+ 0.04*cmd1
        # loss = ce + 16*mse +cosine*0.1
        # loss = ce + 0.04*cmd1

        # for i in range(0,7):
        #     if i != targets[i]:
        #         cosine += torch.div(1,cosine_loss(feature,text_feature[i],targets))
        # loss = ce+mse+cosine




        # ce+mse
        # mse =  16*torch.mean((feature - text_feature[targets])**2)
        # loss = ce + mse
        # loss = ce

        # 余弦外置直接计算损失
        # ce = criterion(cls_result, targets)
        # mse = 16 * mse_loss(feature, text_feature[targets])
        # cosine = 16*cosine_loss(feature, text_feature[targets],targets)

        # ce = criterion(feature, targets)
        # mse = 16 * mse_loss(feature, text_feature[targets])
        # cosine = 16*cosine_loss(feature, text_feature[targets],targets)+ \
        #          cosine_loss(audio_feature, text_feature[targets], targets) + \
        #          cosine_loss(video_feature, text_feature[targets], targets)
        # loss = ce + mse + cosine

        losses.update(loss.data, audio_inputs.size(0))

        prec1, prec5 = calculate_accuracy(result, targets, topk=(1, 2))

        neu1 = calculate_f1_perclass(result,targets,0)
        fru1 = calculate_f1_perclass(result, targets, 1)
        sad1 = calculate_f1_perclass(result, targets, 2)
        hap1 = calculate_f1_perclass(result, targets, 3)
        ang1 = calculate_f1_perclass(result, targets, 4)
        exc1 = calculate_f1_perclass(result, targets, 5)
        sur1 = calculate_f1_perclass(result, targets, 6)
        fea1 = calculate_f1_perclass(result, targets, 7)


        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))

        neu.update(neu1, audio_inputs.size(0))
        fru.update(fru1, audio_inputs.size(0))
        sad.update(sad1, audio_inputs.size(0))
        hap.update(hap1, audio_inputs.size(0))
        ang.update(ang1, audio_inputs.size(0))
        exc.update(exc1, audio_inputs.size(0))
        sur.update(sur1, audio_inputs.size(0))
        fea.update(fea1, audio_inputs.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        #             'mse': mse.item(),
        #             'ce': ce.item(),
        #             'cosine': cosine.item(),
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        # 'mse {mse:.3f}\t'
        # mse = mse,                             'ce {ce:.3f}\t'
        #                   'mse {mse:.3f}\t'
        #                   'cosine {cosine:.3f}\t'     ce=ce,
        #                 mse = mse,
        #                 cosine=cosine,
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'ang': ang.avg.item(),
        'exc': exc.avg.item(),
        'fea': fea.avg.item(),
        'fru': fru.avg.item(),
        'hap': hap.avg.item(),
        'neu': neu.avg.item(),
        'sad': sad.avg.item(),
        'sur': sur.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })


def train_epoch(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    if opt.model == 'multimodalcnn':
        train_epoch_multimodal(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                               epoch_logger, batch_logger)
        return
    elif opt.model == 'oursmodule':
        train_epoch_multimodal(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                               epoch_logger, batch_logger)
        return
    elif opt.model == 'mosei_mosi_module':
        train_epoch_multimodal(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                               epoch_logger, batch_logger)
        return
    elif opt.model == 'transmodal':
        train_epoch_multimodal(epoch, data_loader, model, criterion, mse_loss, cosine_loss, optimizer, opt,
                               epoch_logger, batch_logger)
        return
