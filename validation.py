'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy,calculate_accuracy_perclass,calculate_f1_perclass,calculate_f1_prec_allclass, non_zero_mean
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def text_loader():
    text_label = []
    for n in os.listdir('./label_data'):
        p = os.path.join('./label_data', n)
        with open(p) as file:
            text_label.append([i[:-2] for i in file.readlines()])
    text = [text_label[i][np.random.randint(0, 20)] for i in range(8)]

    return text


def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger,modality='both',dist=None ):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()

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

    f1score = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    precise = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

    end_time = time.time()
    for i, (_, inputs_visual, vlens, inputs_audio, alens, label) in enumerate(data_loader):
        data_time.update(time.time() - end_time)


        # targets = targets.to(opt.device)
        targets = label
        # for index, value in enumerate(targets):
        #     if value[-3:] == 'neu':
        #         targets[index] = 0
        #     elif value[-3:] == 'fru':
        #         targets[index] = 1
        #     elif value[-3:] == 'sad':
        #         targets[index] = 2
        #     elif value[-3:] == 'hap':
        #         targets[index] = 3
        #     elif value[-3:] == 'ang':
        #         targets[index] = 4
        #     elif value[-3:] == 'exc':
        #         targets[index] = 5
        #     elif value[-3:] == 'sur':
        #         targets[index] = 6
        #     elif value[-3:] == 'fea':
        #         targets[index] = 7
        #     else:
        #         targets[index] = 8
        # targets = torch.tensor(targets, dtype=torch.long).cuda()
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




        
        targets = targets.to(opt.device)
        # print(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)

        # outputs = model(inputs_audio.to(opt.device), inputs_visual.to(opt.device), text)
        # loss = criterion(outputs, targets)

        feature,text_feature,result, ap1, vp1 = model(inputs_audio.to(opt.device), inputs_visual.to(opt.device),alens,vlens, text_loader())

        # 将模型输出转成txt文件
        # 将张量转换为 DataFrame
        result_norm = torch.nn.functional.normalize(result,p=2,dim=1)
        text_df = pd.DataFrame(text_feature.cpu().detach().numpy())
        feature_df = pd.DataFrame(feature.cpu().detach().numpy())
        result_df = pd.DataFrame(result_norm.cpu().detach().numpy())
        targets_df = pd.DataFrame(targets.cpu().detach().numpy())

        audio_df = pd.DataFrame(ap1.cpu().detach().numpy())
        video_df = pd.DataFrame(vp1.cpu().detach().numpy())

        # 保存为 CSV 文件
        text_df.to_csv(os.path.join('./csv/',f'text_tensor{i}.csv'), index=False)
        feature_df.to_csv(os.path.join('./csv/',f'feature_tensor{i}.csv'), index=False)
        result_df.to_csv(os.path.join('./csv/',f'result_tensor{i}.csv'), index=False)
        targets_df.to_csv(os.path.join('./csv/',f'targets_tensor{i}.csv'), index=False)

        audio_df.to_csv(os.path.join('./csv/',f'audio_feature_tensor{i}.csv'), index=False)
        video_df.to_csv(os.path.join('./csv/', f'video_feature_tensor{i}.csv'), index=False)





        # targets = torch.cat([targets, targets], dim=0)
        ce = criterion(result, targets)
        loss = ce

        # feature, text_feature, cls_result, audio_feature,  video_feature = model(inputs_audio.to(opt.device), inputs_visual.to(opt.device), text)
        # feature, text_feature, audio_feature,  video_feature = model(inputs_audio.to(opt.device), inputs_visual.to(opt.device), text)
        # feature, text_feature = model(inputs_audio.to(opt.device), inputs_visual.to(opt.device), text)
        # ce = criterion(feature, targets)
        # loss = ce

        prec1, prec5 = calculate_accuracy(result.data, targets.data, topk=(1,2))
        # prec1, prec5 = calculate_accuracy(cls_result.data, targets.data, topk=(1,5))
        f1score1, precise1 = calculate_f1_prec_allclass(result, targets)
        f1score = np.row_stack((f1score, f1score1))
        precise = np.row_stack((precise, precise1))

        # neu1 = calculate_f1_perclass(result, targets, 0)
        # fru1 = calculate_f1_perclass(result, targets, 1)
        # sad1 = calculate_f1_perclass(result, targets, 2)
        # hap1 = calculate_f1_perclass(result, targets, 3)
        # ang1 = calculate_f1_perclass(result, targets, 4)
        # exc1 = calculate_f1_perclass(result, targets, 5)
        # sur1 = calculate_f1_perclass(result, targets, 6)
        # fea1 = calculate_f1_perclass(result, targets, 7)

        neu1 = calculate_accuracy_perclass(result, targets, 0)
        fru1 = calculate_accuracy_perclass(result, targets, 1)
        sad1 = calculate_accuracy_perclass(result, targets, 2)
        hap1 = calculate_accuracy_perclass(result, targets, 3)
        ang1 = calculate_accuracy_perclass(result, targets, 4)
        exc1 = calculate_accuracy_perclass(result, targets, 5)
        sur1 = calculate_accuracy_perclass(result, targets, 6)
        fea1 = calculate_accuracy_perclass(result, targets, 7)


        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))

        neu.update(neu1, inputs_audio.size(0))
        fru.update(fru1, inputs_audio.size(0))
        sad.update(sad1, inputs_audio.size(0))
        hap.update(hap1, inputs_audio.size(0))
        ang.update(ang1, inputs_audio.size(0))
        exc.update(exc1, inputs_audio.size(0))
        sur.update(sur1, inputs_audio.size(0))
        fea.update(fea1, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

    # 计算总f1 和 acc
    f1mean = non_zero_mean(f1score)
    precmean = non_zero_mean(precise)
    print("f1:", np.around(f1mean, 4))
    print("prec:", np.around(precmean, 4))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'neu': neu.avg.item(),
                'fru': fru.avg.item(),
                'sad': sad.avg.item(),
                'hap': hap.avg.item(),
                'ang': ang.avg.item(),
                'exc': exc.avg.item(),
                'sur': sur.avg.item(),
                'fea': fea.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    return losses.avg.item(), top1.avg.item()

def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
    elif opt.model == 'oursmodule':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
    elif opt.model == 'mosei_mosi_module':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
    elif opt.model == 'transmodal':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)