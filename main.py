# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import random

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
from dataloader.data_loader import MSADataset
from dataloader.config import get_args, get_config

from torch.nn import CosineEmbeddingLoss
from dataloader.data_loader import get_single_modal_loader
from transformers import T5Tokenizer
from dataloader.tools import contain_nonum, is_number
torch.backends.cudnn.enabled = False

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

model_path = 't5base.model'
tokenizer = T5Tokenizer.from_pretrained(model_path, mirror="tuna")
dataset_name = get_args().dataset
prompt_dict = ['classification']

# def collate_fn(batch):
#     '''
#     Collate functions assume batch = [Dataset[i] for i in index_set]
#     '''
#     # for later use we sort the batch in descending order of length
#     labels = []
#     ids = []
#     print(len(batch))
#
#     for sample in batch:
#
#         ids.append(sample[2].strip())
#         label = sample[1][0]
#         labels.append(label)
#
#
#     inputs_seq = []
#     outputs_seq = []
#     for sample in batch:
#         text = ""
#         # for i in sample[0][3]:
#         #     text =sample[0][3][i]
#         #     print(text)
#         #     text = text.strip()
#         text =  sample[0][3]
#         print(text)
#         score = str(sample[1])
#         inputs_seq.append(text),
#         outputs_seq.append(score)
#
#     t = []
#     for sequence in inputs_seq:
#         seq = "sst2 sentence:"
#         for word in sequence:
#             seq += " " + word
#         t.append(seq)
#
#     encoding = tokenizer(
#         t,
#         return_tensors="pt", padding=True
#     )
#     # T5 model things are batch_first
#     t5_input_id = encoding.input_ids
#     t5_att_mask = encoding.attention_mask
#     target_encoding = tokenizer(
#         outputs_seq, padding="longest"
#     )
#     t5_labels = target_encoding.input_ids
#     t5_labels = torch.tensor(t5_labels)
#     t5_labels[t5_labels == tokenizer.pad_token_id] = -100
#     # lengths are useful later in using RNNs
#
#     return None, None, None, None, None, labels, t5_input_id, t5_att_mask, t5_labels, ids
def collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    if dataset_name != 'meld1':
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            # if sample[0][1].shape[1] != 35:
            #     print(sample[0][1].shape)
            # if sample[0][2].shape[1] != 74:
            #     print(sample[0][2].shape)
            if len(sample[0]) > 4:  # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:  # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            # labels.append(torch.from_numpy(sample[1]))
            labels.append(sample[1])
            ids.append(sample[2])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)

        # labels = torch.cat(labels, dim=0)

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        # if labels.size(1) == 7:
        #     labels = labels[:,0][:,None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=alens.max().item())

        inputs_seq = []
        outputs_seq = []
        prompt_emb = []
        prompt_id = []
        # A = -0.0001
        # B = 0.0001  # 小数的范围A ~ B
        # C = 6
        for sample in batch:
            # text = " ".join(sample[0][3])
            text = sample[0][3]
            # print('ori sample:{}'.format(text))
            # score = str(sample[1][0][0])
            score = str(sample[1])
            # source = sample[2]
            # print('source:{}'.format(source))


            inputs_seq.append(str(text))
            outputs_seq.append(score)


        # print(type(task_prefix),type(inputs_seq))

        # lengths are useful later in using RNNs
        # lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])

        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return [], visual.to(opt.device), vlens, acoustic.to(opt.device), alens, labels
    # else:
    #     ### 没有视频模态
    #     inputs_seq = []
    #     outputs_seq = []
    #     # A = -0.0001
    #     # B = 0.0001  # 小数的范围A ~ B
    #     # C = 6
    #     for sample in batch:
    #         text = " ".join(sample[0][3])
    #         label = str(sample[1])
    #
    #         inputs_seq.append(text)
    #         outputs_seq.append(label)
    #
    #     encoding = tokenizer(
    #         [task_prefix + sequence for sequence in inputs_seq],
    #         return_tensors="pt", padding=True
    #     )
    #     # T5 model things are batch_first
    #     t5_input_id = encoding.input_ids
    #     t5_att_mask = encoding.attention_mask
    #     target_encoding = tokenizer(
    #         outputs_seq, padding="longest"
    #     )
    #     t5_labels = target_encoding.input_ids
    #     t5_labels = torch.tensor(t5_labels)
    #     t5_labels[t5_labels == tokenizer.pad_token_id] = -100
    #
    #     acoustic = torch.FloatTensor([sample[0][2] for sample in batch])
    #     labels = [sample[1] for sample in batch]
    #     return None, None, None, acoustic.to(opt.device), None, labels


if __name__ == '__main__':
    opt = parse_opts()
    args = get_args()
    train_config = get_config(opt.multi_dataset, mode='train', batch_size=opt.batch_size)
    valid_config = get_config(opt.multi_dataset, mode='valid', batch_size=opt.batch_size)
    test_config = get_config(opt.multi_dataset, mode='test', batch_size=opt.batch_size)



    n_folds = 1
    test_accuracies = []

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained = opt.pretrain_path != 'None'

    # opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    for fold in range(n_folds):
        # if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

        print(opt)
        with open(os.path.join(opt.result_path, 'opts' + str(time.time()) + str(fold) + '.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)

        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        cosine = CosineEmbeddingLoss()
        cosine = cosine.to(opt.device)
        mse = nn.MSELoss()
        mse = mse.to(opt.device)

        if not opt.no_train:
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])

            # training_data = torch.utils.data.DataLoader(
            #     dataset=MSADataset(get_config()),
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     collate_fn=collate_fn)
            train_loader = torch.utils.data.DataLoader(
                dataset=MSADataset(train_config),
                batch_size=opt.batch_size,
                shuffle=True,
                collate_fn=collate_fn)
            # training_data = get_training_set(opt, spatial_transform=video_transform)
#-----------------------------------------------改config--------
            # train_loader = torch.utils.data.DataLoader(
            #     training_data,
            #     batch_size=opt.batch_size,
            #     shuffle=True,
            #     num_workers=opt.n_threads,
            #     pin_memory=True)

            train_logger = Logger(
                os.path.join(opt.result_path, 'train' + str(fold) + '.log'),
                ['epoch', 'loss', 'ang','exc','fea','fru','hap','neu','sad','sur', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch' + str(fold) + '.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

            # optimizer = optim.Adam(
            #     parameters,
            #     lr=0.001,
            #     betas=(0.9, 0.999), 
            #     eps=1e-08, 
            #     weight_decay=0)
            optimizer = optim.SGD(
            #     [{'params': parameters['bert'],'lr':opt.learning_rate * 0.2},
            #      # {'params': parameters['tokenizer'], 'lr': opt.learning_rate * 0.2},
            #      {'params': parameters['visual'], 'lr': opt.learning_rate},
            #      {'params': parameters['audio'], 'lr': opt.learning_rate},
            #      {'params': parameters['va1'], 'lr': opt.learning_rate},
            #      {'params': parameters['av1'], 'lr': opt.learning_rate},
            #      {'params': parameters['classifier'], 'lr': opt.learning_rate}
            #      ],
            #     momentum=opt.momentum,
            #     dampening=opt.dampening,
            #     weight_decay=opt.weight_decay,
            #     nesterov=False
            # ) if opt.model == 'oursmodule' else \
            #     optim.SGD(
            #     model.parameters(),
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)

            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=opt.lr_factor, patience=opt.lr_patience)

        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            validation_data = get_single_modal_loader(args,train_config,shuffle=True)
            # validation_data = get_validation_set(opt, spatial_transform=video_transform)

            val_loader = torch.utils.data.DataLoader(
                MSADataset(valid_config),
                batch_size=opt.batch_size,
                shuffle=False,
                collate_fn=collate_fn)

            val_logger = Logger(
                os.path.join(opt.result_path, 'val' + str(fold) + '.log'), ['epoch', 'loss','ang','exc','fea','fru','hap','neu','sad','sur', 'prec1', 'prec5'])
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss','ang','exc','fea','fru','hap','neu','sad','sur', 'prec1', 'prec5'])

        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, mse, cosine, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
                save_checkpoint(state, False, opt, fold)

            if not opt.no_val:
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                                   val_logger)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }

                save_checkpoint(state, is_best, opt, fold)

        if opt.test:
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss','neu','fru','sad','hap','ang','exc','sur','fea', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            test_data = get_single_modal_loader(args, test_config, shuffle=True)
            # test_data = get_test_set(opt, spatial_transform=video_transform)

            # load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
            model.load_state_dict(best_state['state_dict'])

            test_loader = torch.utils.data.DataLoader(
                MSADataset(valid_config),
                batch_size=opt.batch_size,
                shuffle=False,
                collate_fn=collate_fn)

            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                              test_logger)

            with open(os.path.join(opt.result_path, 'test_set_bestval' + str(fold) + '.txt'), 'a') as f:
                f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1)

    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write(
            'Prec1: ' + str(np.mean(np.array(test_accuracies))) + '+' + str(np.std(np.array(test_accuracies))) + '\n')
