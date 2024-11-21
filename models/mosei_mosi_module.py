# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn
import numpy as np

from models.efficientface import InvertedResidual, LocalFeatureExtractor
from models.modulator import Modulator
from models.transformer_timm import Attention, AttentionBlock
from models.cross_attention.crossAttention import CrossAttention

from models.wav2vec2 import Wav2Vec2ForSpeechClassification
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import BertTokenizer, BertModel

import os
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='reflect'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True))


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths, use_seq=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        bs = x.size(0)
        # print('x_shape:{}'.format(x.shape))
        # print('lengths_shape:{}'.format(lengths.shape))

        # packed_sequence = pack_padded_sequence(x, lengths.cpu().to(torch.int64), enforce_sorted=False)
        # print('x shape:{}'.format(x.shape))
        # print('length shape:{}'.format(lengths.shape))
        out_pack, final_states = self.rnn(x)
        # print('out_pack_data_shape:{}'.format(out_pack.data.shape))

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        # print('h_shape:{}'.format(h.shape))

        if use_seq:
            x_sort_idx = torch.argsort(-lengths)
            x_unsort_idx = torch.argsort(x_sort_idx).long()
            # print('out_pack_shape:{}'.format(out_pack.shape))
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            return y_1, out
        else:
            return y_1, None


class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=1):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )
        self.conv1d_0 = conv1d_block(1, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self.im_per_sample = im_per_sample

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global average pooling
        return x

    def forward_stage1(self, x):
        # Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        # x = x.view(n_samples, self.im_per_sample, x.shape[1])
        # x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)


def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding_mode='same'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode='reflect'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()
        # audio_input=32
        # self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        # self.wav2vec = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

        input_channels = 1
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        # self.wav2vec2 = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        # x = self.wav2vec(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward_feature(self, audios):
        """
        audios:输入音频的numpy数组 (b, l)
        输出:33维向量
        """
        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        output = self.wav2vec(inputs.input_values[0].cuda()).logits
        output = torch.transpose(output, 1, 2)
        # print(output.shape)
        # return self.wav2vec(inputs.input_values[0].cuda(), attention_mask=inputs.attention_mask.cuda()).logits
        return output


# label = ["This is a happy guy.", "This is a sad guy.",
#          "This is a scared guy.", "This is an angry guy.",
#          "This is a calm guy.", "This is a surprised guy.",
#          "This is a neutral guy.", "This is a disgust guy."]

#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# inputs_text = [tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(torch.device('cuda')) for sentence in label]
# print(inputs_text)
# inputs_text = [tokenizer(sentence, return_tensors='pt', padding=True, truncation=True) for sentence in label]


class BertModule(nn.Module):
    def __init__(self):
        super(BertModule, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Add a classification layer on top of BERT
        # self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Run input through BERT model
        # print(self.bert.device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the [CLS] token
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        # Pass the [CLS] token through the classification layer
        # logits = self.fc(cls_hidden_state)
        return cls_hidden_state


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1):
        super(MultiModalCNN, self).__init__()
        assert fusion in ['ia', 'it', 'lt', 'iat', 'ita', 'itt'], print('Unsupported fusion method: {}'.format(fusion))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, 1)

        init_feature_extractor(self.visual_model, pretr_ef)

        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion = fusion

        # input_dim_video = input_dim_video // 2

        # self.video_residual_module = RNNEncoder(in_size=64,
        #                                         hidden_size=32,
        #                                         out_size=256,
        #                                         num_layers=1,
        #                                         dropout=0.3,
        #                                         bidirectional=False)
        # self.audio_residual_module = RNNEncoder(in_size=64,
        #                                         hidden_size=32,
        #                                         out_size=256,
        #                                         num_layers=1,
        #                                         dropout=0.3,
        #                                         bidirectional=False)
        # self.audio_feature1 = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        # )
        #
        # self.video_feature1 = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        # )
        #
        # self.audio_feature2 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        # )
        #
        # self.video_feature2 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        # )
        self.video_residual_module = RNNEncoder(in_size=64,
                                                hidden_size=32,
                                                out_size=256,
                                                num_layers=1,
                                                dropout=0.3,
                                                bidirectional=False)
        self.audio_residual_module = RNNEncoder(in_size=64,
                                                hidden_size=32,
                                                out_size=256,
                                                num_layers=1,
                                                dropout=0.3,
                                                bidirectional=False)
        self.audio_feature1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.video_feature1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.audio_feature2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.video_feature2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                             num_heads=num_heads)
        self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                             num_heads=num_heads)
        # self.cross_attention1 = CrossAttention(hidden_size=128, all_head_size=128,
        #                                       head_num=8)
        self.cross_attention1a = CrossAttention(hidden_size=256, all_head_size=256,
                                              head_num=16)
        self.cross_attention1v = CrossAttention(hidden_size=256, all_head_size=256,
                                              head_num=16)
        self.cross_attention2a = CrossAttention(hidden_size=256, all_head_size=256,
                                               head_num=16)
        self.cross_attention2v = CrossAttention(hidden_size=256, all_head_size=256,
                                               head_num=16)


        self.text = BertModule()
        self.dropout = nn.Dropout(p=0.2)

        # self.classifier_1 = nn.Sequential(
        #     nn.Linear(e_dim * 2, self.text.bert.config.hidden_size),
        # )
        self.classifier_1 = nn.Sequential(
            nn.Linear(self.text.bert.config.hidden_size, num_classes),
        )
        self.out_feature = nn.Sequential(
            nn.Linear(e_dim * 4, self.text.bert.config.hidden_size),
        )
        # self.softmax = nn.Softmax(1)
        # self.single_cls = nn.Sequential(
        #     nn.Linear(e_dim, self.text.bert.config.hidden_size),
        # )

    def forward(self, x_visual, x_audio, video_len, audio_len, inputs_text):


        x_audio, audio_seq = self.audio_residual_module(x_audio, audio_len, use_seq=False)
        x_visual, video_seq = self.video_residual_module(x_visual, video_len, use_seq=False)


        x_audio = x_audio + self.audio_feature1(x_audio)
        x_visual = x_visual + self.video_feature1(x_visual)

        x_audio = self.cross_attention1a(x_audio, x_visual).squeeze(1) * (x_audio+1)
        x_visual = self.cross_attention1v(x_visual, x_audio).squeeze(1) * (x_visual+1)


        x_audio = x_audio + self.audio_feature2(x_audio)
        x_visual = x_visual + self.video_feature2(x_visual)


        audio_pooled2 = self.cross_attention2a(x_audio, x_visual).squeeze(1) * (x_audio+1)
        video_pooled2 = self.cross_attention2v(x_visual, x_audio).squeeze(1) * (x_visual+1)
        x = torch.cat((audio_pooled2, video_pooled2), dim=-1)

        x2 = self.out_feature(x)
        x1 = self.classifier_1(x2)
        text = []
        # text = torch.tensor([])

        for sentence in inputs_text:
            token = self.tokenizer(sentence[0], return_tensors='pt', padding=True, truncation=True).to(
                torch.device('cuda'))
            token = self.text(token['input_ids'], token['attention_mask'])
            text.append(token)

        text = torch.cat(text, dim=0).cuda()




        # # 展平 text 张量
        # text_flattened = text.view(-1)
        #
        # # 将张量值约束在0-1之间
        # text_normalized = text_flattened
        #
        # # 创建特征图 - 文本特征
        # fig, ax = plt.subplots()
        # im = ax.imshow(text_normalized.detach().cpu().numpy().reshape(1, -1), cmap='viridis', aspect='auto', vmin=0, vmax=1)
        # cbar = plt.colorbar(im)
        # cbar.set_label('Feature Value')
        # plt.xticks(ticks=[0, 767], labels=['0', '767'])
        # plt.yticks(ticks=[0], labels=['8'])
        # plt.savefig('text_feature_map.jpg', format='jpg')
        # plt.show()
        #
        # x2_flattened = x2.view(-1)
        # # 创建特征图 - x2 特征
        # x2_normalized = x2_flattened
        #
        # fig, ax = plt.subplots()
        # im = ax.imshow(x2_normalized.detach().cpu().numpy().reshape(1, -1), cmap='viridis', aspect='auto', vmin=0, vmax=1)
        # cbar = plt.colorbar(im)
        # cbar.set_label('Feature Value')
        # plt.xticks(ticks=[0, 767], labels=['0', '767'])
        # plt.yticks(ticks=[0], labels=['16'])
        # plt.savefig('x2_feature_map.jpg', format='jpg')
        # plt.show()



        return x2,text,x1, audio_pooled2, video_pooled2


        # x1 = x1 / torch.sum(x1 ** 2, dim=0) ** 0.5
        # text = text / torch.sum(text ** 2, dim=0) ** 0.5
        # x2 = self.dropout(x1 @ text.T)
        # return x, text, x1  # (bs,dim) * (csn, dim) -> (bs, csn)
        # return x1, text, x2
        # return x1, text, self.single_cls(audio_pooled), self.single_cls(video_pooled)
