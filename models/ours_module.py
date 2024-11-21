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
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='reflect'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True))


class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
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
        self.conv1d_0 = conv1d_block(output_channels, 64)
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
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
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
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

        input_channels = 33
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
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)

        init_feature_extractor(self.visual_model, pretr_ef)

        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion = fusion

        # input_dim_video = input_dim_video // 2
        #
        # self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
        #                      num_heads=num_heads)
        # self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
        #                      num_heads=num_heads)
        input_dim_video = input_dim_video // 2
        self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                                  num_heads=num_heads)
        self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                                  num_heads=num_heads)

        self.cross_attention = CrossAttention(hidden_size=128, all_head_size=128,
                             head_num=8)



        self.text = BertModule()
        self.dropout = nn.Dropout(p=0.2)


        # self.classifier_1 = nn.Sequential(
        #     nn.Linear(e_dim * 2, self.text.bert.config.hidden_size),
        # )
        self.classifier_1 = nn.Sequential(
            nn.Linear(e_dim * 2, self.text.bert.config.hidden_size),
        )
        # self.softmax = nn.Softmax(1)
        self.single_cls = nn.Sequential(
            nn.Linear(e_dim, self.text.bert.config.hidden_size),
        )

    def forward(self, x_audio, x_visual, inputs_text):


        x_audio = self.audio_model.forward_feature(x_audio)
        x_audio = self.audio_model.forward_stage1(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av = self.av1(proj_x_v, proj_x_a)
        h_va = self.va1(proj_x_a, proj_x_v)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio = h_av + x_audio
        x_visual = h_va + x_visual
#------------------------------------------------------------------------------ +1
        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])


        # x = torch.cat((audio_pooled, video_pooled), dim=-1)
        # mask = torch.rand(8,8,1,1)
        audio_pooled1 = self.cross_attention(audio_pooled,video_pooled).squeeze(1)
        video_pooled1= self.cross_attention(video_pooled,audio_pooled).squeeze(1)
        x = torch.cat((audio_pooled1, video_pooled1), dim=-1)
        x1 = self.classifier_1(x)


        text = []
        # text = torch.tensor([])
        
        for sentence in inputs_text:
            token = self.tokenizer(sentence[0], return_tensors='pt', padding=True, truncation=True).to(torch.device('cuda'))
            token = self.text(token['input_ids'], token['attention_mask'])
            text.append(token)


        text = torch.cat(text,dim=0).cuda()

        x1 = x1/torch.sum(x1**2,dim=0)**0.5
        text = text/torch.sum(text**2,dim=0)**0.5
        x2 = self.dropout(x1 @ text.T)
        return x1, text, x2 #(bs,dim) * (csn, dim) -> (bs, csn)
        # return x1, text


