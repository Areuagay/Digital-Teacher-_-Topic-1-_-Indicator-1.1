'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn, ours_module, mosei_mosi_module, transmodal


def generate_model(opt):
    assert opt.model in ['multimodalcnn', 'oursmodule', 'mosei_mosi_module','transmodal']

    if opt.model == 'multimodalcnn':
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                            pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    elif opt.model == 'oursmodule':
        model = ours_module.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                          pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)
    elif opt.model == 'mosei_mosi_module':
        model = mosei_mosi_module.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                          pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)
    elif opt.model == 'transmodal':
        model = transmodal.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                                pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)
        # return model, {
        #                 # 'tokenizer': model.tokenizer.parameters(),
        #                 'bert': model.text.parameters(),
        #                 'visual': model.visual_model.parameters(),
        #                 'audio' : model.audio_model.parameters(),
        #                 'av1': model.av1.parameters(),
        #                 'va1': model.va1.parameters(),
        #                 'classifier': model.classifier_1.parameters()
        #                }

    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                   p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

    return model, model.parameters()
