import numpy as np
import torch
import torch.nn as nn
from modules.swintrans import SwinTransformer as STBackbone
from modules.base_model import Basemodel
from modules.visual_extractor import VisualExtractor


class BaseModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.swintrans = STBackbone(
            img_size=384,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
        # 得要预训练权重
        print('load pretrained weights!')
        # self.swintrans.load_weights(
        #     './swin_tiny_patch4_window7_224.pth'
        # )
        # Freeze parameters 这里要冻结参数，我们这里是迁移学习
        for _name, _weight in self.swintrans.named_parameters():
            _weight.requires_grad = False
        # 这里定义编码解码结构
        self.encoder_decoder = Basemodel(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
#         att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
#         att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
#         fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
# #         print(att_feats_0.shape)
#         att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
#         print(att_feats_0.shape)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
#         print(att_feats.shape)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats_0, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
