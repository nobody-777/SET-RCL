import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self):
        super().__init__()
        from methods.resnet12 import ResNet
        self.encoder = ResNet()

        # if args.backbone_class == 'ConvNet':
        #     from model.networks.convnet import ConvNet
        #     self.encoder = ConvNet()
        # elif args.backbone_class == 'Res12':
        #     hdim = 640
        #     from model.networks.res12 import ResNet
        #     self.encoder = ResNet()
        # elif args.backbone_class == 'Res18':
        #     hdim = 512
        #     from model.networks.res18 import ResNet
        #     self.encoder = ResNet()
        # elif args.backbone_class == 'WRN':
        #     hdim = 640
        #     from model.networks.WRN28 import Wide_ResNet
        #     self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        # else:
        #     raise ValueError('')

    def split_instances(self, data):
        # args = self.args
        shot = eval_shot = 1
        query = eval_query = 15
        way = eval_way =5
        if self.training:
            return  (torch.Tensor(np.arange(way*shot)).long().view(1, shot, way),
                     torch.Tensor(np.arange(way*shot, way * (shot + query))).long().view(1, query, way))
        else:
            return  (torch.Tensor(np.arange(eval_way * eval_shot)).long().view(1, eval_shot,  eval_way),
                     torch.Tensor(np.arange( eval_way* eval_shot,  eval_way * ( eval_shot +  eval_query))).long().view(1,  eval_query,  eval_way))

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')