import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
import LRPtools.utils as LRPutil
import torch.nn.functional as F
from methods import backbone


class Normalize(nn.Module):
  def __init__(self, power=2):
    super(Normalize, self).__init__()
    self.power = power

  def forward(self, x):
    norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
    out = x.div(norm)
    return out


class GnnNet(MetaTemplate):
  FWT=False
  def __init__(self, model_func,  n_way, n_support):
    super(GnnNet, self).__init__(model_func, n_way, n_support)
    # loss function
    self.loss_fn_CE = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.FWT else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'
    # for low-dim transformation
    # self.fc_lowD = nn.Linear(self.feat_dim, 50)
    self.fc_lowD = nn.Sequential(
        nn.Linear(self.feat_dim, self.feat_dim),
        nn.ReLU(),
        nn.Linear(self.feat_dim, 128),
    )
    self.l2norm = Normalize(2)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.fc_lowD.cuda()
    self.l2norm.cuda()
    return self


  def forward_gnn(self, zs):
    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, self.n_way)], dim=1)
    support_label = support_label.view(1, -1, self.n_way).cuda()
    nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def scores(self, x):
    feat_highD = self.feature(x)

    z = self.fc(feat_highD)  # 105 * 128
    z = z.view(self.n_way, -1, z.size(1))  # 5 * 21 * 128
    z_stack = [
      torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2))
      for i in range(self.n_query)]
    assert (z_stack[0].size(1) == self.n_way * (self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores, feat_highD

  def set_forward(self, x, x_aug):
    loss_con = None
    loss_mix = None
    # train with contrastive loss + mix loss, on the original task and the augment task.
    if (x is not None) and (x_aug is not None):
      x = x.cuda()
      x_aug = x_aug.cuda()
      x= x.view(-1, *x.size()[2:])
      x_aug= x_aug.view(-1, *x_aug.size()[2:])

      scores, aug_feat_highD = self.scores(x_aug)
      batch_size = aug_feat_highD.size(0)
      labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support + self.n_query)).unsqueeze(1)
      labels = torch.zeros(self.n_way * (self.n_support + self.n_query), self.n_way).scatter(1, labels, 1).view(self.n_way, self.n_support + self.n_query, self.n_way)
      labels = labels.view(-1, self.n_way).cuda()

      # compute contrastive loss
      x_feat_lowD = self.l2norm(self.fc_lowD(self.feature(x)))  # 105 * 50
      aug_feat_lowD = self.l2norm(self.fc_lowD(aug_feat_highD))  # 105 * 50
      # mean_feat_lowD = (x_feat_lowD + aug_feat_lowD) / 2
      # prototypes = mean_feat_lowD.view(self.n_way, self.n_support + self.n_query, -1).mean(1)

      # compute SS loss
      prototypes_x = x_feat_lowD.view(self.n_way, self.n_support + self.n_query, -1).mean(1)
      prototypes_aug = aug_feat_lowD.view(self.n_way, self.n_support + self.n_query, -1).mean(1)

      # temp_s = self.temp
      temp_s = self.temp
      x_logits_proto = torch.mm(x_feat_lowD, prototypes_aug.t()) / temp_s
      aug_logits_proto = torch.mm(aug_feat_lowD, prototypes_x.t()) / temp_s
      loss_x = -torch.mean(torch.sum(F.log_softmax(x_logits_proto, dim=1) * labels, dim=1))
      loss_aug = -torch.mean(torch.sum(F.log_softmax(aug_logits_proto, dim=1) * labels, dim=1))
      loss_ss = (loss_x + loss_aug) / 2

      # compute contrastive loss
      shuffle_idx = torch.randperm(batch_size)
      mapping = {k: v for (v, k) in enumerate(shuffle_idx)}
      reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])
      aug_feat_lowD = aug_feat_lowD[reverse_idx]

      sim_clean = torch.mm(x_feat_lowD, x_feat_lowD.t())
      mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
      sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)
      sim_aug = torch.mm(x_feat_lowD, aug_feat_lowD.t())
      sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)
      logits_pos = torch.bmm(x_feat_lowD.view(batch_size, 1, -1), aug_feat_lowD.view(batch_size, -1, 1)).squeeze(-1)
      logits_neg = torch.cat([sim_clean, sim_aug], dim=1)
      logits = torch.cat([logits_pos, logits_neg], dim=1)
      instance_labels = torch.zeros(batch_size).long().cuda()
      loss_con = self.loss_fn_CE(logits / self.temp, instance_labels)

      # compute mix loss
      L = np.random.beta(8, 8)
      idx = torch.randperm(batch_size)
      x_mix = L * x + (1 - L) * x[idx]
      aug_mix = L * x_aug + (1 - L) * x_aug[idx]
      labels_mix = L * labels + (1 - L) * labels[idx]
      x_mix_feat = self.l2norm(self.fc_lowD((self.feature(x_mix))))  # 210 * 50
      aug_mix_feat = self.l2norm(self.fc_lowD((self.feature(aug_mix))))  # 210 * 50
      x_logits_proto = torch.mm(x_mix_feat, prototypes_aug.t()) / self.temp
      aug_logits_proto = torch.mm(aug_mix_feat, prototypes_x.t()) / self.temp
      loss_x = -torch.mean(torch.sum(F.log_softmax(x_logits_proto, dim=1) * labels_mix, dim=1))
      loss_aug = -torch.mean(torch.sum(F.log_softmax(aug_logits_proto, dim=1) * labels_mix, dim=1))
      loss_m = (loss_x + loss_aug) / 2

      loss_mix = [loss_ss, loss_m]

    else:
      if x_aug:
        x = x_aug
      x = x.cuda()
      x= x.view(-1, *x.size()[2:])
      scores, _, = self.scores(x)

    return scores, loss_con, loss_mix


  def set_forward_loss(self, x, x_aug):
    y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
    scores, loss_con, loss_mix = self.set_forward(x, x_aug)
    loss = self.loss_fn_CE(scores, y_query)
    return scores, loss, loss_con, loss_mix

