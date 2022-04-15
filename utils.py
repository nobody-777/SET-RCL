import torch
import numpy as np

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std

def instance_norm_mix_random(content_feat):
    """replace content statistics with style statistics"""
    size = content_feat.size()
    content_mean, content_std = calc_ins_mean_std(content_feat)
    m1, m2, m3 = torch.normal(mean=0.4, std=torch.rand(1) * 0.2), torch.normal(mean=0.4,std=torch.rand(1) * 0.2), torch.normal(mean=0.4, std=torch.rand(1) * 0.2)
    s1, s2, s3 = torch.normal(mean=0.2, std=torch.rand(1)*0.1), torch.normal(mean=0.2, std=torch.rand(1)*0.1), torch.normal(mean=0.2, std=torch.rand(1)*0.1)
    style_mean = torch.Tensor([m1, m2, m3]).reshape(1, 3, 1, 1)
    style_std = torch.Tensor([s1, s2, s3]).reshape(1, 3, 1, 1)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return (normalized_feat *  style_std.expand(size) + style_mean.expand(size)).clamp(0.0, 1.0)


def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(2*W)
        cy = np.random.randint(2*H)

        bbx1 = np.clip((cx - cut_w) // 2, 0, W)
        bby1 = np.clip((cy - cut_h) // 2, 0, H)
        bbx2 = np.clip((cx + cut_w) // 2, 0, W)
        bby2 = np.clip((cy + cut_h) // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2

def cn_op_2ins_space_chan_random(x, crop=None, beta=1, bbx_thres=0.1, lam=None):
    """2-instance crossnorm with cropping."""
    if crop in ['content']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix_random(content_feat=x[:, :, bbx1:bbx2, bby1:bby2])

        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix_random(content_feat=x)

    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug

    return x










