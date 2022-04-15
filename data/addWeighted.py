import torch
import cv2
from PIL import Image
from torchvision import transforms
from autoaugment import ImageNetPolicy

# scr_path = "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900001162.jpg"
# scr_path = "/raid/zhangji/CD-dataset/ChestX/images_010/images/00024714_000.png"
# scr_path = "/raid/zhangji/CD-dataset/EuroSAT/2750/Forest/Forest_1000.jpg"
# scr_path = "/raid/zhangji/CD-dataset/EuroSAT/2750/PermanentCrop/PermanentCrop_1103.jpg"
scr_path = "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900000410.jpg"
# scr_path = "/raid/zhangji/CD-dataset/ISIC/ISIC2018_Task3_Training_Input/ISIC_0024311.jpg"
# scr_path = "/raid/zhangji/CD-dataset/CropDiseases/images/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab_3335.JPG"
# scr_path = "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900001114.jpg"
scr1_path = ["/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900001142.jpg",
             "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900001144.jpg",
             "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900000410.jpg",
             "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900000412.jpg",
             "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900000663.jpg",
             "/raid/zhangji/CD-dataset/miniImagenet/source/mini_imagenet_full_size/train/n02120079/n0212007900000664.jpg"]

def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std

def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    # return (normalized_feat *  style_std.expand(size) + style_mean.expand(size)).clamp(0.0, 1.0)
    return normalized_feat *  style_std.expand(size) + style_mean.expand(size)

def instance_norm_mix_random(content_feat):
    """replace content statistics with style statistics"""
    size = content_feat.size()
    # style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    # style_mean =  torch.Tensor(
    #     [torch.normal(mean=0.8, std=torch.rand(1)*0.4),
    #      torch.normal(mean=0.6, std=torch.rand(1)*0.4),
    #      torch.normal(mean=0.4, std=torch.rand(1)*0.4)]).reshape(1,3,1,1)
    style_mean =  torch.Tensor(
        [torch.normal(mean=0.6, std=torch.rand(1)*0.4),
         torch.normal(mean=0.6, std=torch.rand(1)*0.4),
         torch.normal(mean=0.6, std=torch.rand(1)*0.4)]).reshape(1,3,1,1)
    style_std =  torch.Tensor(
        [torch.normal(mean=0.2, std=torch.rand(1)*0.1),
         torch.normal(mean=0.2, std=torch.rand(1)*0.1),
         torch.normal(mean=0.2, std=torch.rand(1)*0.1)]).reshape(1,3,1,1)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return (normalized_feat *  style_std.expand(size) + style_mean.expand(size)).clamp(0.0, 1.0)
    # return normalized_feat *  style_std.expand(size) + style_mean.expand(size)

image_size = 224
transforms_list = [
    # transforms.RandomResizedCrop(image_size),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]
transforms_list2 = [
    transforms.RandomResizedCrop(image_size),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # transforms.RandomHorizontalFlip(),
    # ImageNetPolicy(),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
]
my_transform = transforms.Compose(transforms_list)
my_transform2 = transforms.Compose(transforms_list2)
img = my_transform2(Image.open(scr_path).convert('RGB')).unsqueeze(0)


for i, k in enumerate(scr1_path):
    img_k = my_transform(Image.open(k).convert('RGB')).unsqueeze(0)
    x_aug1 = instance_norm_mix_random(content_feat=img)
    x_aug2 = transforms.Compose([transforms.RandomRotation(30)])(x_aug1)
    # x_aug2 = instance_norm_mix(content_feat=img2, style_feat=img1)
    to_image = transforms.ToPILImage()
    img1_r = to_image(x_aug1.squeeze(0))
    img2_r = to_image(x_aug2.squeeze(0))
    save_path = "/home/zhangji/Project-CD/CDFSL-Style-master/data/image_{}.jpg".format(i+1)
    img1_r.save(save_path)
    save_path2 = "/home/zhangji/Project-CD/CDFSL-Style-master/data/image__{}.jpg".format(i+1)
    img2_r.save(save_path2)


















