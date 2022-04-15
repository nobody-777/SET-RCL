# Free-Lunch for Cross-Domain Few-Shot Learning: Style-Aware Episodic Training with Robust Contrastive Learning
PyTorch implementation of:
<br>
[**Free-Lunch for Cross-Domain Few-Shot Learning: Style-Aware Episodic Training with Robust Contrastive Learning**]
<br>

## Abstract

Cross-Domain Few-Shot Learning (CDFSL) aims for training an adaptable model that can learn out-of-domain classes with a handful of samples. Compared to the well-tudied few-shot learning problem, the difficulty for CDFSL lies in that the available training data from test tasks is not only extremely limited but also presents severe class differences from training tasks. To tackle this challenge, we propose Style-aware Episodic Training with Robust Contrastive Learning (SET-RCL), which is motivated by the key observation that a remarkable style-shift between tasks from source and target domains plays a negative role in cross-domain generalization. SET-RCL addresses the style-shift from two perspectives: 1) simulating the style distributions of unknown target domains (data perspective); and 2) learning a style-invariant representation (model perspective). Specifically, Style-aware Episodic Training (SET) focuses on manipulating the style distribution of training tasks in the source domain, such that the learned model can achieve better adaption on test tasks with domain-specific styles. To further improve cross-domain generalization under style-shift, we develop Robust Contrastive Learning (RCL) to capture style-invariant and discriminative representations from the manipulated tasks. Notably, our SET-RCL is orthogonal to existing FSL approaches, thus can be adopted as a “free-lunch” for boosting their CDFSL performance. Extensive experiments on nine benchmark datasets and six baseline methods demonstrate the effectiveness of our method. 

## Dependencies
* Python >= 3.5
* Pytorch >= 1.2.0 and torchvision (https://pytorch.org/)

## Datasets
We use miniImageNet as the single source domain, and use CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC and ChestX as the target domains.

For miniImageNet, CUB, Cars, Places and Plantae, download and process them seperately with the following commands.
- Set `DATASET_NAME` to: `miniImagenet`, `cub`, `cars`, `places` or `plantae`.
```
cd filelists
python process.py DATASET_NAME
cd ..
```

For CropDiseases, EuroSAT, ISIC and ChestX, download them from

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **CropDiseases**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data
    
    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

and put them under their respective paths, e.g., 'filelists/CropDiseases', 'filelists/EuroSAT', 'filelists/ISIC', 'filelists/chestX', then process them with following commands.
- Set `DATASET_NAME` to: `CropDiseases`, `EuroSAT`, `ISIC` or `chestX`.
```
cd filelists/DATASET_NAME
python write_DATASET_NAME_filelist.py
cd ..
```

## Pre-training
We adopt `baseline` pre-training from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) for all models.
- Download the pre-trained feature encoders from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- Or train your own pre-trained feature encoder.
```
python pretrain.py --dataset miniImagenet --name Pretrain --train_aug
```

## Training

1.Network training
```
python train.py --model ResNet10 --method GNN --n_shot 5 --name GNN_5s --train_aug --p 0.5 --w_s 0.05 --w_m 3.0
python train.py --model ResNet10 --method GNN --n_shot 1 --name GNN_1s --train_aug --p 0.5 --w_s 0.05 --w_m 3.0
```

## Inference

1.Test the trained model on the unseen domains.

- Specify the target dataset with `--dataset`: `cub`, `cars`, `places`, `plantae`, `CropDiseases`, `EuroSAT`, `ISIC` or `chestX`.
- Specify the saved model you want to evaluate with `--name`.
```
python test.py --dataset cub --n_shot 5 --model ResNet10 --method GNN --name GNN_5s
python test.py --dataset cub --n_shot 1 --model ResNet10 --method GNN --name GNN_1s
```

## Note
- This code is built upon the implementation from [Cross-Domain Few-Shot Classification via Adversarial Task Augmentation](https://github.com/Haoqing-Wang/CDFSL-ATA), [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://github.com/amazon-research/crossnorm-selfnorm).
- The dataset, model, and code are for non-commercial research purposes only.
