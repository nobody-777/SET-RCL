import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='few-shot script')
  parser.add_argument('--dataset', default='miniImagenet', help='miniImagenet/cub/cars/places/plantae/CropDiseases/EuroSAT/ISIC/chestX')
  parser.add_argument('--testset', default='miniImagenet', help='miniImagenet/cub/cars/places/plantae/CropDiseases/EuroSAT/ISIC/chestX')
  parser.add_argument('--model', default='ResNet10', help='model: ResNet{10|18|34}')
  parser.add_argument('--method', default='GNN', help='GNN')
  parser.add_argument('--train_n_way', default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way', default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot', default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--train_aug', action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--name', default='Debug', type=str, help='GNN')
  parser.add_argument('--save_dir', default='output', type=str, help='')
  parser.add_argument('--data_dir', default='', type=str, help='')
  parser.add_argument('--save_freq', default=40, type=int, help='Save frequency')
  parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
  parser.add_argument('--stop_epoch', default=400, type=int, help='Stopping epoch')
  parser.add_argument('--resume_epoch', default=0, type=int, help='')
  # Pretrain
  parser.add_argument('--num_classes', default=200, type=int, help='total number of classes in softmax')
  # Train
  parser.add_argument('--resume_dir', default='Pretrain', type=str, help='continue from previous trained model with largest epoch')
  parser.add_argument('--pretrain_epoch', default=399, type=int, help='')
  parser.add_argument('--max_lr', default=80., type=float, help='max_lr')
  parser.add_argument('--T_max', default=5, type=int, help='')
  parser.add_argument('--lamb', default=1., type=float, help='alpha')
  parser.add_argument('--prob', default=0.5, type=float, help='probability of using original Images')
  # For ours
  parser.add_argument('--p', default=0.5, type=float, help='probability of opening style training')
  parser.add_argument('--w_s', default=0.05, type=float, help='balance weight for contrastive loss')
  parser.add_argument('--w_m', default=3.0, type=float, help='balance weight for mix loss')
  parser.add_argument('--temp', default=1.0, type=float, help='temp')
  # Fine-tune
  parser.add_argument('--finetune_epoch', default=50, type=int, help='')
  return parser.parse_args()









