import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

if len(args.gpu_ids) > 1:
    args.sync_bn = True
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torchvision import transforms
from datasets.transforms import get_transform
# from datasets.build_datasets import build_datasets
from datasets.dataset import SegColDataset
from model.deeplab import DeepLab
from utils.saver import Saver
from utils.trainer import Trainer
from utils.misc import get_curtime


def is_interval(epoch):
    return epoch % args.eval_interval == (args.eval_interval - 1)


def main():
    random.seed(args.seed)
    train_transform = A.Compose([   #A.Resize((480, 640)),
                                    # A.HorizontalFlip(p=0.5),
                                    # A.VerticalFlip(p=0.5),
                                    # A.RandomRotate90(p=0.5),
                                    A.RandomBrightnessContrast(p=0.5),
                                    A.HueSaturationValue(p=0.5),
                                    A.GaussianBlur(p=0.5),
                                    A.GaussNoise(p=0.5),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()
                                    ], additional_targets={'mask': 'target'})
    valid_transform = A.Compose([
                                    A.Resize(height=480, width=640),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()
                                ], additional_targets={'mask': 'target'})
    
    simple_transform = transforms.Compose([
                            transforms.Resize((480, 640)),
                            transforms.ToTensor()])
    train_dataset = SegColDataset(args.root_dir, 
                                  args.train_img_file, args.train_segm_file, 
                                  simple_transform)
    valid_dataset = SegColDataset(args.root_dir, 
                                  args.valid_img_file, args.valid_segm_file, 
                                  simple_transform)
    model = DeepLab(args.backbone, args.out_stride, train_dataset.class_count, args.sync_bn)

    saver = Saver(args, timestamp=get_curtime())
    trainer = Trainer(args, model, train_dataset, valid_dataset, saver)

    # train/valid
    for epoch in range(args.epochs):
        trainer.training(epoch, evaluation=False)
        if is_interval(epoch):
            trainer.validation(epoch)
    print('Valid: best Dice:', trainer.best_dice, 'AP:', trainer.best_ap)


if __name__ == '__main__':
    main()
