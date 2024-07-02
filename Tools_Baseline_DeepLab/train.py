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
    root_dir = '/media/razvan/segcol/'
    train_img_file = '/media/razvan/segcol/train/train_list.csv'
    train_segm_file = '/media/razvan/segcol/train/train_segmentation_maps.csv'
    valid_img_file = '/media/razvan/segcol/valid/valid_list.csv'
    valid_segm_file = '/media/razvan/segcol/valid/valid_segmentation_maps.csv'
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
    train_dataset = SegColDataset(root_dir, 
                                  train_img_file, train_segm_file, 
                                  simple_transform)
                                #   get_transform('train', (640, 480), (640, 480)))
    valid_dataset = SegColDataset(root_dir, 
                                  valid_img_file, valid_segm_file, 
                                  simple_transform)
                                #   get_transform('val', (640, 480)))
    model = DeepLab(args.backbone, args.out_stride, train_dataset.class_count, args.sync_bn)

    saver = Saver(args, timestamp=get_curtime())
    trainer = Trainer(args, model, train_dataset, valid_dataset, saver)

    # train/valid
    for epoch in range(args.epochs):
        trainer.training(epoch, evaluation=False)
        if is_interval(epoch):
            trainer.validation(epoch)
    print('Valid: best Dice:', trainer.best_dice, 'AP:', trainer.best_ap)

    # # test
    # epoch = trainer.load_best_checkpoint()
    # test_mIoU, test_Acc = trainer.validation(epoch, test=True)
    # print('Test: best mIoU:', test_mIoU, 'pixelAcc:', test_Acc)


if __name__ == '__main__':
    main()
