import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torchvision import transforms
from datasets.dataset import SegColDataset
from model.deeplab import DeepLab
import torch
from torch.utils.data import DataLoader
from utils.image import save_image_batch_to_disk

model = DeepLab('drn', 16, 4, False)

root_dir = '/media/razvan/segcol/'
train_img_file = '/media/razvan/segcol/train/train_list.csv'
train_segm_file = '/media/razvan/segcol/train/train_segmentation_maps.csv'
valid_img_file = '/media/razvan/segcol/valid/valid_list.csv'
valid_segm_file = '/media/razvan/segcol/valid/valid_segmentation_maps.csv'
simple_transform = transforms.Compose([
                        transforms.Resize((480, 640)),
                        transforms.ToTensor()])
valid_dataset = SegColDataset(root_dir, 
                                valid_img_file, valid_segm_file, 
                                simple_transform)
checkpoint = torch.load('/home/razvan/seg_col_task2/runs/SegCol/None_Jun30_163142/checkpoint.pth.tar')
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to('cuda:0')

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)


with torch.no_grad():
    for samples in valid_loader:
        outputs = model(samples['img'].to('cuda:0'))
        save_image_batch_to_disk(outputs, 'results/', samples['filename'])