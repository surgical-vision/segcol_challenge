import os
import torch
import argument_parser

from utils.trainer import Trainer
from utils.saver import Saver
from model.deeplab import DeepLab
from datasets.dataset import ActiveSegCol
from active_selection import get_active_selector
from torchvision import transforms
from utils.misc import write_list_to_txt

args = argument_parser.parse_args()

simple_transform = transforms.Compose([
                            transforms.Resize((480, 640)),
                            transforms.ToTensor()])

unlabelled_file = '*____*/unlabelled_list.csv'

# Load the training set or part of  the labelled pool of data for active selection:
with open(args.train_img_file, 'r') as file:
    label_image_files = file.readlines()
with open(args.train_segm_file, 'r') as file:
    label_anno_files = file.readlines()

label_anno_files = [line.strip() for line in label_anno_files]
label_image_files = [os.path.join(args.root_dir,line.strip()) for line in label_image_files]

active_trainset = ActiveSegCol(args.root_dir, 
                               unlabelled_file, 
                               [], # no available annotations for the unlabelled
                               label_image_files,
                               label_anno_files,
                               simple_transform, split=0) # use all of the unlabelled data

checkpoint = torch.load('*____*/checkpoint.pth.tar')

model = DeepLab('drn', 16, 4, False)
model.load_state_dict(checkpoint["state_dict"])
saver = saver = Saver(args)
trainer = Trainer(args, model, active_trainset, active_trainset, saver)

# Replace with your active learning method
args.active_selection_mode = 'entropy'
active_selector = get_active_selector(args)

# Select best 400 unlabelled samples:
select_num = 400
select_img_paths = active_selector.select_next_batch(trainer.model, active_trainset, select_num)
# Write the txt file with the selected unlabelled images
write_list_to_txt(select_img_paths, txt_path='candidate_unlabelled_imgs.txt')
print("Unlabelled files selected!")