import argument_parser
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torchvision import transforms
from datasets.dataset import SegColDataset
from model.deeplab import DeepLab
import torch
from torch.utils.data import DataLoader
from utils.image import save_image_batch_to_disk
from utils.trainer import Trainer
from utils.saver import Saver
from utils.misc import get_curtime

from tqdm import tqdm

SAVE_PREDICTED_RESULTS = False

model = DeepLab('drn', 16, 4, False)

args = argument_parser.parse_args()

simple_transform = transforms.Compose([
                        transforms.Resize((480, 640)),
                        transforms.ToTensor()])
valid_dataset = SegColDataset(args.root_dir, 
                                args.valid_img_file, args.valid_segm_file, 
                                simple_transform)

checkpoint = torch.load('*___*/checkpoint.pth.tar')
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to('cuda:0')

valid_loader = DataLoader(valid_dataset, batch_size=40, shuffle=False)
args.batch_size = 40
test_dataset = SegColDataset(args.root_dir, 
                                  args.valid_img_file, args.valid_segm_file, 
                                  simple_transform)
                                #   get_transform('val', (640, 480)))

saver = Saver(args, timestamp=get_curtime())
trainer = Trainer(args, model, valid_dataset, valid_dataset, saver)
with torch.no_grad():
    # epoch = trainer.load_best_checkpoint()
    
    if SAVE_PREDICTED_RESULTS:
      # To save the results!
      for samples in tqdm(valid_loader):
          outputs = model(samples['img'].to('cuda:0'))
          save_image_batch_to_disk(outputs, 'results/', samples['filename'])
    else:
      # To display the current performance of your model:
      trainer.validation(checkpoint)
      print('Valid: best Dice:', trainer.best_dice, 'AP:', trainer.best_ap)