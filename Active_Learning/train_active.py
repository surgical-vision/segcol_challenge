import argument_parser
from pprint import pprint

args = argument_parser.parse_args()
pprint(vars(args))

import os

if len(args.gpu_ids) > 1:
    args.sync_bn = True

import torch
# from torch.utils.tensorboard import SummaryWriter

from model.deeplab import DeepLab
# from utils.calculate_weights import calculate_class_weights
from utils.saver import Saver
from utils.trainer import Trainer
from utils.misc import get_curtime, write_list_to_txt
from pprint import pprint
from active_selection import get_active_selector
import shutil
import random
import math
import glob
import numpy as np
from torchvision import transforms
from datasets.dataset import SegColDataset, ActiveSegCol

def is_interval(epoch):
    return epoch % args.eval_interval == (args.eval_interval - 1)


def main():
    random.seed(args.seed)  # active trainset

    simple_transform = transforms.Compose([
                            transforms.Resize((480, 640)),
                            transforms.ToTensor()])
    active_trainset = ActiveSegCol(args.root_dir, 
                                  args.train_img_file, args.train_segm_file, 
                                  transform=simple_transform, split=args.init_percent)
    valid_dataset = SegColDataset(args.root_dir, 
                                  args.valid_img_file, args.valid_segm_file, 
                                  simple_transform)
    if args.resume_dir and args.resume_percent:  # 
        iter_dir = f'runs/{args.dataset}/{args.resume_dir}/runs_0{args.resume_percent}'
        active_trainset.add_preselect_data(iter_dir)  # add preselect data, and update label/unlabel data

    # global writer
    timestamp = get_curtime()


    active_selector = get_active_selector(args)

    # budget 
    select_num = args.select_num
    if select_num is None:
        if args.percent_step:  
            select_num = math.ceil(active_trainset.__len__() * args.percent_step / 100)
        else:
            raise ValueError('must set select_num or percent_step')

    start_percent = args.resume_percent if args.resume_percent else args.init_percent
    active_trainset.update_iter_img_paths()
    for percent in range(start_percent, args.max_percent + 1, args.percent_step):
        run_id = f'runs_{percent:03d}'
        print(run_id)


        ## ------------ begin training with current percent data ------------

        # saver/writer of each iteration
        saver = Saver(args, exp_dir=args.resume_dir, timestamp=timestamp, suffix=run_id)
        # writer = SummaryWriter(saver.exp_dir)
        # save current data path -> train model -> select new data 
        write_list_to_txt(active_trainset.label_img_paths, txt_path=os.path.join(saver.exp_dir, 'label_imgs.txt'))
        write_list_to_txt(active_trainset.label_target_paths, txt_path=os.path.join(saver.exp_dir, 'label_targets.txt'))

        # create model from scratch
        model = DeepLab(args.backbone, args.out_stride, active_trainset.class_count, args.sync_bn)
        
        trainer = Trainer(args, model, active_trainset, valid_dataset, saver)

        # train/valid
        for epoch in range(args.epochs):
            trainer.training(epoch)
            if is_interval(epoch):
                trainer.validation(epoch)
        print('Valid: best DSC:', trainer.best_dice, 'best AP:', trainer.best_ap)


        # end active training
        if percent == args.max_percent:
            print('end active training')
            break

        # select samples
        active_selector.select_next_batch(trainer.model, active_trainset, select_num)


if __name__ == '__main__':
    main()
