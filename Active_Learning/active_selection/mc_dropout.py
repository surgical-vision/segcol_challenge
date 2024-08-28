import torch
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform
from tqdm import tqdm
import numpy as np
import constants
from torch.utils.data import DataLoader
from utils.misc import *


class MCDropoutEntropySelector:

    def __init__(self, dataset, img_size, num_classes):
        self.dataset = dataset
        self.img_size = img_size
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def cal_entropy(output):  # C,H,W  
        return -np.nansum(np.multiply(output, np.log(output + 1e-12)), axis=0)

    def cal_vote_entropy_score(self, model, input, vote):
        B = input.shape[0]  # batch size
        entropy_maps = torch.FloatTensor(B, input.shape[2], input.shape[3]).fill_(0).to(self.device)

        # MC outputs
        if vote == 'hard':  
            outputs = torch.FloatTensor(B, constants.MC_STEPS, input.shape[2], input.shape[3]).to(self.device)  # (B,20,H,W)
            for step in range(constants.MC_STEPS):  # 20次 argmax 类别计数
                outputs[:, step, :, :] = torch.argmax(model(input), dim=1)  # B,h,w

            for i in range(B):  # outputs: B,20,H,W
                # entropy
                for c in range(self.num_classes):  # C
                    # MC_STEPS 
                    p = torch.sum(outputs[i] == c, dim=0, dtype=torch.float32) / constants.MC_STEPS  # H,W
                    entropy_maps[i] = entropy_maps[i] - (p * torch.log2(p + 1e-12))  # hard prob p=0

        elif vote == 'soft':
            outputs = torch.FloatTensor(B, self.num_classes, input.shape[2], input.shape[3]).fill_(0).to(self.device)  # (B,C,H,W)
            for step in range(constants.MC_STEPS):  # 20 softmax probs 
                probs = self.softmax(model(input))  # B,C,H,W
                outputs += probs  
            outputs = outputs / constants.MC_STEPS  

            for i in range(B):
                for c in range(self.num_classes):
                    p = outputs[i, c, :, :]  # outputs 
                    entropy_maps[i] = entropy_maps[i] - (p * torch.log2(p + 1e-12))
        else:
            raise NotImplementedError

        entropy_maps = entropy_maps.detach().cpu().numpy()
        batch_scores = [np.mean(m) for m in entropy_maps]
        return batch_scores

    @torch.no_grad()
    def select_next_batch(self, model, active_trainset, select_num, vote='hard'):  # vote: hard,soft
        model.eval()
        model.apply(turn_on_dropout)

        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            active_trainset.unlabel_img_paths, active_trainset.unlabel_target_paths, sub_ratio=0.4,
        )
        print('subset_img_paths', len(subset_img_paths))
        print('remset_img_paths', len(remset_img_paths))
        unlabelset = BaseDataset(subset_img_paths, subset_target_paths)
        unlabelset.transform = get_transform('test', base_size=self.img_size)

        dataloader = DataLoader(unlabelset,
                                batch_size=8, shuffle=False,
                                pin_memory=True, num_workers=4)

        scores = []
        tbar = tqdm(dataloader, desc='\r')
        tbar.set_description(f'{vote} vote_entropy_score')

        for sample in tbar:
            img = sample['img'].cuda()
            scores += self.cal_vote_entropy_score(model, img, vote)

        select_idxs = get_topk_idxs(scores, select_num)


        select_img_paths, select_target_paths, remain_img_paths, remain_target_paths = get_select_remain_paths(
            subset_img_paths, subset_target_paths, select_idxs
        )

        remain_img_paths += remset_img_paths
        remain_target_paths += remset_target_paths
        print('select_img_paths', len(select_img_paths))
        print('remain_img_paths', len(remain_img_paths))


        active_trainset.update_label_unlabelset(select_img_paths, select_target_paths,
                                                remain_img_paths, remain_target_paths)
