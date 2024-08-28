import torch
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_transform
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from utils.misc import *
from torchvision import transforms

class CoreSetSelector:
    def __init__(self, dataset, img_size, feature_dim):
        self.dataset = dataset
        self.img_size = img_size
        self.feature_dim = feature_dim  # mobilenet: 304

    def _updated_distances(self, cluster_centers, features, min_distances):
        
        x = features[cluster_centers, :]  # 
        # dist = pairwise_distances(features, x, metric='l2')  
        dist = pairwise_distances(features, x, metric='cosine')  # vaal

        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def _select_batch(self, features, selected_idxs, to_select_num): 
        """
        :param features: all image embedded features
        :param selected_idxs: labelset img idxs
        :param to_select_num: budget,
        :return:
        """
        new_select_idxs = []
        for _ in range(to_select_num):
            all_centers = selected_idxs + new_select_idxs  
            min_distances = self._updated_distances(all_centers, features, None)
            idx = np.argmax(min_distances) 
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert idx not in selected_idxs  
            new_select_idxs.append(idx)
            # print('Maximum distance from cluster centers is %0.5f' % max(min_distances))

        return new_select_idxs

    @torch.no_grad()
    def select_next_batch(self, model, active_trainset, select_num):
        model.eval()
        model.set_return_features(True)  # reture features

        # get a subset from the whole unlabelset
        subset_img_paths, subset_target_paths, remset_img_paths, remset_target_paths = get_subset_paths(
            active_trainset.unlabel_img_paths, active_trainset.unlabel_target_paths, sub_ratio=0.4,
        )

        # cluster centers
        total_img_paths = active_trainset.label_img_paths + subset_img_paths
        total_target_paths = active_trainset.label_target_paths + subset_target_paths
        simple_transform = transforms.Compose([
                            transforms.Resize((480, 640)),
                            transforms.ToTensor()])
        total_dataset = BaseDataset(total_img_paths, total_target_paths, simple_transform)

        # total_dataset.transform = get_transform('test', base_size=self.img_size)

        dataloader = DataLoader(total_dataset,
                                batch_size=8, shuffle=False,
                                pin_memory=True, num_workers=4)

        # image embedded features
        features = np.zeros((0, self.feature_dim))
        tbar = tqdm(dataloader, desc='\r')
        tbar.set_description('feature extracting')
        for sample in tbar:
            img = sample['img'].cuda()
            _, feat = model(img)
            feat = F.avg_pool2d(feat, feat.size()[2:])  # [B, 304, 1, 1]
            features = np.vstack((features, to_numpy(feat)))

        model.set_return_features(False)

        already_selected_idxs = list(range(len(active_trainset.label_img_paths)))

        select_idxs = self._select_batch(features, already_selected_idxs, select_num)
        remain_idxs = list(set(range(len(total_img_paths))) - set(already_selected_idxs) - set(select_idxs))

        # select
        select_img_paths, select_target_paths = [], []
        for i in select_idxs:
            select_img_paths.append(total_img_paths[i])
            if subset_target_paths != []:
                select_target_paths.append(total_target_paths[i])

        remain_img_paths, remain_target_paths = [], []
        for i in remain_idxs:
            remain_img_paths.append(total_img_paths[i])
            if subset_target_paths != []:
                remain_target_paths.append(total_target_paths[i])
        remain_img_paths += remset_img_paths  
        remain_target_paths += remset_target_paths

        active_trainset.add_by_select_remain_paths(select_img_paths, select_target_paths,
                                                   remain_img_paths, remain_target_paths)
        return select_img_paths