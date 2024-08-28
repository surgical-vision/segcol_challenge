from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import datasets.transforms as tr
from utils.vis import get_label_name_colors
from utils.misc import read_txt_as_list
import os
import torch

class BaseDataset(Dataset):
    def __init__(self,
                 img_paths, target_paths, transform=None,
                 split=None, base_size=None, crop_size=None, **kwargs):
        """
        base dataset, with img/target paths
        """
        self.image_files = img_paths
        self.anno_files = target_paths
        self.len_dataset = len(self.image_files)

        self.base_size = base_size  # train size
        self.crop_size = crop_size  # train, valid, test

        self.split = split
        self.transform = transform

        self.class_count = 4
        self.bg_idx = kwargs.get('bg_idx', -1)
        self.mapbg_fn = tr.mapbg(self.bg_idx)
        self.remap_fn = tr.remap(self.bg_idx)

        if 'csv_path' in kwargs:
            self.label_names, self.label_colors = get_label_name_colors(kwargs['csv_path'])
    
    def rgb_to_multi_channel(self, annotation):
        # Convert RGB values to multi-channel class indices
        annotation_np = np.array(annotation).transpose(0,1,2)
        height, width, _ = annotation_np.shape
        class_map = np.zeros((self.class_count, height, width), dtype=np.float32)
        
        # Assuming specific RGB values for each class
        color_to_class = {
            (255, 255, 255): 0,   # Colon folds
            (127, 127, 127): 1,   # balloon
            (128, 128, 128): 2,   # captivator
            (129, 129, 129): 3    # forceps
        }

        for rgb, class_idx in color_to_class.items():
            mask = (annotation_np == rgb).all(axis=2)
            class_map[class_idx, mask] = 1

        return torch.from_numpy(class_map)


    def __getitem__(self, index):
        img_name = self.image_files[index]
        if index >= len(self.anno_files):
            annotation = []
        else:
            segm_map = self.anno_files[index]
            annotation = Image.open(segm_map).convert("RGB")
        image = Image.open(img_name).convert("RGB")
        
        
        sample = {'img': image, 'target': annotation, 'filename': img_name}
        
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        if index >= len(self.anno_files):
            sample['target'] = []
        else:
            sample['target'] = self.rgb_to_multi_channel(sample['target'])
        
        return sample

    def __len__(self):
        return len(self.image_files)

    def get_transform(self, split):
        return None

    def make_dataset_multiple_of_batchsize(self, batch_size):
        remainder = self.len_dataset % batch_size
        if remainder > 0:
            num_new_entries = batch_size - remainder
            self.img_paths.extend(self.img_paths[:num_new_entries])
            self.target_paths.extend(self.target_paths[:num_new_entries])

    def reset_dataset(self):
        self.img_paths = self.img_paths[:self.len_dataset]
        self.target_paths = self.target_paths[:self.len_dataset]


class ActiveBaseDataset(BaseDataset):
    def __init__(self,
                 label_img_paths, label_target_paths,
                 unlabel_img_paths, unlabel_target_paths,
                 split,
                 base_size, crop_size, **kwargs):
        """
        Active base dataset, with label/unlabel img/target paths
        """
        super().__init__(label_img_paths, label_target_paths,
                         split,
                         base_size, crop_size, **kwargs)
        self.base_img_paths = label_img_paths
        self.base_target_paths = label_target_paths

        self.label_img_paths = label_img_paths[:]  # base_img_paths
        self.label_target_paths = label_target_paths[:]

        self.unlabel_img_paths = unlabel_img_paths
        self.unlabel_target_paths = unlabel_target_paths

    def add_by_select_remain_paths(self,
                                   select_img_paths, select_target_paths,
                                   remain_img_paths, remain_target_paths):
        """
            label_data  += select data
            unlabel_data = remain data
        """
        self.label_img_paths += select_img_paths
        self.label_target_paths += select_target_paths

        self.unlabel_img_paths = remain_img_paths
        self.unlabel_target_paths = remain_target_paths

        self.update_iter_img_paths()

    def add_by_select_unlabel_idxs(self, select_idxs):
        """
        :param select_idxs: 从 unlabel idx 划分样本
        :return:
        """
        remain_idxs = set(range(len(self.unlabel_img_paths))) - set(select_idxs)

        select_img_paths = [self.unlabel_img_paths[i] for i in select_idxs]
        select_target_paths = [self.unlabel_target_paths[i] for i in select_idxs]

        self.label_img_paths += select_img_paths
        self.label_target_paths += select_target_paths

        self.unlabel_img_paths = [self.unlabel_img_paths[i] for i in remain_idxs]
        self.unlabel_target_paths = [self.unlabel_target_paths[i] for i in remain_idxs]

        self.update_iter_img_paths()

    def add_preselect_data(self, iter_dir):
        """
        :param iter_dir: read preselect data paths from iter_dir, and update label/unlabel data
        """
        # preselect label data
        label_img_paths = read_txt_as_list(os.path.join(iter_dir, 'label_imgs.txt'))
        label_target_paths = read_txt_as_list(os.path.join(iter_dir, 'label_targets.txt'))
        label_data = set(label_img_paths)

        # remain_unlabel = ori_unlabel - preselect_label
        remain_img_paths, remain_target_paths = [], []
        for i in range(len(self.unlabel_img_paths)):
            if self.unlabel_img_paths[i] not in label_data:
                remain_img_paths.append(self.unlabel_img_paths[i])
                remain_target_paths.append(self.unlabel_target_paths[i])

        # update active_trainset
        self.update_label_unlabel_paths(label_img_paths, label_target_paths,
                                        remain_img_paths, remain_target_paths)

    def update_label_unlabel_paths(self,
                                   label_img_paths, label_target_paths,
                                   unlabel_img_paths, unlabel_target_paths):
        """
            reset label/unlabel path, for resume training
        """
        self.label_img_paths = label_img_paths
        self.label_target_paths = label_target_paths

        self.unlabel_img_paths = unlabel_img_paths
        self.unlabel_target_paths = unlabel_target_paths

        self.update_iter_img_paths()

    def update_iter_img_paths(self):  # train get_item
        self.img_paths = self.label_img_paths
        self.target_paths = self.label_target_paths
