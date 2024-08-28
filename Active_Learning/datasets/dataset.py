import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from utils.misc import read_txt_as_list, get_select_paths_by_idxs


# class SegColDataset(Dataset):
#     def __init__(self, root_dir, image_list_path, segm_list_path, transform=None, split=None):
#         self.transform = transform
#         self.split = split
#         self.class_count = 4 
#         self.root_dir = root_dir
#         with open(image_list_path, 'r') as file:
#             self.image_files = file.readlines()
#         with open(segm_list_path, 'r') as file:
#             self.anno_files = file.readlines()
#         self.image_files = [line.strip() for line in self.image_files]
#         self.anno_files = [line.strip() for line in self.anno_files]
            
#     def __len__(self):
#         return len(self.image_files)

#     def rgb_to_multi_channel(self, annotation):
#         # Convert RGB values to multi-channel class indices
#         annotation_np = np.array(annotation).transpose(0,1,2)
#         height, width, _ = annotation_np.shape
#         class_map = np.zeros((self.class_count, height, width), dtype=np.float32)
        
#         # Assuming specific RGB values for each class
#         color_to_class = {
#             (255, 255, 255): 0,   # Colon folds
#             (127, 127, 127): 1,   # balloon
#             (128, 128, 128): 2,   # captivator
#             (129, 129, 129): 3  # forceps
#         }

#         for rgb, class_idx in color_to_class.items():
#             mask = (annotation_np == rgb).all(axis=2)
#             class_map[class_idx, mask] = 1

#         return torch.from_numpy(class_map)


#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         segm_map = self.anno_files[idx]
        
#         image = Image.open(img_name).convert("RGB")
#         annotation = Image.open(segm_map).convert("RGB") 
        
#         sample = {'img': image,
#                   'target': annotation,
#                   'filename':img_name }
            
#         if self.transform:
#             # augmented = self.transform(image=np.array(sample['img']), 
#             #                            mask=np.array(sample['target']))
#             # sample['img'] = augmented['image']
#             sample['img']  = self.transform(sample['img'] )
#             # sample['target'] = augmented['mask']
#         sample['target'] = self.rgb_to_multi_channel(sample['target'])
#         return sample

class SegColDataset(Dataset):
    def __init__(self, root_dir, image_list_path, 
                 segm_list_path, transform=None, split=None, 
                 index_list_path=None):
        self.transform = transform
        self.split = split
        self.class_count = 4
        self.root_dir = root_dir
        
        # Load the list of images and annotations
        with open(image_list_path, 'r') as file:
            self.image_files = file.readlines()
        if segm_list_path==[]:
            # No annotation available for the unlabelled
            self.anno_files = []
        else:
            with open(segm_list_path, 'r') as file:
                self.anno_files = file.readlines()
            self.anno_files = [line.strip() for line in self.anno_files]
        self.image_files = [os.path.join(root_dir, line.strip()) for line in self.image_files]
        

        # Filter based on indices in the index_list_path, if provided
        if index_list_path is not None:
            with open(index_list_path, 'r') as file:
                indices = file.readlines()
            indices = [int(idx.strip()) for idx in indices]
            
            self.image_files = [self.image_files[i] for i in indices]
            self.anno_files = [self.anno_files[i] for i in indices]
            
    def __len__(self):
        return len(self.image_files)

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

    def get_select_paths_by_idxs(img_paths, target_paths, select_idxs):
        select_img_paths, select_target_paths = [], []
        for idx in select_idxs:
            select_img_paths.append(img_paths[idx])
            select_target_paths.append(target_paths[idx])
        return select_img_paths, select_target_paths

    # @staticmethod
    def random_split_train_data(self, percent):
        img_idxs = list(range(len(self.image_files)))
        random.shuffle(img_idxs)


        init_select_num = 300 if percent == 10 else round(len(self.image_files) * percent / 100)

        label_idxs, unlabel_idxs = img_idxs[:init_select_num], img_idxs[init_select_num:]
        label_img_paths, label_target_paths = get_select_paths_by_idxs(self.image_files, 
                                                                       self.anno_files, label_idxs)
        unlabel_img_paths, unlabel_target_paths = get_select_paths_by_idxs(self.image_files, 
                                                                           self.anno_files, unlabel_idxs)

        return label_img_paths, label_target_paths, unlabel_img_paths, unlabel_target_paths

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        segm_map = self.anno_files[idx]
        
        image = Image.open(img_name).convert("RGB")
        annotation = Image.open(segm_map).convert("RGB")
        
        sample = {'img': image, 'target': annotation, 'filename': img_name}
        
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        
        sample['target'] = self.rgb_to_multi_channel(sample['target'])
        
        return sample

class ActiveSegCol(SegColDataset):
    def __init__(self, root_dir, image_list_path, 
                 segm_list_path,
                 label_img_paths=None,
                 label_target_paths=None,
                 transform=None, split=None,  **kwargs):
        """
        Active base dataset, with label/unlabel img/target paths
        """

        super().__init__(root_dir, image_list_path,
                         segm_list_path,
                         transform, split,  **kwargs)
        
        # split data
        if split==0:
            unlabel_img_paths = self.image_files
            unlabel_target_paths = []
        else:
            label_img_paths, label_target_paths, unlabel_img_paths, unlabel_target_paths = \
                self.random_split_train_data(split)

        self.base_img_paths = image_list_path
        self.base_target_paths = segm_list_path

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
        self.image_files = self.label_img_paths
        self.anno_files = self.label_target_paths

def main():
    root_dir = ''
    train_img_file = 'train/train_list.csv'
    train_segm_file = 'train/train_segmentation_maps.csv'
    transform = transforms.Compose([
    transforms.Resize((640, 480)),
    transforms.ToTensor()
        ])
    train_dataset = SegColDataset(root_dir, train_img_file, train_segm_file, transform=transform)
    active_trainset = ActiveSegCol(root_dir, 
                                  train_img_file, train_segm_file, 
                                  transform)
    # Create DataLoaders
    train_loader = DataLoader(active_trainset, batch_size=32, shuffle=True, num_workers=4)
    for epoch in range(2):
        # Training loop
        for images, annotations in train_loader:
            # Train your model
            pass


if __name__ == '__main__':
    main()
    
