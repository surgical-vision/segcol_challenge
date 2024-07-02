import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class SegColDataset(Dataset):
    def __init__(self, root_dir, image_list_path, segm_list_path, transform=None):
        self.transform = transform
        self.class_count = 4 
        self.root_dir = root_dir
        with open(image_list_path, 'r') as file:
            self.image_files = file.readlines()
        with open(segm_list_path, 'r') as file:
            self.anno_files = file.readlines()
        self.image_files = [line.strip() for line in self.image_files]
        self.anno_files = [line.strip() for line in self.anno_files]

    def __len__(self):
        return len(self.image_files)

    def rgb_to_multi_channel(self, annotation):
        # Convert RGB values to multi-channel class indices
        annotation_np = np.array(annotation).transpose(1,0,2)
        height, width, _ = annotation_np.shape
        class_map = np.zeros((self.class_count, height, width), dtype=np.float32)
        
        # Assuming specific RGB values for each class
        color_to_class = {
            (255, 255, 255): 0,   # Colon folds
            (127, 127, 127): 1,   # balloon
            (128, 128, 128): 2,   # captivator
            (129, 129, 129): 3  # forceps
        }

        for rgb, class_idx in color_to_class.items():
            mask = (annotation_np == rgb).all(axis=2)
            class_map[class_idx, mask] = 1

        return torch.from_numpy(class_map)


    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        segm_map = self.anno_files[idx]
        
        image = Image.open(img_name).convert("RGB")
        annotation = Image.open(segm_map).convert("RGB") 
        
        sample = {'img': image,
                  'target': annotation}
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        sample['target'] = self.rgb_to_multi_channel(sample['target'])
        return sample
    
def main():
    root_dir = '/media/razvan/segcol/'
    train_img_file = '/media/razvan/segcol/train/train_list.csv'
    train_segm_file = '/media/razvan/segcol/train/train_segmentation_maps.csv'
    transform = transforms.Compose([
    transforms.Resize((640, 480)),
    transforms.ToTensor()
        ])
    train_dataset = SegColDataset(root_dir, train_img_file, train_segm_file, transform=transform)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    for epoch in range(2):
        # Training loop
        for images, annotations in train_loader:
            # Train your model
            pass

if __name__ == '__main__':
    main()