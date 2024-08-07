import os
import re
import argparse

def sort_numerically(file_name):
    # Extract numbers from the filename
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[-1])

# read list of images in subfolders and create pairs from sorted list and sorted list in segm_maps instead of imgs
# output: list of pairs in a file
# input: path to the folder with subfolders with images and segm_maps
def create_pair_lists(path):
    # get list of subfolders
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    # sort subfolders
    subfolders.sort()
    # create list of pairs
    pairs = []
    for subfolder_full in subfolders:
        subfolder = os.path.relpath(subfolder_full, path)
        # get list of images
        imgs = [f for f in sorted(os.listdir(subfolder_full+"/imgs/"), key=sort_numerically) if f.endswith(('.jpg', '.png', '.tif', '.jpeg'))]
        # get list of segm_maps
        segm_maps = [f for f in sorted(os.listdir(subfolder_full+"/segm_maps/"), key=sort_numerically) if f.endswith(('.jpg', '.png', '.tif', '.jpeg'))]
        # create pairs
        for i in range(len(imgs)):
            pair = [os.path.join(subfolder+"/imgs/", imgs[i]), os.path.join(subfolder+"/segm_maps/", segm_maps[i])]
            pairs.append(pair)
    return pairs
  


if __name__ == "__main__":
    # argument input parser for data_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the folder with subfolders with images and segm_maps")
    args = parser.parse_args()
    
    for sub in ['train','valid']:
        path = f"{args.data_path}{sub}"
        pairs = create_pair_lists(path)
        
        print(pairs)

        # write list of pairs to a file
        with open(os.path.join(os.path.dirname(path),f"{sub}_pair_list.lst"), 'w') as f:
            for pair in pairs:
                f.write(pair[0] + ' ' + pair[1] + '\n')
            