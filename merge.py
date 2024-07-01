
from __future__ import print_function

import argparse
import os
import time, platform
import numpy as np
import cv2
import glob
import tifffile as tiff


def numpy_load(file_path1, file_path2):
    array1 = np.load(file_path1)
    array2 = np.load(file_path2)
    return array1, array2
    

def main(folder1, folder2, output_merge_dir):
    """Main function."""

    print('Merge two numpy')

    # Define folder path
    merged_folder_path = output_merge_dir

    if not os.path.exists(merged_folder_path):
        os.makedirs(merged_folder_path)
    for file in glob.glob(os.path.join(folder1,'*','imgs','*.npy')):
        print('file', file)
        file_path1 = file
        file_path2 = file.replace(folder1, folder2)
        print('file_path2', file_path2)
        array1, array2 = numpy_load(file_path1, file_path2)
        # Change the first channel of the loaded second array to be the loaded first array
        merged_array = array2.copy()
        merged_array[0, :, :] = array1
        # Save
        merged_file_path = file.replace(folder1, merged_folder_path)
        os.makedirs(os.path.dirname(merged_file_path), exist_ok=True)
        np.save(merged_file_path, merged_array.transpose(1, 2, 0))
        print('shape', merged_array.shape)

    
    print('-------------------------------------------------------')
    print('Finish merge')
    print('-------------------------------------------------------')

# merge two folders of numpy files such that the first channel of the second folder is replaced by the first folder
folder1 = "data/output/Seq*/predictions1/"
folder2 = "data/output/Seq*/predictions2/"
output_merge_dir = "data/output/Seq*/predictions/"
main(folder1, folder2, output_merge_dir)

