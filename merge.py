
from __future__ import print_function

import argparse
import os
import time, platform
import numpy as np
import cv2


def numpy_load(file_path1, file_path2):
    array1 = np.load(file_path1)
    array2 = np.load(file_path2)
    return array1, array2
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed merge')
    
    # Data parameters
    parser.add_argument('--folder1',
                        type=str,
                        default=None,
                        help='the path to the directory with the output from folds, 1 channel.')
    parser.add_argument('--folder2',
                        type=str,
                        default=None,
                        help='the path to the directory with the output from tool, 4 channels.')
    parser.add_argument('--output_merge_dir',
                        type=str,
                        default='./merged_results',
                        help='the path to output the results.')
    
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print('Merge two numpy')

    # Define folder path
    merged_folder_path = args.output_merge_dir

    if not os.path.exists(merged_folder_path):
        os.makedirs(merged_folder_path)
    merged_file_path = './merged_results/merged_array.npy'
    # merge
    array1 = np.random.rand(1, 480, 640)
    array2 = np.random.rand(4, 480, 640)
    # Change the first channel of the loaded second array to be the loaded first array
    array2[0, :, :] = array1[0, :, :]
    # Save 
    np.save(merged_file_path, array2)
    # print('shape', array2.shape)

    
    print('-------------------------------------------------------')
    print('Finish merge')
    print('-------------------------------------------------------')

if __name__ == '__main__':
    args = parse_args()
    main(args)

