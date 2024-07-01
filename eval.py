import numpy as np
from PIL import Image
import glob
import json
import os
import torch
from metrics import find_optimal_thresholds, check_for_zeros, OIS, ODS, compute_dice, AP, compute_CLDice

def load_image(file_path, npy=False):
    try:
        if npy:
            return np.load(file_path)
        else:
            img = Image.open(file_path)
            img_data = np.array(img)
            class_channels = [np.where(img_data == val, 1, 0) for val in [255, 127, 128, 129]]
            return np.stack(class_channels, axis=-1)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def evaluate_classes(pred_list, gt_list, num_classes):
    # calculate the dice score, optimal thresholds, AP, CLDice, ODS, OIS
    if num_classes == 2:
        pred_list = [np.stack([img[:,:,0], np.max(img[:,:,1:4], axis=2)], axis=2) for img in pred_list]
        gt_list = [np.stack([img[:,:,0], np.max(img[:,:,1:4], axis=2)], axis=2) for img in gt_list]
    
    optimal_thresholds_orig = find_optimal_thresholds(pred_list, gt_list, num_classes=num_classes)
    pred_list_handled, gt_list_handled = check_for_zeros(pred_list, gt_list, optimal_thresholds_orig, num_classes=num_classes)
    ois = OIS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=num_classes)
    ods, optimal_thresholds_ods = ODS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=num_classes)
    optimal_thresholds = [torch.tensor(npy_array[0], dtype=torch.float64) for npy_array in optimal_thresholds_ods]
    dice_score = compute_dice(pred_list_handled, gt_list_handled, optimal_thresholds)
    ap = AP(pred_list_handled, gt_list_handled, thresholds=list(np.array(optimal_thresholds)), num_classes=num_classes, average=None)
    clDice_score = compute_CLDice(pred_list_handled, gt_list_handled, optimal_thresholds, num_classes=num_classes)
    
    print(f"{num_classes} class evaluation:")
    print("ods", ods)
    print("ois", ois)
    print("dice_score", np.array(dice_score))
    print("optimal_thresholds", np.array(optimal_thresholds))
    print("ap", np.array(ap))
    print("clDice_score", np.array(clDice_score))

def main():
    folder = "data/output/Seq*/predictions/" # change according to final structure

    # get pred_list and gt_list by reading images from the directory
    # the prediction data should be multichannel/class (4) tif files.
    files = sorted(glob.glob(folder + "*.npy"))
    pred_list = [load_image(file, npy = True) for file in files]
    gt_list = [load_image(file.replace("output", "input").replace("predictions", "segm_maps").replace(".npy", ".png")) for file in files]
    
    evaluate_classes(pred_list, gt_list, 2)
    evaluate_classes(pred_list, gt_list, 4)

if __name__ == "__main__":
    main()