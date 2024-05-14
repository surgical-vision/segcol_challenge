import numpy as np
from PIL import Image
from metrics import *
import glob


def get_images(file, npy=False):
    if npy:
        img_data = np.load(file)


    if not npy:
        # Open the image
        img = Image.open(file)

        # Convert the image data to a numpy array
        img_data = np.array(img)

        # Create class channels using np.where
        class1 = np.where(img_data == 255, 1, 0) # fold
        class2 = np.where(img_data == 127, 1, 0) # tool1 
        class3 = np.where(img_data == 128, 1, 0) # tool2
        class4 = np.where(img_data == 129, 1, 0) # tool3

        # Stack all class channels together
        img_data = np.stack((class1, class2, class3, class4), axis=-1)

    return img_data

# initialize lists and variables                         
pred_list = []
gt_list = []
pred_list_orig = []
thresh_list = np.linspace(0.01, 0.99, 99)
folder = "/home/xinwei/segcol_challenge/data/output/Seq*/predictions/" # change according to final structure
num_classes = 4
# if we want the results per seg add another for loop


# get pred_list and gt_list by reading images from the directory
# the prediction data should be multichannel/class (4) tif files.
for file in sorted(glob.glob(folder + "*.npy")):
    pred_list.append(get_images(file, npy = True))
    gt_list.append(get_images(file.replace("output", "input").replace("predictions", "segm_maps").replace(".npy", ".png")))


#----------------------------------------------------------------------------------------------------
print('Four class evaluation:')
# calculate the dice score, optimal thresholds, AP, CLDice, ODS, OIS
optimal_thresholds = find_optimal_thresholds(pred_list, gt_list, num_classes = num_classes)
pred_list_handled, gt_list_handled = check_for_zeros(pred_list, gt_list, optimal_thresholds, num_classes = num_classes)
dice_score = compute_dice(pred_list_handled, gt_list_handled, optimal_thresholds)
ap = AP(pred_list_handled, gt_list_handled, thresholds = list(np.array(optimal_thresholds)), num_classes = num_classes, average = None)
clDice_score = compute_CLDice(pred_list, gt_list, optimal_thresholds, num_classes = num_classes)
print("dice_score", np.array(dice_score))
print("optimal_thresholds", np.array(optimal_thresholds))
print("ap", np.array(ap))
print("clDice_score", np.array(clDice_score))

ods = ODS(pred_list, gt_list, thresh_list, num_classes=num_classes)
ois = OIS(pred_list, gt_list, thresh_list, num_classes=num_classes)
print("ods", ods)
print("ois", ois)


#----------------------------------------------------------------------------------------------------
print('Two class evaluation:')
# tool considered as 1 class
pred_list_2class = [np.stack([img[:,:,0], np.max(img[:,:,1:4], axis=2)], axis=2) for img in pred_list]
gt_list_2class = [np.stack([img[:,:,0], np.max(img[:,:,1:4], axis=2)], axis=2) for img in gt_list]
num_classes = 2

# calculate the dice score, optimal thresholds, AP, CLDice, ODS, OIS
optimal_thresholds_2class = find_optimal_thresholds(pred_list_2class, gt_list_2class, num_classes = num_classes)
pred_list_handled_2class, gt_list_handled_2class = check_for_zeros(pred_list_2class, gt_list_2class, optimal_thresholds_2class, num_classes = num_classes)
dice_score_2class = compute_dice(pred_list_handled_2class, gt_list_handled_2class, optimal_thresholds_2class)
ap_2class = AP(pred_list_handled_2class, gt_list_handled_2class, thresholds = list(np.array(optimal_thresholds_2class)), num_classes = num_classes, average = None)
clDice_score_2class = compute_CLDice(pred_list_2class, gt_list_2class, optimal_thresholds_2class, num_classes = num_classes)
print("dice_score", np.array(dice_score_2class))
print("optimal_thresholds", np.array(optimal_thresholds_2class))
print("ap", np.array(ap_2class))
print("clDice_score", np.array(clDice_score_2class))

ods_2class = ODS(pred_list_2class, gt_list_2class, thresh_list, num_classes=num_classes)
ois_2class = OIS(pred_list_2class, gt_list_2class, thresh_list, num_classes=num_classes)
print("ods", ods_2class)
print("ois", ois_2class)