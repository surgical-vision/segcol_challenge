import numpy as np
from PIL import Image
from metrics import AP, ODS, OIS, compute_dice, compute_CLDice
import glob


def get_images(file, tiff=False):
    # Open the image
    img = Image.open(file)

    # Convert the image data to a numpy array
    img_data = np.array(img)

    if not tiff:
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
folder = "segcol_challenge/Seq*/predictions/" # change according to final structure
# if we want the results per seq add another for loop


# get pred_list and gt_list by reading images from the directory
# the prediction data should be multichannel/class (4) tif files.
for file in glob.glob(folder + "*/*.tif"):
    pred_list.append(get_images(file, tiff = True))
    gt_list.append(get_images(file.replace("predictions", "seqm_maps").replace(".tif", ".png")))


# calculate the dice score, optimal thresholds, AP, CLDice, ODS, OIS
dice_score, optimal_thresholds = compute_dice(pred_list, gt_list)
ap = AP(pred_list, gt_list, thresholds = list(np.array(optimal_thresholds)), num_classes = 4, average = None)
clDice_score = compute_CLDice(pred_list, gt_list, optimal_thresholds)
print("dice_score", np.array(dice_score))
print("optimal_thresholds", np.array(optimal_thresholds))
print("ap", np.array(ap))
print("clDice_score", np.array(clDice_score))

pred_list_fold = [img[:,:,0] for img in pred_list]
gt_list_fold = [img[:,:,0] for img in gt_list]

ods = ODS(pred_list_fold, gt_list_fold, thresh_list)
ois = OIS(pred_list_fold, gt_list_fold, thresh_list)
print("ods", ods)
print("ois", ois)