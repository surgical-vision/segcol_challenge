import numpy as np
from PIL import Image
from metrics import AP, ODS, OIS, compute_dice, compute_CLDice
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
folder = "data/output/Seq*/predictions/" # change according to final structure
# if we want the results per seg add another for loop


# get pred_list and gt_list by reading images from the directory
# the prediction data should be multichannel/class (4) tif files.
for file in glob.glob(folder + "*.npy"):
    pred_list.append(get_images(file, npy = True))
    gt_list.append(get_images(file.replace("output", "input").replace("predictions", "segm_maps").replace(".npy", ".png")))


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


'''
dice_score         [0.00644047 0.         0.13483548 0.        ]
optimal_thresholds [0.55148822 1.         0.46746114 1.        ]
ap                 [ 0.0028987 -0.       0.07122158 -0.        ]
clDice_score       [0.00633226 0.         0.10956853 0.        ]
ods 0.01131080926255049
ois 0.01131809195190073

dice_score         [0.03177795 0.         0.4406434  0.        ]
optimal_thresholds [0.50495468 1.         0.48528391 1.        ]
ap                 [ 0.02862942 -0.          0.3087773  -0.        ]
clDice_score       [0.03009712 0.         0.17968246 0.        ]
ods 0.344925629307999
ois 0.34524105547086176
'''