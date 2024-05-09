# for tool: F1 score
# for edge: AP, OIS, ODS, CLDice
# for ranking: mAP

from torchmetrics.functional.classification import multilabel_average_precision, dice
from torchmetrics.classification import MultilabelROC
import numpy as np

# import cv2
import torch
from torchvision import transforms

from skimage.morphology import skeletonize, skeletonize_3d


# Dice (torchmetrics.classification.Dice)
# Dice = 2*TP / (2*TP + FP + FN)

# AP (torchmetrics.classification.AveragePrecision)

# OIS, ODS (https://github.com/lllyasviel/DanbooRegion/blob/master/code/ap_ois_ods/ap_ois_ods.py)

'''def compute_precision(ground_truth_region_map, estimated_region_map, threshold):
    ground_truth_edge_map = ground_truth_region_map > threshold 
    estimated_edge_map = estimated_region_map > threshold
    return np.sum(ground_truth_edge_map * estimated_edge_map) / np.sum(estimated_edge_map)


def AP(image_list, threshold):
    ap = 0.0
    for img_path in image_list:
        ground_truth = cv2.imread(img_path + '.ground_truth.png')
        estimation = cv2.imread(img_path + '.estimation.png')
        ap += compute_precision(ground_truth, estimation, threshold)
    ap /= float(len(image_list))
    return ap'''



# Function to find the optimal thresholds using Youden's Index
# compute dice score for each class
def compute_dice(pred_list, gt_list):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor with values in [0, 1]
    ])
    gt_list_tensor = [None] * len(pred_list)
    pred_list_tensor = gt_list_tensor.copy()
    for i in range(len(pred_list)):
        gt_list_tensor[i] = transform(gt_list[i])
        pred_list_tensor[i] = transform(pred_list[i])
    # input shape: [n, num_classes, w, h]
    gt_list_tensor = torch.stack(gt_list_tensor, dim=0)
    pred_list_tensor = torch.stack(pred_list_tensor, dim=0)
    metric = MultilabelROC(num_labels=4, thresholds=None)
    fpr, tpr, thresholds = metric(pred_list_tensor, gt_list_tensor)
    optimal_thresholds = []
    dice_list = []
    for i in range(len(tpr)):
        J = tpr[i] - fpr[i]  # Youden's index
        idx = np.argmax(J)
        optimal_thresholds.append(thresholds[i][idx])
        dice_list.append(dice(pred_list_tensor[:,i,:,:], gt_list_tensor[:,i,:,:], threshold = thresholds[i][idx]))
    return dice_list, optimal_thresholds


def AP(pred_list, gt_list, thresholds, num_classes = 4, average = None):
    # Define a transformation to convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor with values in [0, 1]
    ])
    gt_list_tensor = [None] * len(pred_list)
    pred_list_tensor = gt_list_tensor.copy()
    for i in range(len(pred_list)):
        gt_list_tensor[i] = transform(gt_list[i])
        pred_list_tensor[i] = transform(pred_list[i])
    # input shape: [n, num_classes, w, h]
    gt_list_tensor = torch.stack(gt_list_tensor, dim=0)
    pred_list_tensor = torch.stack(pred_list_tensor, dim=0)
    AP = multilabel_average_precision(pred_list_tensor, gt_list_tensor, num_classes, average, thresholds=thresholds)#None)
    return AP


def compute_f1_score(ground_truth_region_map, estimated_region_map, threshold):
    ground_truth_edge_map = ground_truth_region_map > threshold
    estimated_edge_map = estimated_region_map > threshold
    
    true_positive = np.sum(ground_truth_edge_map * estimated_edge_map)
    total_predicted_positive = np.sum(estimated_edge_map)
    total_actual_positive = np.sum(ground_truth_edge_map)
    
    precision = true_positive / total_predicted_positive if total_predicted_positive != 0 else 0
    recall = true_positive / total_actual_positive if total_actual_positive != 0 else 0
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score


def f1(pred_list, gt_list, threshold):
    total_f1_score = 0.0
    for i in range(len(pred_list)):
        ground_truth = gt_list[i]
        estimation = pred_list[i]
        total_f1_score += compute_f1_score(ground_truth, estimation, threshold)
    average_f1 = total_f1_score / float(len(pred_list))
    return average_f1 

def OIS(pred_list, gt_list, thresh_list):
    best_f1_scores = 0.0
    for i in range(len(pred_list)):
        ground_truth = gt_list[i]
        estimation = pred_list[i]
        best_f1_scores += max([compute_f1_score(ground_truth, estimation, threshold) for threshold in thresh_list])
    ois = best_f1_scores / float(len(pred_list))
    return ois

def ODS(pred_list, gt_list, thresh_list):
    # Calculate average F1 score for each threshold and find the maximum
    max_f1 = max([f1(pred_list, gt_list, threshold) for threshold in thresh_list])
    return max_f1

# val_list = ['./images/1', './images/2', './images/3']
# print('AP (Average precision) = %.5f' % AP(val_list))
# print('OIS (Optimal Image Scale) = %.5f' % OIS(val_list))
# print('ODS (Optimal Dataset Scale) = %.5f' % ODS(val_list))


# mAP (torchmetrics.detection.MeanAveragePrecision)


# CLDice https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py



def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/(np.sum(s)+np.finfo(float).eps)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens+np.finfo(float).eps)


def compute_CLDice(pred_list, gt_list, optimal_thresholds, num_classes=4):
    average_clDice_perclass = []
    for class_i in range(num_classes):
        total_clDice = 0.0
        for i in range(len(pred_list)):
            ground_truth = gt_list[i][:,:,class_i] > optimal_thresholds[class_i].item()
            estimation = pred_list[i][:,:,class_i] > optimal_thresholds[class_i].item()
            # if np.sum(ground_truth) == 0 and np.sum(estimation) == 0:
            #     score = 1.0
            # else:
            score = clDice(estimation, ground_truth)  #### ask
            total_clDice += score
        average_clDice_perclass.append(total_clDice / float(len(pred_list)))
    return average_clDice_perclass




