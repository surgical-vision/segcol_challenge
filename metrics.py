# metrics: Dice, AP, OIS, ODS, CLDice
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


def check_for_zeros(pred_list, gt_list, thresholds, num_classes = 4):
    gt_list_handled = gt_list.copy()
    pred_list_handled = pred_list.copy()
    # change thresholds to np array
    thresholdsnp = np.array(thresholds)
    for i in range(len(pred_list)):
        for j in range(num_classes):
            if np.sum(gt_list[i][:,:,j]) == 0 and np.sum(pred_list[i][:,:,j] > thresholdsnp[j]) == 0:
                gt_list_handled[i][:,:,j] = np.ones_like(gt_list[i][:,:,j]).astype(int)
                pred_list_handled[i][:,:,j] = np.ones_like(pred_list[i][:,:,j])
            else:
                gt_list_handled[i][:,:,j] = gt_list[i][:,:,j]
                pred_list_handled[i][:,:,j] = pred_list[i][:,:,j]
            
    return pred_list_handled, gt_list_handled

# Function to find the optimal thresholds using Youden's Index
# compute dice score for each class
def find_optimal_thresholds(pred_list, gt_list, num_classes):
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
    metric = MultilabelROC(num_labels=num_classes, thresholds=None)
    fpr, tpr, thresholds = metric(pred_list_tensor, gt_list_tensor)
    optimal_thresholds = []
    dice_list = []
    for i in range(len(tpr)):
        J = tpr[i] - fpr[i]  # Youden's index
        idx = np.argmax(J)
        optimal_thresholds.append(thresholds[i][idx])
    return optimal_thresholds

def compute_dice(pred_list, gt_list, thresholds):
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

    dice_list = []
    for i in range(gt_list_tensor.shape[1]):
        dice_list.append(dice(pred_list_tensor[:, i, :, :], gt_list_tensor[:, i, :, :], threshold=thresholds[i]))
    return dice_list

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

def OIS(pred_list, gt_list, thresh_list, num_classes):
    ois_list = []
    for class_i in range(num_classes):
        best_f1_scores = 0.0
        for i in range(len(pred_list)):
            ground_truth = gt_list[i][:,:,class_i]
            estimation = pred_list[i][:,:,class_i]
            best_f1_scores += max([compute_f1_score(ground_truth, estimation, threshold) for threshold in thresh_list])
        ois_list.append(best_f1_scores / float(len(pred_list)))

    return ois_list

def ODS(pred_list, gt_list, thresh_list, num_classes):
    max_f1_list = []
    best_threshold = []
    for class_i in range(num_classes):
        # Calculate average F1 score for each threshold and find the maximum
        pred_list_class = [pred[:,:,class_i] for pred in pred_list]
        gt_list_class = [gt[:,:,class_i] for gt in gt_list]
        computed_list = [f1(pred_list_class, gt_list_class, threshold) for threshold in thresh_list]
        max_f1_list.append(max(computed_list))
        best_threshold.append(thresh_list[np.where(computed_list == max(computed_list))[0]])
    return max_f1_list, best_threshold


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
            if np.sum(ground_truth) == 0 and np.sum(estimation) == 0:
                score = 1.0
            else:
                score = clDice(estimation, ground_truth)  
            total_clDice += score
        average_clDice_perclass.append(total_clDice / float(len(pred_list)))
    return average_clDice_perclass




