# metrics: Dice, AP, OIS, ODS, CLDice
# for ranking: mAP

from torchmetrics.functional.classification import binary_average_precision, dice
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
    gt_list_tensor = [transform(gt) for gt in gt_list]
    pred_list_tensor = [transform(pred) for pred in pred_list]

    # input shape: [n, num_classes, w, h]
    gt_list_tensor = torch.stack(gt_list_tensor, dim=0)
    pred_list_tensor = torch.stack(pred_list_tensor, dim=0)
    AP = []
    for i in range(num_classes):
        AP.append(binary_average_precision(pred_list_tensor[:,i,:,:], gt_list_tensor[:,i,:,:], thresholds=[thresholds[i]])) #None)
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
        computed_list = np.array([f1(pred_list_class, gt_list_class, threshold) for threshold in thresh_list])
        max_f1_list.append(np.max(computed_list))
        best_threshold.append(thresh_list[np.where(computed_list == np.max(computed_list))][0])
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

# https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier

def calculate_miou(confusion_matrix):
    MIoU = np.divide(np.diag(confusion_matrix), (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix)))
    MIoU = np.nanmean(MIoU)
    return MIoU


class Evaluator(object):

    def __init__(self, num_class):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # TP / ALL pixels
        return Acc

    def Acc_of_each_class(self):
        Accs = np.divide(np.diag(self.confusion_matrix),  # TP of each class
                         self.confusion_matrix.sum(axis=1))  # (TP+FP) of each class 列
        return Accs  # vector

    def Mean_Pixel_Accuracy(self):
        Acc = np.nanmean(self.Acc_of_each_class())  # mean of vector
        return Acc

    def IOU_of_each_class(self):
        inter = np.diag(self.confusion_matrix)  # TP
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - inter  # TP+FP+FN
        IoUs = np.divide(inter, union)  # IoU of each class
        return IoUs  # vector

    def Mean_Intersection_over_Union(self):
        MIoU = np.nanmean(self.IOU_of_each_class())  # mIoU
        return MIoU

    def Mean_Intersection_over_Union_20(self):  # 20类之后
        MIoU = 0
        if self.num_class > 20:
            subset_20 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 23, 27, 32, 33, 35, 38])
            confusion_matrix = self.confusion_matrix[subset_20[:, None], subset_20]  # 取出子矩阵
            MIoU = np.divide(np.diag(confusion_matrix), (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix)))
            MIoU = np.nanmean(MIoU)
        return MIoU

    def Mean_Dice(self):
        inter = np.diag(self.confusion_matrix)  # vector
        dices = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        dice = np.divide(2 * inter, dices)
        dice = np.nanmean(dice)
        return dice

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.divide(np.diag(self.confusion_matrix), (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix)))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]  # i:gt, j:pre
        count = np.bincount(label, minlength=self.num_class ** 2)  # total classes on confusion_matrix
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, return_miou=False):
        assert gt_image.shape == pre_image.shape  # np img, B,H,W
        confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += confusion_matrix
        if return_miou:
            return calculate_miou(confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def dump_matrix(self, path):
        np.save(path, self.confusion_matrix)