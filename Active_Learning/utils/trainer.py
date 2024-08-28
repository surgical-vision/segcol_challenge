import torch
import constants
import numpy as np
from torch.utils.data import DataLoader
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.metrics import Evaluator, find_optimal_thresholds, compute_CLDice, compute_dice, check_for_zeros, AP, OIS, ODS
from utils.misc import AverageMeter, get_learning_rate
from utils.lr_scheduler import LR_Scheduler
from utils.loss import CELoss, MaskLoss
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
        super().__init__()
        # use standard CE loss without reducion as basis
        class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0])
        self.CE = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        '''
        input (B, N)
        target (B)
        '''
        minus_logpt = self.CE(input, target)
        pt = torch.exp(-minus_logpt) # don't forget the minus here
        focal_loss = (1-pt)**self.gamma * minus_logpt

        # apply class weights
        if self.alpha != None:
            focal_loss *= self.alpha.gather(0, target)
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


class Trainer:

    def __init__(self, args, model, train_set, valid_set, saver):
        self.args = args
        self.saver = saver
        self.saver.save_experiment_config()  # save cfgs

        self.num_classes = train_set.class_count

        # dataloaders
        # kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        self.dataset_size = {'train': len(train_set), 'val': len(valid_set)}
        print('dataset size:', self.dataset_size)

        # iters_per_epoch
        self.iters_per_epoch = args.iters_per_epoch if args.iters_per_epoch else len(self.train_loader)

        # optimizer & lr_scheduler
        train_params = [
            {'params': model.get_1x_lr_params(), 'lr': args.lr},  # backbone
            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10},  # aspp,decoder
        ]
        if args.with_mask and args.with_pam:  # make gamma learnable
            train_params.append({'params': model.mask_head.pam.gamma, 'lr': args.lr * 10})

        self.optimizer = torch.optim.SGD(train_params,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         nesterov=args.nesterov)
        self.lr_scheduler = LR_Scheduler(mode=args.lr_scheduler, base_lr=args.lr,
                                         lr_step=args.lr_step,
                                         num_epochs=args.epochs,
                                         warmup_epochs=args.warmup_epochs,
                                         iters_per_epoch=self.iters_per_epoch)
        num_gpu = len(args.gpu_ids.split(','))
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
            patch_replication_callback(model)
            print(args.gpu_ids)

        self.model = model.cuda()

        # loss
        # self.criterion = FocalLoss().to('cuda')  # naive ce loss
        self.criterion = nn.MSELoss().to('cuda')
        self.criterion_mask = MaskLoss(mode=self.args.mask_loss)

        # evaluator
        self.evaluator = Evaluator(self.num_classes)
        self.best_mIoU = 0.
        self.best_Acc = 0.
        self.best_dice = 0.
        self.best_ap = 0.
    
    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Generate a 2D Gaussian kernel."""
        # Create a coordinate grid of (size x size)
        x_coord = torch.arange(size)
        x_grid = x_coord.repeat(size).view(size, size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2D Gaussian kernel
        kernel = (1. / (2. * np.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )

        # Normalize the kernel so that the sum is 1
        kernel = kernel / torch.sum(kernel)

        return kernel    
    
    def apply_gaussian_filter(self, tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply a Gaussian filter to a tensor."""
        # Generate the Gaussian kernel
        kernel = self.gaussian_kernel(kernel_size, sigma)

        # Expand the dimensions of the kernel to match the input tensor's shape
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.to(tensor.device)

        # Apply the Gaussian filter using conv2d
        filtered_tensor = F.conv2d(tensor.unsqueeze(1), kernel, padding=kernel_size//2)
        return filtered_tensor
        

    def training(self, epoch, prefix='Train', evaluation=False):
        self.model.train()
        self.evaluator.reset()

        train_losses, seg_losses, mask_losses = AverageMeter(), AverageMeter(), AverageMeter()
        tbar = tqdm(self.train_loader, total=self.iters_per_epoch)

        for i, sample in enumerate(tbar):
            if i == self.iters_per_epoch:
                break

            # update lr each iteration
            self.lr_scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            image, target = Variable(sample['img'].cuda()), Variable(sample['target'].cuda())

            if not self.args.with_mask:  # ori
                output = self.model(image)
                
                loss = sigmoid_focal_loss(output, target, reduction='mean')
                smooth_edges = self.apply_gaussian_filter(target[:,1,:,:], 5, 1.0)
                edge_loss = self.criterion(output[:,0, :, :], smooth_edges)
                loss += edge_loss
                loss.backward()
                self.optimizer.step()

                train_losses.update(loss.item())
                tbar.set_description('Epoch {}, train loss: {:.4f}'.format(epoch, train_losses.avg))
            else:
                output, soft_mask = self.model(image)
                loss = sigmoid_focal_loss(output, target, reduction='mean')
                smooth_edges = self.apply_gaussian_filter(target[:,1,:,:], 5, 1.0)
                edge_loss = self.criterion(output[:,0, :, :], smooth_edges)
                loss += edge_loss
                # mask
                target_error_mask = self.generate_target_error_mask(output, target)  # B,H,W
                mask_loss = self.criterion_mask(soft_mask, target_error_mask)

                loss += mask_loss
                loss.backward()
                self.optimizer.step()

                # loss
                train_losses.update(loss.item())
                seg_losses.update(loss.item())
                mask_losses.update(mask_loss.item())

                tbar.set_description('Epoch {}, train loss: {:.3f}, seg: {:.3f}, mask: {:.3f}'.format(
                    epoch, train_losses.avg, seg_losses.avg, mask_losses.avg))


    @torch.no_grad()
    def validation(self, epoch, test=False):
        self.model.eval()
        self.evaluator.reset()

        tbar, prefix = (tqdm(self.test_loader), 'Test') if test else (tqdm(self.valid_loader), 'Valid')

        # loss
        seg_losses, mask_losses = AverageMeter(), AverageMeter()
        for i, sample in enumerate(tbar):
            image, target = sample['img'].cuda(), sample['target'].cuda()

            if not self.args.with_mask:
                output = self.model(image)
                seg_loss = sigmoid_focal_loss(output, target, reduction='mean')
                smooth_edges = self.apply_gaussian_filter(target[:,1,:,:], 5, 1.0)
                edge_loss = self.criterion(output[:,0, :, :], smooth_edges)
                seg_loss += edge_loss
                seg_losses.update(seg_loss.item())
                tbar.set_description(f'{prefix} loss: %.4f' % seg_losses.avg)
            else:
                output, soft_mask = self.model(image)
                seg_loss = sigmoid_focal_loss(output, target, reduction='mean')
                # seg_loss = self.criterion(output, target)
                # mask
                target_error_mask = self.generate_target_error_mask(output, target)  # B,H,W
                mask_loss = self.criterion_mask(soft_mask, target_error_mask)

                # loss
                seg_losses.update(seg_loss.item())
                mask_losses.update(mask_loss.item())

                tbar.set_description('{} segment loss: {:.3f}, mask loss: {:.3f}'.format(prefix, seg_losses.avg, mask_losses.avg))

            # pred = torch.argmax(output, dim=1)
            # self.evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())  # B,H,W
            
            pred_list = torch.sigmoid(output.to(torch.float64)).detach().cpu()
            gt_list = target.cpu().long()
            pred_list = pred_list.permute(0,2,3,1).numpy()
            gt_list = gt_list.permute(0,2,3,1).numpy()
            pred_list = [pred_list[i] for i in range(pred_list.shape[0])]
            gt_list = [gt_list[i] for i in range(gt_list.shape[0])]
            prediction = pred_list[0]
            
            assert type(prediction) == np.ndarray, \
                    "Wrong type of predicted map, expected np.ndarray, got {}".format(type(prediction))
                    
                # asset prediction is between 0 and 1
            assert np.max(prediction) <= 1 and np.min(prediction) >= 0, \
                "Wrong range of predicted map, expected [0, 1], got [{}, {}]".format(np.min(prediction), np.max(prediction))
            
            assert prediction.shape == (480, 640, 4), \
                "Wrong size of predicted map, expected (480, 640, 4), got {}".format(list(prediction.shape))
            
            assert prediction.dtype == np.float64, \
                "Wrong data type of predicted map, expected np.float64, got {}".format(prediction.dtype) 

            if i == 0:
                optimal_thresholds_orig = find_optimal_thresholds(pred_list, gt_list, num_classes=4)
                pred_list_handled, gt_list_handled = check_for_zeros(pred_list, gt_list, optimal_thresholds_orig, num_classes=4)
                ois = np.array(OIS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=4))
                ods_t, optimal_thresholds_ods = ODS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=4)
                optimal_thresholds = [torch.tensor(npy_array, dtype=torch.float64) for npy_array in optimal_thresholds_ods]
                ods = np.array(ods_t)
                dice_score = np.array(compute_dice(pred_list_handled, gt_list_handled, optimal_thresholds))
                ap = np.array(AP(pred_list_handled, gt_list_handled, thresholds=list(np.array(optimal_thresholds)), num_classes=4, average=None))
                clDice_score = np.array(compute_CLDice(pred_list_handled, gt_list_handled, optimal_thresholds, num_classes=4))
                break
            else:
                optimal_thresholds_orig = find_optimal_thresholds(pred_list, gt_list, num_classes=4)
                pred_list_handled, gt_list_handled = check_for_zeros(pred_list, gt_list, optimal_thresholds_orig, num_classes=4)
                ois += np.array(OIS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=4))
                ods_t, optimal_thresholds_ods = ODS(pred_list_handled, gt_list_handled, np.linspace(0.01, 0.99, 99), num_classes=4)
                ods += np.array(ods_t)
                optimal_thresholds = [torch.tensor(npy_array, dtype=torch.float64) for npy_array in optimal_thresholds_ods]
                dice_score += np.array(compute_dice(pred_list_handled, gt_list_handled, optimal_thresholds))
                ap += np.array(AP(pred_list_handled, gt_list_handled, thresholds=list(np.array(optimal_thresholds)), num_classes=4, average=None))
                clDice_score += np.array(compute_CLDice(pred_list_handled, gt_list_handled, optimal_thresholds, num_classes=4))
                
            
        ap /= (i+1)
        dice_score /= (i+1)
        ods /= (i+1)
        ois /= (i+1)
        print(ap)
        print(dice_score)
        print(ods, ois)
        print('AP: {:.3f}, Dice: {:.3f}, ODS: {:.3f}, OIS: {:.3f}'.format(sum(np.array(ap))/4, sum(dice_score)/4, sum(ods/4), sum(ois/4)))
        avg_dice = sum(dice_score)/4
        if not test and avg_dice >= self.best_dice:
            print('saving model...')
            self.best_dice = avg_dice
            self.best_ap = sum(np.array(ap))/4
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(), 
                'optimizer': self.optimizer.state_dict(),
                'Dice': avg_dice,
                'AP': sum(np.array(ap))/4,
            }
            self.saver.save_checkpoint(state)
            print('save model at epoch', epoch)

        return dice_score, ap

    def load_best_checkpoint(self, file_path=None, load_optimizer=False):
        checkpoint = self.saver.load_checkpoint(file_path=file_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if file_path:
            print('load', file_path)
        print(f'=> loaded checkpoint - epoch {checkpoint["epoch"]}')
        return checkpoint["epoch"]

    @staticmethod
    def generate_target_error_mask(output, target):
        pred = torch.argmax(output, dim=1)
        target_error_mask = (pred != target).float()  # error=1
        target_error_mask[target == constants.BG_INDEX] = 0.  # ingore bg
        return target_error_mask
