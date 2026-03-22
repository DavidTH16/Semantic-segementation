import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def dice_coefficient(predictions, targets):
    # dummt function to get dice coefficient

    probabilities = F.softmax(predictions, dim=1)
    ground_truth = F.one_hot(targets, predictions.size(1)).permute(0, 3, 1, 2).float()
    interseccion = torch.sum(probabilities * ground_truth, dim=(2, 3))
    union = torch.sum(probabilities + ground_truth, dim=(2, 3))
    dice_coef = (2. * interseccion + 1e-8) / (union + 1e-8)
    mean_dice_coef = dice_coef.mean(dim=1)
    return mean_dice_coef

class CustomLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.3):
        super(CustomLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # if you don't want to use my own implementation
        # feel free to use Diceloss from smp
        self.dice_loss_smp = smp.losses.DiceLoss(mode='multiclass')
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)

    def forward(self, predictions, targets):
        targets_long = targets.long() 
        probabilities = F.log_softmax(predictions, dim=1).exp()
        
        batch_size = targets_long.size(0)
        num_classes = predictions.size(1)
        
        target_flat = targets_long.view(batch_size, -1)
        prob_flat = probabilities.view(batch_size, num_classes, -1)

        ground_truth = F.one_hot(target_flat, num_classes).permute(0, 2, 1).float()

        intersection = torch.sum(prob_flat * ground_truth, dim=2)
        union = torch.sum(prob_flat + ground_truth, dim=2)
        dice_coef = (2. * intersection + 1e-8) / (union + 1e-8)

        mask = ground_truth.sum(dim=2) > 0 
        loss_per_class = 1.0 - dice_coef
        loss_masked = loss_per_class * mask.float()
        
        total_dice_loss = loss_masked.sum() / mask.sum().clamp_min(1e-7)
        mean_dice_coef = 1.0 - total_dice_loss

        loss_focal = self.focal_loss(predictions, targets_long)
        final_loss = 0.7 * total_dice_loss + 0.3 * loss_focal
        return mean_dice_coef, final_loss
