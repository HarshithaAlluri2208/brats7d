import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels, num_classes):
    # labels: (B, D, H, W), returns (B, C, D, H, W)
    return F.one_hot(labels.long(), num_classes).permute(0,4,1,2,3).float()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target, ignore_index=None):
        # logits: (B, C, D, H, W), target: (B, D, H, W)
        probs = F.softmax(logits, dim=1)
        C = logits.shape[1]
        target_one = one_hot(target, C)  # (B,C,D,H,W)
        dims = (0,2,3,4)
        intersection = (probs * target_one).sum(dims)
        cardinality = (probs + target_one).sum(dims)
        dice = (2. * intersection + self.eps) / (cardinality + self.eps)
        # optionally ignore background (class 0)
        if ignore_index is not None:
            mask = torch.ones(C, device=logits.device)
            mask[ignore_index] = 0.0
            dice = dice * mask
            return 1.0 - dice.sum() / mask.sum()
        return 1.0 - dice.mean()

def combined_loss(logits, target):
    ce = nn.CrossEntropyLoss()(logits, target.long())
    dloss = DiceLoss()(logits, target, ignore_index=0)
    return ce + dloss
