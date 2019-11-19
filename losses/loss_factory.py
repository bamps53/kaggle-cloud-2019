from catalyst.dl.utils import criterion
from torch.nn.modules.loss import _Loss
from functools import partial
from catalyst.contrib.criterion import FocalLossBinary
from . import functions
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.insert(0, '../..')


class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


class WeightedBCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice * 0.75 + bce * 1.25


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=0.75, neg_weight=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logit, truth):
        batch_size, num_class, H, W = logit.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy_with_logits(
            logit, truth, reduction='none')

        if weight is None:
            loss = loss.mean()

        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.pos_weight * pos * loss / pos_sum +
                    self.neg_weight * neg * loss / neg_sum).sum()
            # raise NotImplementedError

        return loss


class FocalDiceLoss(DiceLoss):
    __name__ = 'focal_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.focal = FocalLossBinary(gamma=2.0, alpha=0.25)

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        focal = self.focal(y_pr, y_gt)
        return dice + focal


def get_loss(config):
    if config.loss.name == 'BCEDice':
        criterion = BCEDiceLoss(eps=1.)
    elif config.loss == 'WeightedBCE':
        criterion = WeightedBCELoss()
    elif config.loss == 'WeightedBCEDice':
        criterion = WeightedBCEDiceLoss()
    elif config.loss.name == 'Focal':
        criterion = FocalLossBinary(gamma=config.loss.params.focal_gamma)
    elif config.loss.name == 'FocalDice':
        criterion = FocalDiceLoss(eps=1.)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    return criterion
