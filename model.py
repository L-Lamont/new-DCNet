import logging

from fastai.vision.all import *
import torch

def acc_nuclei(input, target):
    mask = target > 0.1
    acc = (input[mask] > 0.1) == (target[mask] > 0.1)

    return acc.float().mean()


def _neg_loss(pred, gt, eps=1e-12):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0: 
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


class FocalLoss(nn.Module):
    """nn.Module wrapper for focal loss"""

    def __init__(self, loss=_neg_loss):
        super().__init__()
        self.neg_loss = loss


    def forward(self, out, target):

        bs, ch = target.shape[:2]
        sout = torch.sigmoid(out)
        loss1 = self.neg_loss(sout, target.data)
        loss1 = loss1.sum() / bs
        
        return loss1.to(torch.float)
    

    def activation(self, x): 
        return torch.sigmoid(x).to(torch.float)


def get_model(dls, dist):
    """Returns pretrained unet model"""

    loss_func = FocalLoss()

    learner = unet_learner(
        dls,
        resnet34,
        metrics = acc_nuclei,
        loss_func = loss_func,
        pretrained = True
    )

    if dist.rank == 0:
        logging.info('Got pretrained model')

    return learner
