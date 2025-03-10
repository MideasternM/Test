from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

def cal_cross_entropy(pred, target, weight, crossEntropy):
    loss = 0
    for i, predi in enumerate(pred):
        for j, predj in enumerate(predi):
            loss += crossEntropy(predj.view(1, -1), target[i][j]) * weight[i][j]
    return loss * 0.01

class HeirarchicalCrossEntropyLoss(object):
    def __init__(self, weights, device):
        super(HeirarchicalCrossEntropyLoss, self).__init__()
        self.weights = [torch.tensor(w, dtype=torch.float, device=device) for w in weights]
        """Loss functions for semantic segmentation"""
        self.CrossEntropyLosses = [nn.CrossEntropyLoss(weight=w,ignore_index=-1) for w in self.weights]
    def __call__(self, pred, target, level=0):
        pred = pred.view(-1, len(self.weights[level]))
        target=target.view(-1)
        return self.CrossEntropyLosses[level](pred, target)

class ConsistencyLoss:
    def __init__(self, CM, CLW, device="cpu"):
        super(ConsistencyLoss, self).__init__()
        self.gather_id = [np.argmax(m, axis=0)for m in CM]
        self.weights = torch.Tensor(CLW).to(device)
    def __call__(self, preds):
        probs = [nn.functional.softmax(pred.permute(0, 2, 1), dim=2) for pred in preds]
        loss = 0
        for i, gid in enumerate(self.gather_id):
            probs_ = probs[i + 1] - probs[i][..., gid]
            loss += nn.functional.relu(probs_).sum() * self.weights[i]
        return loss * 0.01
            
def cal_consistency_loss(CM, preds, CLW):
    probs = [nn.functional.softmax(pred.permute(0, 2, 1), dim=2) for pred in preds]
    for i, matrix in enumerate(CM):
        m = torch.Tensor(matrix).cuda()
        prob_ = torch.matmul(m, probs[i + 1].permute(0, 2, 1)).permute(0, 2, 1)
        CLoss = CLW[i] * torch.nn.functional.relu(prob_- probs[i]).sum()
    return CLoss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf """
    def __init__(self, temperature=0.07,
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, mask=None):
        """Compute loss for model. 
        Args:
            features: hidden vector of size [npoints, ...].
            labels: ground truth of shape [npoints].
            mask: contrastive mask of shape [npoints, npoints], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)
        
        contrast_feature = features
        anchor_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.size(0)).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask
        positive_counts = mask.sum(1) - 1

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+ 1e-9)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
