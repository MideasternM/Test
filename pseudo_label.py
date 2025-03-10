import torch
import torch.nn.functional as F
import numpy as np


def gen_label(input, labels, threshold):

    input_softmax = F.softmax(input, dim=-1)

    max_value = torch.max(input_softmax, -1)[0]
    max_index=torch.max(input_softmax, -1)[1]

    pseudo_label_idx = torch.ge(max_value, threshold)
    targets_idx = torch.eq(labels, -1).view(pseudo_label_idx.shape)

    pseudo_label_idx = pseudo_label_idx & targets_idx
    pseudo_label = torch.where(pseudo_label_idx, max_index, torch.tensor(-1, device=pseudo_label_idx.device))

    return pseudo_label

#input(batch_size, num_points, cls)
# labels(32,2048)
# torch.max(input_softmax,-1)[0] (32,2048)
