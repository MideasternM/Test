import numpy as np


def DataProcess(labels,ignore_ratio):

    num_elements = labels.size
    num_to_ignore = int(num_elements * ignore_ratio)

    mask = np.random.rand(num_elements) < ignore_ratio
    mask = mask.reshape(labels.shape)

    modified_labels = np.where(mask, -1, labels)

    return modified_labels