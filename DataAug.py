import numpy as np
import copy
import transforms as t


class AugmentedData:
    ROTATION_AXIS = 'z'

    def __init__(self, xyz, colors, labels):
        self.xyz = xyz
        self.colors = colors
        self.labels = labels
        self.use_augs = {'scale': True, 'rotate': True,
                         'elastic': True,'translate':True}
        self.aug_func = self.build_aug_func()

    def augment(self, xyz, colors, labels):
        aug_xyz = []
        aug_colors = []
        aug_labels = []
        for i in range(xyz.shape[0]):
            coords = copy.deepcopy(xyz[i])
            feats = copy.deepcopy(colors[i])
            label = copy.deepcopy(labels[i])

            new_coords, new_feats, new_labels = self.aug_func(coords, feats, label)
            aug_xyz.append(new_coords)
            aug_colors.append(new_feats)
            aug_labels.append(new_labels)
        aug_points = np.stack(aug_xyz, axis=0).astype(np.float32)
        aug_feats = np.stack(aug_colors, axis=0).astype(np.float32)
        aug_label = np.stack(aug_labels, axis=0).astype(np.int32)
        return aug_points, aug_feats, aug_label

    def build_aug_func(self):
        aug_funcs = []

        if self.use_augs.get('elastic', False):
            aug_funcs.append(
                t.RandomApply([
                    t.ElasticDistortion([(0.2, 0.4), (0.8, 1.6)])
                ], 0.95)
            )
        if self.use_augs.get('rotate', False):
            aug_funcs += [
                t.Random360Rotate(self.ROTATION_AXIS, around_center=True),
                t.RandomApply([
                    t.RandomRotateEachAxis(
                        [(-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (0, 0)])
                ], 0.95)
            ]
        if self.use_augs.get('scale', False):
            aug_funcs.append(
                t.RandomApply([t.RandomScale(0.9, 1.1)], 0.95)
            )
        if self.use_augs.get('translate', False):
            # Positive translation should do at the end. Otherwise, the coords might be in negative space
            aug_funcs.append(
                t.RandomApply([
                    t.RandomPositiveTranslate([0.2, 0.2, 0])
                ], 0.95)
            )
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None
