from utils import config, metric
import os
from dataset.loader import TorchDataset, TorchDataLoader
from collections import Counter

def read(cfg):
    train_dataset = TorchDataset("TRAIN_SET", params=cfg.DATASET, is_training=True, )
    train_loader = TorchDataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=int(cfg.TRAIN.BATCH_SIZE/4))
                                   
    # validation_dataset = TorchDataset("VALIDATION_SET", params=cfg.DATASET,
    #                                   is_training=True, )
    # validation_loader = TorchDataLoader(dataset=validation_dataset,
    #                                     batch_size=cfg.TRAIN.BATCH_SIZE,
    #                                     num_workers=int(cfg.TRAIN.BATCH_SIZE/4))
    label_cnt = [Counter() for _ in range(5)]
    for data_ in train_loader:
        points_centered, labels, colors, label_weights = data_
        if labels.shape[0] < cfg.TRAIN.BATCH_SIZE:
            break
        for level in range(5):
            level_labels=labels[:,:,level].flatten()
            category_counts = Counter(level_labels)
            for category, count in category_counts.items():
                label_cnt[level][category]+=count
    
    for level in range(5):
        print("level{}".format(level))
        for category, count in label_cnt[level].items():
            print(f"category: {category}, nums: {count}")
                

if __name__ =="__main__":
    abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../configs"))
    config.merge_cfg_from_dir(abs_cfg_dir)
    cfg = config.CONFIG
    read(cfg)