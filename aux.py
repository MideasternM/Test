import torch
import torch.nn as nn
from models.PointNet2.pointnet2_model import PointNet2Encoder
from models.PointNet2.pointnet2_model import PointNet2Decoder
from utils.loss import SupConLoss

class aux_branch(nn.Module):
    def __init__(self, level, num_class):
        super(aux_branch,self).__init__()
        self.beta = 0.999
        self.encoder = PointNet2Encoder()
        self.projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True))
        with torch.no_grad():
            prototypes = []
            for level_classes in num_class[2:5]:
                num_classes_in_level = level_classes
                level_prototypes = torch.rand(num_classes_in_level, 128)
                prototypes.append(level_prototypes)
            self.prior_ema = prototypes
            for i in range(level):
                self.prior_ema[i] = nn.functional.normalize(self.prior_ema[i], dim=1)
        
    def ema(self, proir,cur_status, level, class_index):
        cur_status[level][class_index] = proir.mean(0)
        with torch.no_grad():
            self.prior_ema[level][class_index] = self.beta * self.prior_ema[level][class_index] + (1 - self.beta) * cur_status[level][class_index]

    def split(self, num_class, label, points, level):
        level_info = []
        # label = torch.chunk(label, level, dim=2)
        for i in range(level):
            feat = []
            coord = []
            offset = []
            class_index = []
            category = torch.unique(label[i])
            for j in category:
                label1=label[i].squeeze(-1)
                index = (label1==j)
                index = index.nonzero(as_tuple=False)
                if index.size(0) == 0:
                    continue
                batch_idx = index[:, 0]
                point_idx = index[:, 1]
                selected = points[batch_idx, 3:, point_idx]
                selected_coord = points[batch_idx, :3, point_idx]
                point_num = selected.shape[0]
                if 0 < point_num < 256:
                    required = 256 - point_num
                    select_index = torch.randint(0, point_num, (required,))
                    sampled_points = selected[select_index]
                    sampled_coord = selected_coord[select_index]
                    selected = torch.cat([selected, sampled_points], dim=0)
                    selected_coord = torch.cat([selected_coord, sampled_coord], dim=0)
                    point_num = 256
                feat.append(selected)
                coord.append(selected_coord)
                offset.append(point_num)
                class_index.append(j)
            feat = torch.cat(feat,dim=0)
            coord = torch.cat(coord,dim=0)
            offset = torch.cumsum(torch.IntTensor(offset), dim=0, dtype=torch.int32)
            level_info.append([coord, feat, offset, class_index])
        return level_info

    def forward(self, num_class, label, points, level):
        level_info = self.split(num_class, label, points, level)
        cur_status = [p.clone() for p in self.prior_ema]
        cur_feat = [[] for i in range(level)]
        for level_index, level_points in enumerate(level_info):
            class_index = level_info[level_index][-1]
            point = level_points[:-2]
            point = torch.cat(point,dim=1)
            offset = level_points[-2]
            class_len = 0
            for c, c_index in zip(offset, class_index):
                end = c
                begin = offset[class_len-1] if class_len-1>=0 else 0
                c1 = point[begin:end]
                c1 = c1.reshape(1,6,end - begin)
                c_coord, c_feat = self.encoder(c1)
                c_feat = c_feat.reshape(16,512)
                c_feat = self.projection(c_feat)
                c_feat = nn.functional.normalize(c_feat, dim=1)
                self.ema(c_feat.detach(), cur_status, level_index, c_index-1)
                c_feat = torch.cat([c_feat, torch.ones([c_feat.size(0),1], device=c_feat.device)*255], dim=1)
                c_feat[:, -1] = c_index
                cur_feat[level_index].append(c_feat)
                class_len += 1
            cur_feat[level_index] = torch.cat(cur_feat[level_index], dim=0)
        return cur_feat, self.prior_ema

if __name__ == "__main__":
    num_class = [3, 5, 8, 5, 8]

    label_first = torch.randint(1, 4, (16, 2048))

    label_second = torch.randint(4, 9, (16, 2048))

    label_third = torch.randint(9, 17, (16, 2048))

    label = torch.stack((label_first, label_second, label_third), dim=-1)

    points=torch.rand(16,6,2048)
    print(torch.version.cuda)

    aux = aux_branch(3,num_class)
    current, whole = aux(num_class, label, points, 3)
    LOSS = SupConLoss()
    loss = []
    for i in range(3):
        loss_i = LOSS(current[i][:,:-1],current[i][:,-1])
        loss.append(loss_i)
    print(loss)