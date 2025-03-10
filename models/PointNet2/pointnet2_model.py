import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation
from torch_scatter import segment_csr

class PointNet2(nn.Module):
    def __init__(self, cfg=None, args=None):
        super(PointNet2, self).__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.DATA.LABEL_NUMBER
        self.level = args.mc_level
        self.beta = 0.999
        # self.aux = aux_branch(3, self.num_class)
        # with torch.no_grad():
        #     prototypes = []
        #     for level_classes in self.num_class[2:]:
        #         level_prototypes = torch.rand(level_classes, 128, device='cuda:0')
        #         prototypes.append(level_prototypes)
        #     self.prior_ema = prototypes
        #     for i in range(3):
        #         self.prior_ema[i] = nn.functional.normalize(self.prior_ema[i], dim=1)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        if self.training:
            self.decoder = PointNet2DecoderWithoutClsHead()

        if self.level == 0:
            self.decoder0 = PointNet2Decoder(self.num_class[0], self.cfg)
        elif self.level == 1:
            self.decoder1 = PointNet2Decoder(self.num_class[1], self.cfg)
        elif self.level == 2:
            self.decoder2 = PointNet2Decoder(self.num_class[2], self.cfg)
        elif self.level == 3:
            self.decoder3 = PointNet2Decoder(self.num_class[3], self.cfg)
        elif self.level == 4:
            self.decoder4 = PointNet2Decoder(self.num_class[4], self.cfg)
        elif self.level == -1:
            self.decoder0 = PointNet2Decoder(self.num_class[0], self.cfg)
            self.decoder1 = PointNet2Decoder(self.num_class[1], self.cfg)
            self.decoder2 = PointNet2Decoder(self.num_class[2], self.cfg)
            self.decoder3 = PointNet2Decoder(self.num_class[3], self.cfg)
            self.decoder4 = PointNet2Decoder(self.num_class[4], self.cfg)
    

    def ema_update(self, correct_feat, correct_labels, cur_status, level):
        num_classes = self.num_class[level + 2]
        if num_classes == 0 or correct_feat.size(0) == 0:
            return

        one_hot = F.one_hot(correct_labels, num_classes).float()
        sum_feats = torch.mm(one_hot.T, correct_feat)
        counts = one_hot.sum(dim=0)

        valid_counts = counts.float().unsqueeze(1) + 1e-6
        mean_feats = sum_feats / valid_counts
        valid_mask = counts > 0

        cur_status[level][valid_mask] = mean_feats[valid_mask]

        with torch.no_grad():
            ema_values = (self.beta * self.prior_ema[level] + 
                        (1 - self.beta) * cur_status[level])
            self.prior_ema[level].copy_(ema_values)

        self.prior_ema[level] = F.normalize(self.prior_ema[level], dim=1)

    def forward(self, xyz, labels=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        input_list = [l0_xyz, l0_points,
                      l1_xyz, l1_points,
                      l2_xyz, l2_points,
                      l3_xyz, l3_points,
                      l4_xyz, l4_points]
        
        if self.level == 0:
            x0 = self.decoder0(*input_list)
            return x0
        elif self.level == 1:
            x1 = self.decoder1(*input_list)
            return x1
        elif self.level == 2:
            x2 = self.decoder2(*input_list)
            return x2
        elif self.level == 3:
            x3 = self.decoder3(*input_list)
            return x3
        elif self.level == 4:
            x4 = self.decoder4(*input_list)
            return x4
        elif self.level == -1:
            x0 = self.decoder0(*input_list)
            x1 = self.decoder1(*input_list)
            x2 = self.decoder2(*input_list)
            x3 = self.decoder3(*input_list)
            x4 = self.decoder4(*input_list)
            logits = [x0, x1, x2, x3, x4]
            # if self.training:
            #     prior_feat, prior_prototype = self.aux(self.num_class, labels[2:5], xyz, 3)
            #     feat = self.decoder(*input_list)
            #     feat = feat.permute(0, 2, 1)
            #     feat = F.normalize(feat, dim=1)

            #     cur_status = [p.clone() for p in self.prior_ema]
            #     labels = [label.squeeze(dim=2) for label in labels]

            #     for i in range(2,5):
            #         level = i - 2
            #         pred_class = logits[i].argmax(dim=1)
            #         correct_mask = (pred_class == labels[i])
            #         if correct_mask.any():
            #             correct_feat = feat[correct_mask]
            #             correct_labels = labels[i][correct_mask]
            #             self.ema_update(correct_feat, correct_labels, cur_status, level)

            #     return logits, feat, self.prior_ema, prior_feat, prior_prototype
            
            return logits
            
class PointNet2DecoderWithoutClsHead(nn.Module):
    def __init__(self):
        super(PointNet2DecoderWithoutClsHead, self).__init__()
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

    def forward(self,   l0_xyz, l0_points,
                        l1_xyz, l1_points,
                        l2_xyz, l2_points,
                        l3_xyz, l3_points,
                        l4_xyz, l4_points):
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l0_points

class PointNet2Decoder(nn.Module):
    def __init__(self, num_class, cfg):
        super(PointNet2Decoder, self).__init__()
        self.cfg = cfg
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        self.conv_out = nn.Conv1d(128, num_class, 1)

    def forward(self,   l0_xyz, l0_points,
                        l1_xyz, l1_points,
                        l2_xyz, l2_points,
                        l3_xyz, l3_points,
                        l4_xyz, l4_points):
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv_out(x)
        return x  
    
class PointNet2Encoder(nn.Module):
    def __init__(self):
        super(PointNet2Encoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
    
    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        return l4_xyz, l4_points
    
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
                level_prototypes = torch.rand(num_classes_in_level, 128, device="cuda:0")
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
        for i in range(level):
            current_label = label[i].squeeze(-1)  # (B, N)
            B, N = current_label.shape
            
            # Flatten labels and get sorted indices
            label_flat = current_label.view(-1)
            categories, inverse_indices, counts = torch.unique(
                label_flat, return_inverse=True, return_counts=True)
            
            # Sort points by category
            sorted_indices = torch.argsort(inverse_indices)
            points_flat = points.permute(0, 2, 1).reshape(-1, points.size(1))[sorted_indices]
            
            # Split points into categories using cumulative counts
            cum_counts = torch.cat([torch.tensor([0], device=points.device), counts.cumsum(0)])
            
            feat_list, coord_list, offset_list, class_index_list = [], [], [], []
            for j in range(len(categories)):
                start, end = cum_counts[j:j+2]
                if (end - start) == 0:
                    continue
                
                # Process category points
                category_points = points_flat[start:end]
                m = category_points.size(0)
                
                # Sample points to 256 if needed
                if 0 < m < 256:
                    indices = torch.randint(0, m, (256-m,), device=points.device)
                    category_points = torch.cat([category_points, category_points[indices]])
                
                # Split features and coordinates
                coord = category_points[:, :3]
                feat = category_points[:, 3:]
                
                feat_list.append(feat)
                coord_list.append(coord)
                offset_list.append(category_points.size(0))
                class_index_list.append(categories[j].item())
            
            # Concatenate results for this level
            feat = torch.cat(feat_list)
            coord = torch.cat(coord_list)
            offset = torch.cumsum(torch.tensor(offset_list, dtype=torch.int32, device=points.device), 0)
            level_info.append([coord, feat, offset, class_index_list])
        
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
    

if __name__=='__main__':
    device = torch.device("cuda")
    enc = PointNet2Encoder().to(device)
    rand_input = torch.rand(1,6,10000).to(device)
    coord, feat = enc(rand_input)
    feat = feat.to(device)
    feat = feat.reshape(16,512)
    projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 48),
                nn.BatchNorm1d(48),
                nn.ReLU(inplace=True)).to(device)
    final = projection(feat)
    print(final)
        
        
