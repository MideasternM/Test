import torch
import torch.nn as nn
from ptv2 import GVAPatchEmbed, Encoder, Decoder, PointBatchNorm 

class PTSegV2_Balance_Prior(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes=13,
                 patch_embed_depth=2,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=16,
                 enc_depths=(2, 6, 2),
                 enc_channels=(96, 192, 384),
                 enc_groups=(12, 24, 48),
                 enc_neighbours=(16, 16, 16),
                 grid_sizes=(0.1, 0.2, 0.4),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 enable_checkpoint=False,
                 beta=0.999,
                 **kwargs):
        super(PTSegV2_Balance_Prior, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.beta = beta
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            self.enc_stages.append(enc)
        self.projection = nn.Sequential(nn.Linear(enc_channels[-1], enc_channels[-2]), nn.BatchNorm1d(enc_channels[-2]), nn.ReLU(inplace=True),
                                        nn.Linear(enc_channels[-2], 48), nn.BatchNorm1d(48), nn.ReLU(inplace=True))
        # ema
        self.register_buffer('prior_ema', torch.rand(num_classes, 48))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    @torch.no_grad()
    def _ema(self, prior):
        """prior: n*(dim+1), feature dim + label"""
        cur_status = self.prior_ema.clone()
        for label in range(self.num_classes):
            mask_c = prior[:, -1] == label
            if mask_c.nonzero().numel() > 0:
                cur_status[label, :] = prior[mask_c, :-1].mean(0)
        self.prior_ema = self.beta * self.prior_ema + (1 - self.beta) * cur_status

    def forward(self, p0, x0=None, o0=None, is_train=False, mask=None, ignore_index=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        
        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage
        coord, feat, offset = skips[-1][0][0], skips[-1][0][1], skips[-1][0][2]  # unpooling feature info in the last enc stage
        feat = self.projection(feat)

        feat = nn.functional.normalize(feat, dim=1)
        current_prior = torch.cat([feat, torch.ones([feat.size(0),1], device=feat.device)*255], dim=1)
        class_index_len = 0
        for i in range(o0.size(0)):
            for j, c_index in enumerate(class_index_batch[i]):
                begin = offset[class_index_len-1] if class_index_len-1 >= 0 else 0
                end = offset[class_index_len]
                current_prior[begin:end, -1] = c_index
                class_index_len += 1
        self._ema(current_prior.detach())
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return current_prior, self.prior_ema
    
class PTSegV2_Balance_Main(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes=13,
                 patch_embed_depth=2,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=16,
                 enc_depths=(2, 6, 2),
                 enc_channels=(96, 192, 384),
                 enc_groups=(12, 24, 48),
                 enc_neighbours=(16, 16, 16),
                 dec_depths=(1, 1, 1),
                 dec_channels=(48, 96, 192),
                 dec_groups=(6, 12, 24),
                 dec_neighbours=(16, 16, 16),
                 grid_sizes=(0.1, 0.2, 0.4),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 enable_checkpoint=False,
                 unpool_backend="interp",
                 beta=0.999,
                 **kwargs):
        super(PTSegV2_Balance_Main, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.beta = beta
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

        # ema
        self.register_buffer('prior_ema', torch.rand(num_classes, dec_channels[0]))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)

    @torch.no_grad()
    def _ema(self, prior):
        """prior: n*(dim+1), feature dim + label"""
        cur_status = self.prior_ema.clone()
        for label in range(self.num_classes):
            mask_c = prior[:, -1] == label
            if mask_c.nonzero().numel() > 0:
                cur_status[label, :] = prior[mask_c, :-1].mean(0)
        self.prior_ema = self.beta * self.prior_ema + (1 - self.beta) * cur_status

    def forward(self, p0, x0=None, o0=None, is_train=False, mask=None, ignore_index=None):
        # p, x, o: points, features, batches 
        # The dataloader input here is different from PointTransformer source code; 
        # it's need to modify input data{'pos':, 'x':, 'label':, } to the following form.
        if x0 is None:  # this means p0 is a dict.
            p0, x0, o0, labels = p0['pos'], p0.get('x', None), p0.get('offset', None), p0.get('y', None)
            if x0 is None: x0 = p0
            if o0 == None:
                o0, count = [], 0
                for _ in range(p0.size()[0]):
                    count += p0.size()[1]
                    o0.append(count)
                o0 = torch.IntTensor(o0).cuda(device=p0.device)
            if len(x0.size())>2: # means x0:(b, c ,n), need to be catted to (b*n, c)
                x0 = x0.transpose(1,2).contiguous() # po(b, n, 3), x0(b, n, c=3)
                p0 = torch.cat([p0_split.squeeze() for p0_split in p0.split(1,0)])
                x0 = torch.cat([x0_split.squeeze() for x0_split in x0.split(1,0)]) 
                if is_train:
                    labels = torch.cat([labels.squeeze() for labels in labels.split(1,0)]) 
        if x0.size(1)<6:
            x0 = torch.cat((p0,x0),1) # x0(n, c=in_channels+3)
        coord, feat, offset = p0, x0, o0

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        seg_logits_src = self.seg_head(feat)
        if not is_train: return seg_logits_src

        feas_norm = nn.functional.normalize(feat, dim=1)
        if mask is not None:
            labels = labels[mask]
            if ignore_index == 0:
                labels = labels - 1
            seg_logits, feas_norm = seg_logits_src[mask, :], feas_norm[mask, :]
        else:
            seg_logits, feas_norm = seg_logits_src, feas_norm
        logits_softmax = nn.functional.softmax(seg_logits, dim=1)
        preds = logits_softmax.argmax(dim=1)
        mask_true = (preds==labels)
        # # method1: current scene prototype
        # feas_true_mean = []
        # for c in range(self.num_classes):
        #     mask_true_c = mask_true & (labels==c)
        #     if mask_true_c.sum() > 0:
        #         feas_true_mean.append(torch.cat([feas_norm[mask_true_c, :].mean(0), \
        #             torch.tensor([c], device=feas_norm.device)]).unsqueeze(0))
        # feas_true_mean = torch.cat(feas_true_mean, dim=0)
        # method2: all scenes prototype
        self._ema(torch.cat([feas_norm.detach()[mask_true, :], labels[mask_true].unsqueeze(1)], dim=1))
        self.prior_ema = nn.functional.normalize(self.prior_ema, dim=1)
        return seg_logits_src, feas_norm, self.prior_ema