import torch
import torch.nn as nn
import torch.nn.functional as F

from distiller.models.util import MLP, Rescaler


class IFACRDLoss(nn.Module):
    """IFACRD Loss function
    
    Args:
        opt.nce_t: the temperature
        opt.cont_no_l: no of layers for contrast
        
        opt.rs_ln: use ln for rescalers (otherwise bn)
        opt.rs_no_l: no of layers for rescaler
        opt.rs_hid_dim: the dimension of the rescaler hidden layer space
        
        opt.proj_ln: use ln for projectors (otherwise bn)
        opt.proj_no_l: no of layers for projection
        opt.proj_hid_dim: the dimension of the projector hidden layer space
        
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
    """
    def __init__(self, opt, model_t):
        super(IFACRDLoss, self).__init__()
        
        self.cont_s = opt.cont_s
        self.cont_t = opt.cont_t
        self.cont_no_l = opt.cont_no_l
    
        self.proj_s  = MLP(
            layer_norm=opt.proj_ln, no_layers=opt.proj_no_l, 
            in_features=opt.s_dim, out_features=opt.feat_dim, hidden_size=opt.proj_hid_dim)
        self.proj_t  = MLP(
            layer_norm=opt.proj_ln, no_layers=opt.proj_no_l, 
            in_features=opt.t_dim, out_features=opt.feat_dim, hidden_size=opt.proj_hid_dim)
        
        self.rescaler = Rescaler(opt, model_t)
        
        self.criterion = SupConLoss(temperature=opt.nce_t, base_temperature=opt.nce_t, contrast_mode='all')
       
    def forward(self, f_s, f_t):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the list of intermediate features of teacher network, size [batch_size, t_dim]
        Returns:
            The contrastive loss
        """
        if self.cont_s in [0, 1, 2]:
            f_s = self.proj_s(f_s)
        elif self.cont_s == 3:
            f_s_1, f_s_2 = f_s
            f_s_1 = self.proj_s(f_s_1)
            f_s_2 = self.proj_s(f_s_2)
            f_s = torch.stack([f_s_1, f_s_2], dim=1)
        elif self.cont_s == 4:
            f_s_0, f_s_1, f_s_2 = f_s 
            f_s_0 = self.proj_s(f_s_0)   
            f_s_1 = self.proj_s(f_s_1)
            f_s_2 = self.proj_s(f_s_2)
            f_s = torch.stack([f_s_0, f_s_1, f_s_2], dim=1)
        
        if self.cont_t in [0, 1, 2]:
            f_t = f_t[-self.cont_no_l:]
            f_t = self.rescaler(f_t)
            f_t = [self.proj_t(feat) for feat in f_t]
        elif self.cont_t == 3:
            f_t_1, f_t_2 = f_t
            f_t_1 = f_t_1[-self.cont_no_l:]
            f_t_1 = self.rescaler(f_t_1)
            f_t_1 = [self.proj_t(feat) for feat in f_t_1]
            f_t_2 = f_t_2[-self.cont_no_l:]
            f_t_2 = self.rescaler(f_t_2)
            f_t_2 = [self.proj_t(feat) for feat in f_t_2]
            f_t = f_t_1 + f_t_2
        elif self.cont_t == 4:
            f_t_0, f_t_1, f_t_2 = f_t
            f_t_0 = f_t_0[-self.cont_no_l:]
            f_t_0 = self.rescaler(f_t_0)
            f_t_0 = [self.proj_t(feat) for feat in f_t_0] 
            f_t_1 = f_t_1[-self.cont_no_l:]
            f_t_1 = self.rescaler(f_t_1)
            f_t_1 = [self.proj_t(feat) for feat in f_t_1]
            f_t_2 = f_t_2[-self.cont_no_l:]
            f_t_2 = self.rescaler(f_t_2)
            f_t_2 = [self.proj_t(feat) for feat in f_t_2]
            f_t = f_t_0 + f_t_1 + f_t_2
       
        if self.cont_s in [0, 1, 2]:
            z = torch.cat([f_s.unsqueeze(1), torch.stack(f_t, dim=1)], dim=1)     
        else:
            z = torch.cat([f_s, torch.stack(f_t, dim=1)], dim=1)
        loss = self.criterion(F.normalize(z, dim=2))
        return loss 



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, return_logits=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.return_logits = return_logits

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if self.return_logits:
            return exp_logits

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    


