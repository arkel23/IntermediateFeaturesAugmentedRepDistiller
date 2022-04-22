import torch
import torch.nn as nn
import torch.nn.functional as F

from .IFACRD import SupConLoss


class IFACRDv2Loss(nn.Module):
    """IFACRD Loss function

    Args:
        opt.nce_t: the temperature
        opt.cont_no_l: no of layers for contrast
    """

    def __init__(self, opt, return_logits=False):
        super(IFACRDv2Loss, self).__init__()

        self.cont_no_l = opt.cont_no_l
        self.criterion = SupConLoss(
            temperature=opt.nce_t, base_temperature=opt.nce_t,
            contrast_mode='all', return_logits=return_logits)
        if opt.proj_ind:
            self.proj_ind = True

    def forward(self, f_1, f_2, rescaler, proj):
        """
        Args:
            f_1: the features of aug1, size [batch_size, dim]
            f_2: the features of aug2, size [batch_size, dim]
        Returns:
            The contrastive loss
        """
        f_1 = f_1[-self.cont_no_l:]
        h_1 = rescaler(f_1)
        if hasattr(self, 'proj_ind'):
            z_1 = [proj[i](feat) for i, feat in enumerate(h_1)]
        else:
            z_1 = [proj(feat) for feat in h_1]

        f_2 = f_2[-self.cont_no_l:]
        h_2 = rescaler(f_2)
        if hasattr(self, 'proj_ind'):
            z_2 = [proj[i](feat) for i, feat in enumerate(h_2)]
        else:
            z_2 = [proj(feat) for feat in h_2]

        z = torch.cat([torch.stack(z_1, dim=1),
                      torch.stack(z_2, dim=1)], dim=1)
        loss = self.criterion(F.normalize(z, dim=2))
        return loss
