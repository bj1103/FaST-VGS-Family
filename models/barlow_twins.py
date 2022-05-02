import torch
from torch import nn
from .utils import off_diagonal, Projector

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # projector
        self.projector_a = Projector(self.args)
        if self.args.same_projector:
            self.projector_i = self.projector_a
        else:
            self.projector_i = Projector(self.args)

        # normalization layer for the representations z1 and z2
        self.num_features = int(args.projector_mlp.split("-")[-1])
        self.bn = nn.BatchNorm1d(self.num_features, affine=False)

    def forward(self, y1, y2, ids):
        B = y1.shape[0]
        z1 = self.projector_a(y1)
        z2 = self.projector_i(y2)

        # empirical cross-correlation matrix
        unique_id = set()
        unique_index = []
        for i, img_id in enumerate(ids):
            if img_id not in unique_id:
                unique_id.add(img_id)
                unique_index.append(i)
        
        z1 = z1[unique_index,]
        z2 = z2[unique_index,]
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus

        c.div_(B)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        # loss = on_diag + self.args.lambd * off_diag
        return {'on_diag' : on_diag, 'off_diag' : off_diag}