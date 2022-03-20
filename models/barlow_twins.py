import torch
from torch import nn
from .utils import off_diagonal, Projector

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # projector
        self.projector = Projector(args)

        # normalization layer for the representations z1 and z2
        self.num_features = int(args.mlp.split("-")[-1])
        self.bn = nn.BatchNorm1d(self.num_features, affine=False)

    def forward(self, y1, y2):
        B = y1.shape[0]
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus

        c.div_(B)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss