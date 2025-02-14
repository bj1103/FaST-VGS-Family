import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from .utils import off_diagonal, Projector

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class VICReg(nn.Module):
    def __init__(self, args, pos='coarse'):
        super().__init__()
        self.args = args
        self.pos = pos
        self.num_features = int(self.args.projector_mlp.split("-")[-1])
        self.projector_a = Projector(self.args)
        if self.args.same_projector:
            self.projector_i = self.projector_a
        else:
            self.projector_i = Projector(self.args)

    def forward(self, x, y, ids):
        B = x.shape[0]
        x = self.projector_a(x)
        y = self.projector_i(y)

        repr_loss = F.mse_loss(x, y)
        unique_id = set()
        unique_index = []
        for i, img_id in enumerate(ids):
            if img_id not in unique_id:
                unique_id.add(img_id)
                unique_index.append(i)
        
        x = x[unique_index,]
        y = y[unique_index,]

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = {
            f'{self.pos}_sim_loss' : repr_loss,
            f'{self.pos}_std_loss' : std_loss,
            f'{self.pos}_cov_loss' : cov_loss
        }
        return loss

