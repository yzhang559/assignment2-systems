import torch
import torch.distributed as dist
from torch import nn


class DPP(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, async_op=False, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
