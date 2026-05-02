import torch
import torch.distributed as dist
from torch import nn


class DDPOverlapIndividualParameters(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(param):
            handle = dist.all_reduce(param.grad, async_op=True, op=dist.ReduceOp.SUM)
            self.handles.append(handle)

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad /= dist.get_world_size()

        self.handles.clear()
