import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


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

    def finish_gradient_synchronization_flatten(self):
        grad_list = []
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                grad_list.append(param.grad)

        flatten_grad = _flatten_dense_tensors(grad_list)
        dist.all_reduce(flatten_grad, async_op=False, op=dist.ReduceOp.SUM)

        flatten_grad /= dist.get_world_size()
        for grad, new_grad in zip(grad_list, _unflatten_dense_tensors(flatten_grad, grad_list)):
            grad.copy_(new_grad)
