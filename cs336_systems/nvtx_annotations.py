import math
from typing import Optional

import torch
import torch.cuda.nvtx as nvtx


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with nvtx.range("computing attention scores"):
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -torch.inf)

    with nvtx.range("computing softmax"):
        attention_weights = torch.softmax(scores, dim=-1)

    with nvtx.range("final matmul"):
        output = attention_weights @ v

    return output
