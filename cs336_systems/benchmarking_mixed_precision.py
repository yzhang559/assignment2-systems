import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"fc1 output: {x.dtype}")

        x = self.ln(x)
        print(f"ln output: {x.dtype}")

        x = self.fc2(x)
        print(f"fc2 output: {x.dtype}")

        return x


if __name__ == '__main__':
    """
    fc1 weight dtype: torch.float32
    fc1 weight dtype within autocase: torch.float32
    fc1 output: torch.float16
    ln output: torch.float32
    fc2 output: torch.float16
    loss dtype: torch.float32
    fc1 weight grad dtype: torch.float32
    """
    model = ToyModel(10, 1).cuda()
    print(f"fc1 weight dtype: {model.fc1.weight.dtype}")

    x = torch.randn(1, 10).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        print(f"fc1 weight dtype within autocase: {model.fc1.weight.dtype}")
        out = model(x)
        loss = out.sum()

        print(f"loss dtype: {loss.dtype}")
        loss.backward()
        print(f"fc1 weight grad dtype: {model.fc1.weight.grad.dtype}")
