import torch
from torch import Tensor, nn
from .int8_fused_kernel import triton_int8_linear_per_row


def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    # Per-Tensor
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    # Per-Channel
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale

class TritonInt8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device="cpu", compute_dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.device = device
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8, device=self.device), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty((out_features, 1), dtype=torch.float32, device=self.device), requires_grad=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=self.compute_dtype, device=self.device), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_int8_linear_per_row(
            x=x,
            weight=self.weight,
            weight_scale=self.weight_scale,
            bias=self.bias,
            compute_dtype=self.compute_dtype
        )

def quantize_linear(model: nn.Module, excluded_names=None, prefix="") -> nn.Module:
    if excluded_names is None:
        excluded_names = []

    for old_layer, child in model.named_children():
        full_name = f"{prefix}.{old_layer}" if prefix else old_layer
        
        if isinstance(child, nn.Linear):
            if old_layer in excluded_names or full_name in excluded_names:
                continue
                
            has_bias = child.bias is not None
            
            new_layer = TritonInt8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=has_bias,
                device=child.weight.device,
                compute_dtype=child.weight.dtype
            )

            q_weight, scale = quantize_int8_axiswise(child.weight.data, dim=1)
            
            new_layer.weight.copy_(q_weight)
            new_layer.weight_scale.copy_(scale)
            
            if has_bias:
                new_layer.bias.copy_(child.bias.data)

            setattr(model, old_layer, new_layer)
        else:
            quantize_linear(child, excluded_names=excluded_names, prefix=full_name)
            
    return model