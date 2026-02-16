from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(
        self,
        x: torch.Tensor,
        n_q: tp.Optional[int] = None,
        layers: tp.Optional[list] = None,
    ) -> QuantizedResult:
        n_q = n_q if n_q else self.n_q
        if layers and max(layers) >= n_q:
            raise ValueError(
                f"Last layer index in layers: A {max(layers)}. Number of quantizers in RVQ: B {self.n_q}. A must less than B."
            )
        quantized, codes, commit_loss, quantized_list = self.vq(x, n_q=n_q, layers=layers)
        return quantized, codes, torch.mean(commit_loss), quantized_list

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None) -> torch.Tensor:
        n_q = n_q if n_q else self.n_q
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        quantized = self.vq.decode(codes, st=st)
        return quantized
