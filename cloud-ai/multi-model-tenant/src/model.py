"""Multi-model tenant pool."""

import torch
import torch.nn as nn


class TenantModelPool(nn.Module):
    def __init__(self, tenants: int = 4, hidden: int = 128):
        super().__init__()
        self.models = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(tenants)]
        )

    def forward(self, tenant_id: int, x: torch.Tensor) -> torch.Tensor:
        return self.models[tenant_id](x)
