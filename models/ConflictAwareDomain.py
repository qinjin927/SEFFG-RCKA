

import torch
from torch import nn


class DomainDiscriminator(nn.Module):


    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(1)


def multilinear_condition(features, class_probabilities):

    if features.ndim != 2 or class_probabilities.ndim != 2:
        raise ValueError("features and class probabilities must be matrices")
    if features.shape[0] != class_probabilities.shape[0]:
        raise ValueError("features and class probabilities need equal batch size")
    joint = torch.bmm(
        class_probabilities.unsqueeze(2), features.unsqueeze(1)
    )
    return joint.flatten(start_dim=1)


def weighted_binary_cross_entropy(logits, labels, weights=None):

    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(dtype=logits.dtype), reduction="none"
    )
    if weights is None:
        return losses.mean()
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def project_conflicting_gradient(adaptation_gradient, semantic_gradient):

    adaptation = adaptation_gradient
    semantic = semantic_gradient.detach()
    dot_product = torch.sum(adaptation * semantic)
    semantic_norm = torch.sum(semantic * semantic).clamp_min(1e-12)
    projection_scale = torch.clamp(dot_product / semantic_norm, max=0.0)
    protected = adaptation - projection_scale * semantic
    cosine = dot_product / (
        adaptation.norm().clamp_min(1e-12) * semantic.norm().clamp_min(1e-12)
    )
    conflict = (dot_product < 0).to(dtype=adaptation.dtype)
    return protected, cosine.detach(), conflict.detach()
