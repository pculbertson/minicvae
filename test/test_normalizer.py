"""
Unit tests for the Normalizer module.
"""

import torch
from safety_filter.learning.models import Normalizer

NUM_SAMPLES = 124


def test_normalization():
    data = torch.randn(NUM_SAMPLES, 3)
    d_mean = data.mean(dim=0)
    d_var = data.var(dim=0)
    normalizer = Normalizer(d_mean, d_var)

    d_normalized = normalizer.normalize(data)

    assert torch.allclose(d_normalized.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(d_normalized.var(dim=0), torch.ones(3), atol=1e-5)

    d_unnormalized = normalizer.unnormalize(d_normalized)
    assert torch.allclose(d_unnormalized.mean(dim=0), d_mean, atol=1e-5)
    assert torch.allclose(d_unnormalized.var(dim=0), d_var, atol=1e-5)

    mean_unnormalized, var_unnormalized = normalizer.unnormalize(
        d_normalized.mean(dim=0), d_normalized.var(dim=0)
    )

    assert torch.allclose(mean_unnormalized, d_mean, atol=1e-5)
    assert torch.allclose(var_unnormalized, d_var, atol=1e-5)
