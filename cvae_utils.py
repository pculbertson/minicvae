import torch
import numpy as np
from safety_filter.minicvae.models import CVAE


def elbo_loss(model: CVAE, data: torch.tensor, cond: torch.tensor) -> torch.tensor:
    """
    Computes the evidence lower bound (ELBO) loss.

    Args:
        model: CVAE; the CVAE model.
        data: torch.tensor with shape (batch_size, output_dim); the output data.
        cond: torch.tensor with shape (batch_size, cond_dim); the conditioning data.

    Returns:
        torch.tensor; the ELBO loss.
    """
    # Normalize the data.
    data = model.output_normalizer.normalize(data)
    cond = model.cond_normalizer.normalize(cond)

    # Encode the data.
    z_mean, z_var = model.encode(data, cond)
    assert z_mean.shape == z_var.shape
    assert z_mean.shape[-1] == model.latent_dim

    # Sample from the latent space.
    latent_dist = torch.distributions.MultivariateNormal(
        z_mean, torch.diag_embed(z_var)
    )
    z = latent_dist.rsample()

    # Decode the latent code.
    output_mean, output_var = model.decode(z, cond, unnormalize=False)
    assert output_mean.shape == output_var.shape
    assert output_mean.shape[-1] == model.output_dim

    # Compute the reconstruction loss.
    recon_loss = torch.distributions.MultivariateNormal(
        output_mean, torch.diag_embed(output_var)
    ).log_prob(data)

    # Compute the KL divergence.
    prior_mean, prior_var = model.prior(cond)
    kl_div = torch.distributions.kl.kl_divergence(
        latent_dist,
        torch.distributions.MultivariateNormal(prior_mean, torch.diag_embed(prior_var)),
    )

    # Compute the ELBO.
    elbo = recon_loss - kl_div

    return -elbo.mean(dim=0)


def planar_rotation(angle: torch.tensor) -> torch.tensor:
    """
    Returns the planar rotation matrix for the given angle.

    Args:
        angle: torch.tensor with shape (batch_size,); the angle in radians.

    Returns:
        torch.tensor with shape (batch_size, 3, 3); the planar rotation matrix, where
        the third dimension is 1 (to pass through yaw angle).
    """
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)

    return torch.stack(
        [
            torch.stack([cos_angle, -sin_angle, zeros], dim=-1),
            torch.stack([sin_angle, cos_angle, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def sample_outputs(model: CVAE, cond: torch.tensor, num_samples: int = 1000):
    with torch.no_grad():
        cond = model.cond_normalizer.normalize(cond)

        # Sample the prior.
        prior_mean, prior_var = model.prior(cond)
        prior_dist = torch.distributions.MultivariateNormal(
            loc=prior_mean, covariance_matrix=torch.diag_embed(prior_var)
        )

        z = prior_dist.sample((num_samples,))

        cond_expanded = cond.unsqueeze(0).expand(num_samples, -1, -1)

        # Decode the samples
        pred_mean_expanded, pred_var_expanded = model.decode(
            z, cond_expanded, unnormalize=True
        )

        pred_mean = pred_mean_expanded.mean(dim=0)
        pred_var_expanded = torch.diag_embed(pred_var_expanded)

        # import time
        # start = time.time()
        # pred_var = torch.mean(
        #     torch.stack(
        #         [
        #             pred_var_expanded[ii]
        #             + pred_mean_expanded[ii].unsqueeze(-1)
        #             @ pred_mean_expanded[ii].unsqueeze(-2)
        #             for ii in range(num_samples)
        #         ],
        #         dim=0,
        #     ),
        #     dim=0,
        # )
        # end = time.time()

        # print(f"Stack: {end - start}")
        pred_var = torch.mean(
            pred_var_expanded
            + pred_mean_expanded.unsqueeze(-1) * pred_mean_expanded.unsqueeze(-2),
            dim=0,
        )

        pred_var = pred_var - pred_mean.unsqueeze(-1) @ pred_mean.unsqueeze(-2)

        return pred_mean, pred_var


def wrap_angle(th: torch.tensor):
    """
    Wraps the angle to [-pi, pi].

    Args:
        th: torch.tensor with shape (batch_size,); the angle in radians.

    Returns:
        torch.tensor with shape (batch_size,); the wrapped angle.
    """
    return torch.remainder(th + np.pi, 2 * np.pi) - np.pi
