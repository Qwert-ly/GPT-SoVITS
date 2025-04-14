import torch
from torch.nn import functional as F
import numpy as np
from functools import lru_cache
from typing import Tuple, Optional, Literal

# Constants
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


# Use torch.jit.script to compile functions for performance
@torch.jit.script
def _compute_widths(
        unnormalized_widths: torch.Tensor,
        num_bins: int,
        min_bin_width: float,
        left: float,
        right: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute bin widths and cumulative widths from unnormalized inputs."""
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    return widths, cumwidths


@torch.jit.script
def _compute_heights(
        unnormalized_heights: torch.Tensor,
        num_bins: int,
        min_bin_height: float,
        bottom: float,
        top: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute bin heights and cumulative heights from unnormalized inputs."""
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    return heights, cumheights


@torch.jit.script
def _compute_derivatives(
        unnormalized_derivatives: torch.Tensor,
        min_derivative: float
) -> torch.Tensor:
    """Compute derivatives from unnormalized inputs."""
    return min_derivative + F.softplus(unnormalized_derivatives)


@torch.jit.script
def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Find indices where inputs would be sorted into bin_locations."""
    # Create a copy to avoid modifying the input
    bin_locations_modified = bin_locations.clone()
    bin_locations_modified[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations_modified, dim=-1) - 1


@torch.jit.script
def _inverse_transform_formula(
        inputs: torch.Tensor,
        input_cumheights: torch.Tensor,
        input_derivatives: torch.Tensor,
        input_derivatives_plus_one: torch.Tensor,
        input_delta: torch.Tensor,
        input_heights: torch.Tensor,
        input_bin_widths: torch.Tensor,
        input_cumwidths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the inverse transform and its log absolute determinant."""
    a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) + input_heights * (input_delta - input_derivatives)

    b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
    )

    c = -input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    # Use clamp to ensure numerical stability
    discriminant = torch.clamp(discriminant, min=0.0)

    # Use a numerically stable version of the quadratic formula
    root = (2 * c) / (-b - torch.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
    )
    derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, -logabsdet


@torch.jit.script
def _forward_transform_formula(
        inputs: torch.Tensor,
        input_cumwidths: torch.Tensor,
        input_bin_widths: torch.Tensor,
        input_cumheights: torch.Tensor,
        input_delta: torch.Tensor,
        input_derivatives: torch.Tensor,
        input_derivatives_plus_one: torch.Tensor,
        input_heights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the forward transform and its log absolute determinant."""
    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


@torch.jit.script
def rational_quadratic_spline(
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a rational-quadratic spline transform to the inputs.
    """
    # Input validation with torch.jit friendly comparisons
    min_val = torch.min(inputs).item()
    max_val = torch.max(inputs).item()
    if min_val < left or max_val > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # Compute widths, heights, and derivatives
    widths, cumwidths = _compute_widths(unnormalized_widths, num_bins, min_bin_width, left, right)
    heights, cumheights = _compute_heights(unnormalized_heights, num_bins, min_bin_height, bottom, top)
    derivatives = _compute_derivatives(unnormalized_derivatives, min_derivative)

    # Find bin indices
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    # Gather required bin parameters
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    # Apply the appropriate transform
    if inverse:
        return _inverse_transform_formula(
            inputs,
            input_cumheights,
            input_derivatives,
            input_derivatives_plus_one,
            input_delta,
            input_heights,
            input_bin_widths,
            input_cumwidths
        )
    else:
        return _forward_transform_formula(
            inputs,
            input_cumwidths,
            input_bin_widths,
            input_cumheights,
            input_delta,
            input_derivatives,
            input_derivatives_plus_one,
            input_heights
        )


# Cache the linear tails constant calculation
@lru_cache(maxsize=4)
def _get_linear_tail_constant(min_derivative: float) -> float:
    """Cache the computation of the linear tail constant."""
    return float(np.log(np.exp(1 - min_derivative) - 1))


def unconstrained_rational_quadratic_spline(
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        tails: Optional[Literal["linear"]] = "linear",
        tail_bound: float = 1.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a rational-quadratic spline transform with unconstrained inputs.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        # Handle padding of derivatives for linear tails
        constant = _get_linear_tail_constant(min_derivative)
        unnormalized_derivatives_padded = F.pad(unnormalized_derivatives, pad=(1, 1))
        unnormalized_derivatives_padded[..., 0] = constant
        unnormalized_derivatives_padded[..., -1] = constant

        # Process inputs outside the interval
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        # logabsdet is already zero for these inputs
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    # Only process inputs inside the interval if there are any
    if torch.any(inside_interval_mask):
        # Get the indices of elements inside the interval
        inside_indices = torch.nonzero(inside_interval_mask, as_tuple=True)

        # Apply the spline transform to inputs inside the interval
        transformed_outputs, transformed_logabsdet = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_indices[0], :] if len(
                inside_indices) > 0 else unnormalized_widths,
            unnormalized_heights=unnormalized_heights[inside_indices[0], :] if len(
                inside_indices) > 0 else unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives_padded[inside_indices[0], :] if len(
                inside_indices) > 0 else unnormalized_derivatives_padded,
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

        # Update the outputs and logabsdet for elements inside the interval
        outputs[inside_interval_mask] = transformed_outputs
        logabsdet[inside_interval_mask] = transformed_logabsdet

    return outputs, logabsdet


def piecewise_rational_quadratic_transform(
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        tails: Optional[Literal["linear"]] = None,
        tail_bound: float = 1.0,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a piecewise rational-quadratic transform to the inputs.
    This is the main entry point that delegates to the appropriate spline function.
    """
    # Pre-compute device and dtype for consistent tensor creation
    if tails is None:
        return rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
    else:
        return unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            tails=tails,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
