import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
    The following gaussian functions were utilized from the Fooocus UI, many thanks to github.com/Illyasviel !
'''
def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2  # Ensure output size matches input size
        self.register_buffer('kernel', torch.tensor(gaussian_kernel(kernel_size, sigma), dtype=torch.float32))
        self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)  # Repeat the kernel for each input channel

    def forward(self, x):
        x = F.conv2d(x, self.kernel.to(x), padding=self.padding, groups=self.channels)
        return x

gaussian_filter_2d = GaussianBlur(4, 7, 0.8)

'''
    As of August 18th (on Fooocus' GitHub), the gaussian functions were replaced by an anisotropic function for better stability.
'''
Tensor = torch.Tensor
Device = torch.DeviceObjType
Dtype = torch.Type
pad = torch.nn.functional.pad


def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, '2D Kernel size should have a length of 2.'
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def gaussian(
    window_size: int, sigma: Tensor | float, *, device: Device | None = None, dtype: Dtype | None = None
) -> Tensor:

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:

    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def _bilateral_blur(
    input: Tensor,
    guidance: Tensor | None,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:

    if isinstance(sigma_color, Tensor):
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def bilateral_blur(
    input: Tensor,
    kernel_size: tuple[int, int] | int = (13, 13),
    sigma_color: float | Tensor = 3.0,
    sigma_space: tuple[float, float] | Tensor = 3.0,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    return _bilateral_blur(input, None, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


def joint_bilateral_blur(
    input: Tensor,
    guidance: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    sigma_space: tuple[float, float] | Tensor,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
) -> Tensor:
    return _bilateral_blur(input, guidance, kernel_size, sigma_color, sigma_space, border_type, color_distance_type)


class _BilateralBlur(torch.nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] | int,
        sigma_color: float | Tensor,
        sigma_space: tuple[float, float] | Tensor,
        border_type: str = 'reflect',
        color_distance_type: str = "l1",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type
        self.color_distance_type = color_distance_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"sigma_color={self.sigma_color}, "
            f"sigma_space={self.sigma_space}, "
            f"border_type={self.border_type}, "
            f"color_distance_type={self.color_distance_type})"
        )


class BilateralBlur(_BilateralBlur):
    def forward(self, input: Tensor) -> Tensor:
        return bilateral_blur(
            input, self.kernel_size, self.sigma_color, self.sigma_space, self.border_type, self.color_distance_type
        )


class JointBilateralBlur(_BilateralBlur):
    def forward(self, input: Tensor, guidance: Tensor) -> Tensor:
        return joint_bilateral_blur(
            input,
            guidance,
            self.kernel_size,
            self.sigma_color,
            self.sigma_space,
            self.border_type,
            self.color_distance_type,
        )


# Below is perlin noise from https://github.com/tasptz/pytorch-perlin-noise/blob/main/perlin_noise/perlin_noise.py
from torch import Generator, Tensor, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
from math import pi

def get_positions(block_shape: Tuple[int, int]) -> Tensor:
    """
    Generate position tensor.

    Arguments:
        block_shape -- (height, width) of position tensor

    Returns:
        position vector shaped (1, height, width, 1, 1, 2)
    """
    bh, bw = block_shape
    positions = torch.stack(
        torch.meshgrid(
            [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
            indexing="xy",
        ),
        -1,
    ).view(1, bh, bw, 1, 1, 2)
    return positions


def unfold_grid(vectors: Tensor) -> Tensor:
    """
    Unfold vector grid to batched vectors.

    Arguments:
        vectors -- grid vectors

    Returns:
        batched grid vectors
    """
    batch_size, _, gpy, gpx = vectors.shape
    return (
        unfold(vectors, (2, 2))
        .view(batch_size, 2, 4, -1)
        .permute(0, 2, 3, 1)
        .view(batch_size, 4, gpy - 1, gpx - 1, 2)
    )


def smooth_step(t: Tensor) -> Tensor:
    """
    Smooth step function [0, 1] -> [0, 1].

    Arguments:
        t -- input values (any shape)

    Returns:
        output values (same shape as input values)
    """
    return t * t * (3.0 - 2.0 * t)


def perlin_noise_tensor(
    vectors: Tensor, positions: Tensor, step: Callable = None
) -> Tensor:
    """
    Generate perlin noise from batched vectors and positions.

    Arguments:
        vectors -- batched grid vectors shaped (batch_size, 4, grid_height, grid_width, 2)
        positions -- batched grid positions shaped (batch_size or 1, block_height, block_width, grid_height or 1, grid_width or 1, 2)

    Keyword Arguments:
        step -- smooth step function [0, 1] -> [0, 1] (default: `smooth_step`)

    Raises:
        Exception: if position and vector shapes do not match

    Returns:
        (batch_size, block_height * grid_height, block_width * grid_width)
    """
    if step is None:
        step = smooth_step

    batch_size = vectors.shape[0]
    # grid height, grid width
    gh, gw = vectors.shape[2:4]
    # block height, block width
    bh, bw = positions.shape[1:3]

    for i in range(2):
        if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
            raise Exception(
                f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
            )

    if positions.shape[0] not in (1, batch_size):
        raise Exception(
            f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
        )

    vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
    positions = positions.view(positions.shape[0], bh * bw, -1, 2)

    step_x = step(positions[..., 0])
    step_y = step(positions[..., 1])

    row0 = lerp(
        (vectors[:, 0] * positions).sum(dim=-1),
        (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
        step_x,
    )
    row1 = lerp(
        (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
        (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
        step_x,
    )
    noise = lerp(row0, row1, step_y)
    return (
        noise.view(
            batch_size,
            bh,
            bw,
            gh,
            gw,
        )
        .permute(0, 3, 1, 4, 2)
        .reshape(batch_size, gh * bh, gw * bw)
    )


def perlin_noise(
    grid_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    batch_size: int = 1,
    generator: Generator = None,
    *args,
    **kwargs,
) -> Tensor:
    """
    Generate perlin noise with given shape. `*args` and `**kwargs` are forwarded to `Tensor` creation.

    Arguments:
        grid_shape -- Shape of grid (height, width).
        out_shape -- Shape of output noise image (height, width).

    Keyword Arguments:
        batch_size -- (default: {1})
        generator -- random generator used for grid vectors (default: {None})

    Raises:
        Exception: if grid and out shapes do not match

    Returns:
        Noise image shaped (batch_size, height, width)
    """
    # grid height and width
    gh, gw = grid_shape
    # output height and width
    oh, ow = out_shape
    # block height and width
    bh, bw = oh // gh, ow // gw

    if oh != bh * gh:
        raise Exception(f"Output height {oh} must be divisible by grid height {gh}")
    if ow != bw * gw != 0:
        raise Exception(f"Output width {ow} must be divisible by grid width {gw}")

    angle = torch.empty(
        [batch_size] + [s + 1 for s in grid_shape], *args, **kwargs
    ).uniform_(to=2.0 * pi, generator=generator)
    # random vectors on grid points
    vectors = unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
    # positions inside grid cells [0, 1)
    positions = get_positions((bh, bw)).to(vectors)
    return perlin_noise_tensor(vectors, positions).squeeze(0)

def generate_1f_noise(tensor, alpha, k):
    """Generate 1/f noise for a given tensor.

    Args:
        tensor: The tensor to add noise to.
        alpha: The parameter that determines the slope of the spectrum.
        k: A constant.

    Returns:
        A tensor with the same shape as `tensor` containing 1/f noise.
    """
    fft = torch.fft.fft2(tensor)
    freq = torch.arange(1, len(fft) + 1, dtype=torch.float)
    spectral_density = k / freq**alpha
    noise = torch.randn(tensor.shape) * spectral_density
    return noise

def green_noise(width, height):
    noise = torch.randn(width, height)
    scale = 1.0 / (width * height)
    fy = torch.fft.fftfreq(width)[:, None] ** 2
    fx = torch.fft.fftfreq(height) ** 2
    f = fy + fx
    power = torch.sqrt(f)
    power[0, 0] = 1
    noise = torch.fft.ifft2(torch.fft.fft2(noise) / torch.sqrt(power))
    noise *= scale / noise.std()
    return torch.real(noise)

# Tonemapping functions

def train_difference(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    diff_AB = a.float() - b.float()
    distance_A0 = torch.abs(b.float() - c.float())
    distance_A1 = torch.abs(b.float() - a.float())

    sum_distances = distance_A0 + distance_A1

    scale = torch.where(
        sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.0).float()
    )
    sign_scale = torch.sign(b.float() - c.float())
    scale = sign_scale * torch.abs(scale)
    new_diff = scale * torch.abs(diff_AB)
    return new_diff

# Contrast function

def contrast(x: Tensor):
    # Calculate the mean and standard deviation of the pixel values
    mean = x.mean(dim=(1,2,3), keepdim=True)
    stddev = x.std(dim=(1,2,3), keepdim=True)
    # Scale the pixel values by the standard deviation
    scaled_pixels = (x - mean) / stddev
    return scaled_pixels

class ModelSamplerLatentMegaModifier:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sharpness_multiplier": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                              "sharpness_method": (["anisotropic", "gaussian"], ),
                              "tonemap_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "tonemap_method": (["reinhard", "arctan", "quantile"], ),
                              "tonemap_percentile": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                              "contrast_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                              "rescale_cfg_phi": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "extra_noise_type": (["gaussian", "perlin", "pink", "green"], ),
                              "extra_noise_method": (["add", "add_scaled", "speckle"], ),
                              "extra_noise_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mega_modify"

    CATEGORY = "clybNodes"

    def mega_modify(self, model, sharpness_multiplier, sharpness_method, tonemap_multiplier, tonemap_method, tonemap_percentile, contrast_multiplier, rescale_cfg_phi, extra_noise_type, extra_noise_method, extra_noise_multiplier):
        match sharpness_method:
            case "anisotropic":
                degrade_func = bilateral_blur
            case "gaussian":
                degrade_func = gaussian_filter_2d
            case _:
                print("For some reason, the sharpness filter could not be found.")
        
        def modify_latent(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            noise_pred = (cond - uncond)

            # Extra noise
            if extra_noise_multiplier > 0:
                match extra_noise_type:
                    case "gaussian":
                        extra_noise = torch.randn_like(cond)
                    case "perlin":
                        cond_size_0 = cond.size(dim=2)
                        cond_size_1 = cond.size(dim=3)
                        extra_noise = perlin_noise(grid_shape=(cond_size_0, cond_size_1), out_shape=(cond_size_0, cond_size_1), batch_size=4).to(cond.device).unsqueeze(0)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                    case "pink":
                        extra_noise = generate_1f_noise(cond, 2, extra_noise_multiplier).to(cond.device)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                    case "green":
                        cond_size_0 = cond.size(dim=2)
                        cond_size_1 = cond.size(dim=3)
                        extra_noise = green_noise(cond_size_0, cond_size_1).to(cond.device)
                        mean = torch.mean(extra_noise)
                        std = torch.std(extra_noise)

                        extra_noise.sub_(mean).div_(std)
                alpha_noise = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
                alpha_noise *= 0.001 * extra_noise_multiplier # User-input and weaken the strength so we don't annihilate the latent.
                match extra_noise_method:
                    case "add":
                        cond = cond + extra_noise * alpha_noise
                    case "add_scaled":
                        cond = cond + train_difference(cond, extra_noise, cond) * alpha_noise
                    case "speckle":
                        cond = cond + cond * extra_noise * alpha_noise
                    case _:
                        cond = cond + extra_noise * alpha_noise

            # Sharpness
            alpha = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
            alpha *= 0.001 * sharpness_multiplier # User-input and weaken the strength so we don't annihilate the latent.
            degraded_cond = degrade_func(cond) * alpha + cond * (1.0 - alpha) # Mix the modified latent with the existing latent by the alpha
            noise_pred_degraded = (degraded_cond - uncond) # New noise pred

            # After this point, we use `noise_pred_degraded` instead of just `cond` for the final set of calculations

            # Tonemap noise
            if tonemap_multiplier == 0:
                new_magnitude = 1.0
            else:
                match tonemap_method:
                    case "reinhard":
                        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred_degraded, dim=(1)) + 0.0000000001)[:,None]
                        noise_pred_degraded /= noise_pred_vector_magnitude

                        mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                        std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)

                        top = (std * 3 * (100 / tonemap_percentile) + mean) * tonemap_multiplier

                        noise_pred_vector_magnitude *= (1.0 / top)
                        new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
                        new_magnitude *= top

                        noise_pred_degraded *= new_magnitude
                    case "arctan":
                        noise_pred_vector_magnitude = (torch.linalg.vector_norm(noise_pred_degraded, dim=(1)) + 0.0000000001)[:,None]
                        noise_pred_degraded /= noise_pred_vector_magnitude

                        noise_pred_degraded = (torch.arctan(noise_pred_degraded * tonemap_multiplier) * (1 / tonemap_multiplier)) + (noise_pred_degraded * (100 - tonemap_percentile) / 100)

                        noise_pred_degraded *= noise_pred_vector_magnitude
                    case "quantile":
                        s: FloatTensor = torch.quantile(
                            (uncond + noise_pred_degraded * cond_scale).flatten(start_dim=1).abs(),
                            tonemap_percentile / 100,
                            dim = -1
                        ) * tonemap_multiplier
                        s.clamp_(min = 1.)
                        s = s.reshape(*s.shape, 1, 1, 1)
                        noise_pred_degraded = noise_pred_degraded.clamp(-s, s) / s
                    case _:
                        print("Could not tonemap, for the method was not found.")

            # Contrast, after tonemapping, to ensure user-set contrast is expected to behave similarly across tonemapping settings
            alpha = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
            alpha *= 0.001 * contrast_multiplier # User-input and weaken the strength so we don't annihilate the latent.
            noise_pred_degraded = contrast(noise_pred_degraded) * alpha + noise_pred_degraded * (1.0 - alpha) # Mix the modified latent with the existing latent by the alpha

            # Rescale CFG
            if rescale_cfg_phi == 0:
                x_final = uncond + noise_pred_degraded * cond_scale
            else:
                x_cfg = uncond + noise_pred_degraded * cond_scale
                ro_pos = torch.std(degraded_cond, dim=(1,2,3), keepdim=True)
                ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

                x_rescaled = x_cfg * (ro_pos / ro_cfg)
                x_final = rescale_cfg_phi * x_rescaled + (1.0 - rescale_cfg_phi) * x_cfg


            return x_final # General formula for CFG. uncond + (cond - uncond) * cond_scale

        m = model.clone()
        m.set_model_sampler_cfg_function(modify_latent)
        return (m, )