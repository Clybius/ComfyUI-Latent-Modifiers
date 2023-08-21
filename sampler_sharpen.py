import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
    The following gaussian functions were utilized from the Fooocus UI, many thanks to github.com/lllyasviel !
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

class ModelSamplerSharpenNoiseTest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "multiplier": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                              "method": (["anisotropic", "gaussian"], ),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "sharpness_patch"

    CATEGORY = "clybNodes"

    def sharpness_patch(self, model, multiplier, method):
        #if method == "anisotropic":
        #    degrade_func = BilateralBlur()
        match method:
            case "anisotropic":
                degrade_func = bilateral_blur
            case "gaussian":
                degrade_func = gaussian_filter_2d
            case _:
                print("For some reason, the sharpness filter could not be found.")
        
        def sampler_sharpen(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            noise_pred = (cond - uncond)

            alpha = 1.0 - (timestep / 999.0)[:, None, None, None].clone() # Get alpha multiplier, lower alpha at high sigmas/high noise
            alpha *= 0.001 * multiplier # User-input and weaken the strength so we don't annihilate the latent.
            degraded_cond = degrade_func(cond) * alpha + cond * (1.0 - alpha) # Mix the modified latent with the existing latent by the alpha

            return uncond + (degraded_cond - uncond) * cond_scale # General formula for CFG. uncond + (cond - uncond) * cond_scale

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_sharpen)
        return (m, )


NODE_CLASS_MAPPINGS = {
    "ModelSamplerSharpenNoiseTest": ModelSamplerSharpenNoiseTest,
}
