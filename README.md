# ComfyUI-Latent-Modifiers
A repository of ComfyUI nodes which modify the latent during the diffusion process.


## Latent Diffusion Mega Modifier (sampler_mega_modifier.py)
### Adds multiple parameters to control the diffusion process towards a quality the user expects.
* Sharpness: utilizes code from Fooocus's sampling process to sharpen the noise in the middle of the diffusion process.
This can lead to more perceptual detail, especially at higher strengths.

* Tonemap: Clamps conditioning noise (CFG) using a user-chosen method, which can allow for the use of higher CFG values.

* Rescale: Scales the CFG by comparing the standard deviation to the existing latent to dynamically lower the CFG.

* Extra Noise: Adds extra noise in the middle of the diffusion process, akin to how sharpness sharpens the noise.
