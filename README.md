# ComfyUI-Latent-Modifiers
A repository of ComfyUI nodes which modify the latent during the diffusion process.


## Latent Diffusion Mega Modifier (sampler_mega_modifier.py)
### Adds multiple parameters to control the diffusion process towards a quality the user expects.
* Sharpness: utilizes code from Fooocus's sampling process to sharpen the noise in the middle of the diffusion process.
This can lead to more perceptual detail, especially at higher strengths.

* Tonemap: Clamps conditioning noise (CFG) using a user-chosen method, which can allow for the use of higher CFG values.

* Rescale: Scales the CFG by comparing the standard deviation to the existing latent to dynamically lower the CFG.

* Extra Noise: Adds extra noise in the middle of the diffusion process, akin to how sharpness sharpens the noise.

* Contrast: Adjusts the contrast of the conditioning, can lead to more pop-style results. Essentially functions as a secondary CFG slider for stylization, without changing subject pose and location much, if at all.

### Tonemapping Methods Explanation:
* Reinhard: <p>Uses the reinhard method of tonemapping (from comfyanonymous' ComfyUI Experiments) to clamp the CFG if the difference is too strong.

  Lower `tonemap_multiplier` clamps more noise, and a lower `tonemap_percentile` will increase the calculated standard deviation from the original noise. Play with it!</p>
* Arctan: <p>Clamps the values dynamically using a simple arctan curve. [Link to interactive Desmos visualization](https://www.desmos.com/calculator/e4nrcdpqbl).

  Recommended values for testing: tonemap_multiplier of 5, tonemap_percentile of 90.</p>
* Quantile: <p>Clamps the values using torch.quantile for obtaining the highest magnitudes, and clamping based on the result.


  `Closer to 100 percentile == stronger clamping`. Recommended values for testing: tonemap_multiplier of 1, tonemap_percentile of 99.</p>

### Contrast Explanation:
<p>Scales the pixel values by the standard deviation, achieving a more contrasty look. In practice, this can effectively act as a secondary CFG slider for stylization. It doesn't modify subject poses much, if at all, which can be great for those looking to get more oomf out of their low-cfg setups.</p>

#### Current Pipeline:
>##### Add extra noise to conditioning -> Sharpen conditioning -> Tonemap conditioning -> Modify contrast of conditioning -> Rescale CFG

#### Why use this over `x` node?
Since the `set_model_sampler_cfg_function` hijack in ComfyUI can only utilize a single function, we bundle many latent modification methods into one large function for processing. This is simpler than taking an existing hijack and modifying it, which may be possible, but my (Clybius') lack of Python/PyTorch knowledge leads to this being the optimal method for simplicity. If you know how to do this, feel free to reach out through any means!

#### Can you implement `x` function?
Depends. Is there existing code for such a function, with an open license for possible use in this repository? I could likely attempt adding it! Feel free to start an issue or to reach out for ideas you'd want implemented.
