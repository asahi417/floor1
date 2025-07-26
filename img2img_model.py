"""Model class for stable diffusion2."""
import logging
import random

import torch
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps
from PIL import Image


MODEL_IMAGE_RESOLUTION = (512, 512)
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_generator(seed: int | None = None) -> torch.Generator:
    if seed:
        return torch.Generator().manual_seed(seed)
    return torch.Generator().manual_seed(random.randint(0, np.iinfo(np.int32).max))


def resize_image(
        image: Image.Image | np.ndarray,
        width: int,
        height: int,
        return_array: bool = False
) -> Image.Image | np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.size == (width, height):
        return image
    # Calculate aspect ratios
    target_aspect = width / height  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image
    if image_aspect > target_aspect:  # Resize the image to match the target height, maintaining aspect ratio
        new_width = int(height * image_aspect)
        resized_image = image.resize((new_width, height), Image.LANCZOS)
        left, top, right, bottom = (new_width - width) / 2, 0, (new_width + width) / 2, height
    else:  # Resize the image to match the target width, maintaining aspect ratio
        new_height = int(width / image_aspect)
        resized_image = image.resize((width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left, top, right, bottom = 0, (new_height - height) / 2, width, (new_height + height) / 2
    resized_image = resized_image.crop((left, top, right, bottom))
    if return_array:
        return np.array(resized_image)
    return resized_image


def add_noise(waveform: torch.Tensor, noise_scale: float, seed: int) -> torch.Tensor:
    if noise_scale == 0:
        return waveform
    noise = torch.randn(*waveform.shape, dtype=waveform.dtype, generator=get_generator(seed)).to(waveform.device)
    energy_signal = torch.linalg.vector_norm(waveform) ** 2
    energy_noise = torch.linalg.vector_norm(noise) ** 2
    if energy_signal == float("inf"):
        scaled_noise = noise_scale * noise
    else:
        scale = energy_signal/energy_noise * noise_scale
        scaled_noise = scale.unsqueeze(-1) * noise
    return waveform + scaled_noise


class SDXLTurboImg2Img:

    height: int
    width: int
    base_model_id: str
    base_model: StableDiffusionXLImg2ImgPipeline
    cached_latent_prompt: dict[str, str | torch.Tensor] | None

    def __init__(self,
                 base_model_id: str = "stabilityai/sdxl-turbo",
                 height: int = MODEL_IMAGE_RESOLUTION[0],
                 width: int = MODEL_IMAGE_RESOLUTION[1],
                 deep_cache: bool = False):
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
        if torch_device.type in ["cuda", "mps"]:
            config = dict(
                variant="fp16",
                torch_dtype=torch.float16,
                device_map="balanced",
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
        else:
            config = dict(use_safetensors=True)
        self.base_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_model_id, **config)
        self.height = height
        self.width = width
        if deep_cache:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=self.base_model)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        self.base_model = self.base_model.to(torch_device)
        self.cached_latent_prompt = None

    def __call__(self,
                 image: Image.Image,
                 prompt: str,
                 seed: int | None = None,
                 noise_scale_latent_image: float | None = None,
                 noise_scale_latent_prompt: float | None = None) -> Image.Image:
        generator = get_generator(seed)
        image = resize_image(image, self.width, self.height)
        image_tensor = self.base_model.image_processor.preprocess(image)
        if self.cached_latent_prompt is None or self.cached_latent_prompt["prompt"] != prompt:
            LOGGER.info("generating latent text embedding")
            with torch.no_grad():
                prompt_embedding = self.base_model.encode_prompt(prompt=prompt)
            self.cached_latent_prompt = {
                "prompt": prompt,
                "prompt_embeds": prompt_embedding[0],
                "pooled_prompt_embeds": prompt_embedding[2],
            }
        prompt_embeds = self.cached_latent_prompt["prompt_embeds"]
        pooled_prompt_embeds = self.cached_latent_prompt["pooled_prompt_embeds"]
        if noise_scale_latent_prompt:
            prompt_embeds = add_noise(prompt_embeds, noise_scale_latent_prompt, seed)
            pooled_prompt_embeds = add_noise(pooled_prompt_embeds, noise_scale_latent_prompt, seed)
        LOGGER.info("generating latent image embedding")
        ts, nis = retrieve_timesteps(self.base_model.scheduler, 2, self.base_model.device)
        ts, _ = self.base_model.get_timesteps(nis, 0.5, self.base_model.device)
        with torch.no_grad():
            latents = self.base_model.prepare_latents(
                image=image_tensor,
                timestep=ts[:1],
                batch_size=1,
                num_images_per_prompt=1,
                dtype=prompt_embeds.dtype,
                device=self.base_model.device,
                generator=generator,
                add_noise=True
            )
        if noise_scale_latent_image:
            latents = add_noise(latents, noise_scale_latent_image, seed)
        LOGGER.info("generating image")
        with torch.no_grad():
            output = self.base_model(
                image=image_tensor,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=2,
                num_images_per_prompt=1,
                height=self.height,
                width=self.width,
                generator=generator,
                guidance_scale=0,
                strength=0.5
            ).images
        return output[0]

    @staticmethod
    def export(data: Image.Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
