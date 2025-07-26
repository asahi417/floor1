"""Model class for stable diffusion2."""
from dataclasses import dataclass
import logging

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps
from PIL import Image
from DeepCache import DeepCacheSDHelper

import img2img_model_config
from util import resize_image


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


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


@dataclass()
class CachedPrompts:
    prompt: list[str]
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor | None = None
    pointer: float = -1

    def __post_init__(self):
        if not (len(self.prompt) == len(self.prompt_embeds) == len(self.pooled_prompt_embeds) == len(self.std)):
            raise ValueError("length mismatch")

    def get_embedding(self, pointer: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pointer != pointer:
            self.pointer = pointer
            var = self.std ** 2
            n = len(var)
            assert 0 <= pointer <= n
            denominator = (2 * torch.pi * var) ** .5
            weight = torch.exp(-(pointer - torch.arange(n)) ** 2 / (2 * var)) / denominator
            self.weight = (weight / weight.sum()).to(self.prompt_embeds.device)
        prompt_embeds = (self.weight.reshape(-1, 1, 1, 1) * self.prompt_embeds).sum(0)
        pooled_prompt_embeds = (self.weight.reshape(-1, 1, 1) * self.pooled_prompt_embeds).sum(0)
        return prompt_embeds.type(self.prompt_embeds.dtype), pooled_prompt_embeds.type(pooled_prompt_embeds.dtype)


class SDXLTurboImg2Img:

    base_model: StableDiffusionXLImg2ImgPipeline
    cached_prompt: CachedPrompts | None

    def __init__(self):
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            config = dict(
                variant="fp16",
                torch_dtype=torch.float16,
                device_map="balanced",
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
        else:
            config = dict(use_safetensors=True)
        self.base_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(img2img_model_config.MODEL_NAME, **config)
        self.cached_prompt = None
        helper = DeepCacheSDHelper(pipe=self.base_model)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    def get_text_embedding(self, prompt: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        LOGGER.info("generating latent text embeddings")
        with torch.no_grad():
            prompt_embeds = []
            pooled_prompt_embeds = []
            for p in prompt:
                embedding = self.base_model.encode_prompt(prompt=p)
                prompt_embeds.append(embedding[0])
                pooled_prompt_embeds.append(embedding[2])
        return torch.stack(prompt_embeds), torch.stack(pooled_prompt_embeds)

    def __call__(self,
                 image: Image.Image,
                 prompt: list[str],
                 pointer: float = 0,
                 std: list[float] | None = None,
                 seed: int = 42,
                 noise_scale_latent_image: float | None = None,
                 noise_scale_latent_prompt: float | None = None) -> Image.Image:

        LOGGER.info("process text embedding")
        if self.cached_prompt is None or self.cached_prompt.prompt != prompt:
            prompt_embeds, pooled_prompt_embeds = self.get_text_embedding(prompt)
            self.cached_prompt = CachedPrompts(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                std=torch.tensor(std, dtype=torch.float32)
            )
        prompt_embeds, pooled_prompt_embeds = self.cached_prompt.get_embedding(pointer)
        if noise_scale_latent_prompt:
            LOGGER.info("adding noise to the text embedding")
            prompt_embeds = add_noise(prompt_embeds, noise_scale_latent_prompt, seed)
            pooled_prompt_embeds = add_noise(pooled_prompt_embeds, noise_scale_latent_prompt, seed)

        LOGGER.info("generating latent image embedding")
        generator = get_generator(seed)
        image = resize_image(image, height=img2img_model_config.IMAGE_HEIGHT, width=img2img_model_config.IMAGE_WIDTH)
        image_tensor = self.base_model.image_processor.preprocess(image)
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
            return self.base_model(
                image=image_tensor,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=2,
                num_images_per_prompt=1,
                height=img2img_model_config.IMAGE_HEIGHT,
                width=img2img_model_config.IMAGE_WIDTH,
                generator=generator,
                guidance_scale=0,
                strength=0.5
            ).images[0]

    @staticmethod
    def export(data: Image.Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
