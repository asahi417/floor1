from dataclasses import dataclass, field
import logging
import traceback
from time import time

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from img2img_model import SDXLTurboImg2Img
from util import bytes2image, image2bytes


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# model config
model = SDXLTurboImg2Img()

# launch app
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


class ItemUpdateConfig(BaseModel):
    pointer: int | None = None
    prompt: list[str] | None = None
    std: list[float] | None = None
    noise_scale_latent_image: float | None = None
    noise_scale_latent_prompt: float | None = None


class ItemGenerateImage(ItemUpdateConfig):
    id: int
    image_hex: str


@dataclass()
class GenerationConfig:
    pointer: int = 0
    prompt: list[str] = field(
        default_factory=lambda: [
            "A portrait, painting, Renaissance, by Michelangelo, passionate, mountains and forest in the back, HQ, 4k",
            "Cubism, geometric, Picasso, 20th, modern art, warm color, happiness cheerful, HQ, 4k",
            "Abstraction, distorted noisy picture, chaos, symbolism, sacred, dripping, pattern, HQ, 4k"
        ]
    )
    std: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    noise_scale_latent_image: float = 0.0
    noise_scale_latent_prompt: float = 0.0

    def update(self, item: ItemUpdateConfig | ItemGenerateImage):
        if item.prompt is not None:
            self.prompt = item.prompt
        if item.std is not None:
            self.std = item.std
        if item.pointer is not None:
            self.pointer = item.pointer
        if item.noise_scale_latent_image is not None:
            self.noise_scale_latent_image = item.noise_scale_latent_image
        if item.noise_scale_latent_prompt is not None:
            self.noise_scale_latent_prompt = item.noise_scale_latent_prompt

    @property
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt, "std": self.std, "pointer": self.pointer,
            "noise_scale_latent_image": self.noise_scale_latent_image,
            "noise_scale_latent_prompt": self.noise_scale_latent_prompt
        }


generation_config = GenerationConfig()


@app.post("/update_config")
async def update_config(item: ItemUpdateConfig):
    try:
        generation_config.update(item)
        return JSONResponse(content=generation_config.to_dict)
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.get("/get_config")
async def get_config():
    try:
        return JSONResponse(content=generation_config.to_dict)
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.post("/generate_image")
async def generate_image(item: ItemGenerateImage):
    try:
        generation_config.update(item)
        start = time()
        generated_image = model(
            image=bytes2image(item.image_hex),
            pointer=generation_config.pointer,
            prompt=generation_config.prompt,
            std=generation_config.std,
            noise_scale_latent_image=generation_config.noise_scale_latent_image,
            noise_scale_latent_prompt=generation_config.noise_scale_latent_prompt,
        )
        elapsed = time() - start
        image_hex = image2bytes(generated_image)
        return JSONResponse(content={"id": item.id, "image_hex": image_hex, "time": elapsed})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
