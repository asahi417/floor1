import os
import logging
import traceback
from typing import Optional
from time import time

import torch
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
width = int(os.getenv("WIDTH", 512))
height = int(os.getenv("HEIGHT", 512))
model = SDXLTurboImg2Img(height=height, width=width, deep_cache=True)

# launch app
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


class GenerationConfig:
    prompt: str = "surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
    seed: int = 42
    noise_scale_latent_image: float = 0.0
    noise_scale_latent_prompt: float = 0.0


def _update_config(
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        noise_scale_latent_image: Optional[float] = None,
        noise_scale_latent_prompt: Optional[float] = None,
):
    GenerationConfig.prompt = prompt or GenerationConfig.prompt
    GenerationConfig.seed = seed or GenerationConfig.seed
    if noise_scale_latent_image is not None:
        GenerationConfig.noise_scale_latent_image = noise_scale_latent_image
    if noise_scale_latent_prompt is not None:
        GenerationConfig.noise_scale_latent_prompt = noise_scale_latent_prompt


class ItemUpdateConfig(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    noise_scale_latent_image: Optional[float] = None
    noise_scale_latent_prompt: Optional[float] = None


@app.post("/update_config")
async def update_config(item: ItemUpdateConfig):
    try:
        _update_config(
            prompt=item.prompt,
            seed=item.seed,
            noise_scale_latent_image=item.noise_scale_latent_image,
            noise_scale_latent_prompt=item.noise_scale_latent_prompt,
        )
        return JSONResponse(content={
            "prompt": GenerationConfig.prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.get("/get_config")
async def get_config():
    try:
        return JSONResponse(content={
            "prompt": GenerationConfig.prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


class ItemGenerateImage(ItemUpdateConfig):
    id: int
    image_hex: str


@app.post("/generate_image")
async def generate_image(item: ItemGenerateImage):
    try:
        _update_config(
            prompt=item.prompt,
            seed=item.seed,
            noise_scale_latent_image=item.noise_scale_latent_image,
            noise_scale_latent_prompt=item.noise_scale_latent_prompt,
        )
        image = bytes2image(item.image_hex)
        start = time()
        with torch.no_grad():
            generated_image = model(
                image=image,
                prompt=GenerationConfig.prompt,
                seed=GenerationConfig.seed,
                noise_scale_latent_image=GenerationConfig.noise_scale_latent_image,
                noise_scale_latent_prompt=GenerationConfig.noise_scale_latent_prompt,
            )
        elapsed = time() - start
        image_hex = image2bytes(generated_image)
        return JSONResponse(content={
            "id": item.id,
            "image_hex": image_hex,
            "time": elapsed,
            "prompt": GenerationConfig.prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
