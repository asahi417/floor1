from io import BytesIO

from diffusers.utils import load_image
from PIL import Image


def bytes2image(image_hex: str) -> Image.Image:
    image_bytes = bytes.fromhex(image_hex)
    return Image.open(BytesIO(image_bytes))


def image2bytes(image: str | Image.Image) -> str:
    if isinstance(image, str):
        image = load_image(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return image_bytes.hex()
