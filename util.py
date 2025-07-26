from io import BytesIO

import numpy as np
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
