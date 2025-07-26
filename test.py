import os
import requests
from util import bytes2image, image2bytes

# specify the endpoint you want to test
endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
sample_image = "sample_image_human.png"
prompt = [
    "A portrait, painting, Renaissance, by Michelangelo, passionate, mountains and forest in the back, warm color,"
    " happiness cheerful, HQ, 4k",
    "Cubism, geometric, Picasso, 20th, modern art, warm color, happiness cheerful, HQ, 4k",
    "Abstraction, distorted noisy picture, chaos, symbolism, sacred, dripping, pattern, HQ, 4k"
]
std = [0.1, 0.1, 0.1]
image_hex = image2bytes(sample_image)

# generate image
with requests.post(f"{endpoint}/generate_image", json={
    "id": 0,
    "pointer": 0,
    "image_hex": image_hex,
    "prompt": prompt,
    "std": std,
    "noise_scale_latent_image": 0.0,
    "noise_scale_latent_prompt": 0.0,
}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.0.jpg")

with requests.post(f"{endpoint}/generate_image", json={
    "id": 0,
    "pointer": 2,
    "image_hex": image_hex,
    "prompt": prompt,
    "std": std,
    "noise_scale_latent_image": 0.0,
    "noise_scale_latent_prompt": 0.0,
}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.2.jpg")

# update config
with requests.post(f"{endpoint}/update_config", json={
    "noise_scale_latent_image": 0.4,
    "noise_scale_latent_prompt": 0.0,
}) as r:
    assert r.status_code == 200, r.status_code
with requests.post(f"{endpoint}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.noise_scale_latent_image.jpg")

# update config
with requests.post(f"{endpoint}/update_config", json={
    "noise_scale_latent_image": 0.0,
    "noise_scale_latent_prompt": 2,
}) as r:
    assert r.status_code == 200, r.status_code
with requests.post(f"{endpoint}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.noise_scale_latent_prompt.jpg")

