import os
import logging
import requests
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

import img2img_model_config
from util import resize_image, bytes2image, image2bytes

refresh_rate_sec = 0.3
endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


@dataclass()
class ImageQueue:

    max_num_image: int = 100
    ids: list[int] = field(default_factory=lambda: [])
    images: list[Image.Image] = field(default_factory=lambda: [])

    def add_input_image(self, image_id: int, image: Image.Image):
        self.ids = self.ids[-self.max_num_image:] + [image_id]
        self.images = self.images[-self.max_num_image:] + [image]

    def pop_input_image(self, image_id: int) -> Image.Image | None:
        try:
            index = self.ids.index(image_id)
        except ValueError:
            return None
        self.ids = self.ids[index:]
        self.images = self.images[index:]
        self.ids.pop(0)
        return self.images.pop(0)


image_queue = ImageQueue()


def main():
    # set window
    cv2.namedWindow("raw")
    cv2.namedWindow("source")
    cv2.namedWindow("output")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, img2img_model_config.IMAGE_WIDTH)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, img2img_model_config.IMAGE_HEIGHT)
    # start main loop
    frame_index = 0
    flag, frame = vc.read()

    start = time.time()
    while flag:
        flag, image = vc.read()
        frame_index += 1
        logger.info(f"[image_id={frame_index}] new frame")
        image = resize_image(image, width=img2img_model_config.IMAGE_WIDTH, height=img2img_model_config.IMAGE_HEIGHT)
        cv2.imshow("raw", np.array(image))
        image_queue.add_input_image(frame_index, image)
        image_hex = image2bytes(image)
        elapsed = time.time() - start
        if elapsed >= refresh_rate_sec:
            start = time.time()
            with requests.post(f"{endpoint}/add_input", json={"id": frame_index, "image_hex": image_hex}) as r:
                assert r.status_code == 200, r.status_code
                response = r.json()
                logger.info(f"[image_id={frame_index}] input_queue: {response['input_queue']}")

            with requests.put(f"{endpoint}/process_input") as r:
                assert r.status_code == 200, r.status_code

        with requests.get(f"{endpoint}/pop_output") as r:
            assert r.status_code == 200, r.status_code
            response = r.json()
            if response:
                image_source = image_queue.pop_input_image(response["id"])
                if image_source is not None:
                    image_output = bytes2image(response["image_hex"])
                    cv2.imshow("output", np.array(image_output))
                    cv2.imshow("source", np.array(image_source))
        wait_key = cv2.waitKey(10)  # mil-sec
        if wait_key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("raw")
    cv2.destroyWindow("output")
    cv2.destroyWindow("source")


if __name__ == '__main__':
    main()
