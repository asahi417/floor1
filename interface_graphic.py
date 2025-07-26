import os
import logging
from threading import Thread
from time import time
import requests

import cv2
import numpy as np
from PIL import Image

from img2img_model import resize_image
from app import bytes2image, image2bytes


# set logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# set endpoint
max_queue = int(os.getenv("MAX_QUEUE", 10))
refresh_rate_sec = int(os.getenv("REFRESH_RATE_SEC", 30))
endpoint = os.getenv("ENDPOINT", None)
if endpoint is None:
    raise ValueError("Endpoint not set.")
endpoint = endpoint.split(",")
max_concurrent_job = os.getenv("MAX_CONCURRENT_JOB", "1")
max_concurrent_job = [int(i) for i in max_concurrent_job.split(",")]
if len(max_concurrent_job) == 1:
    max_concurrent_job = max_concurrent_job * len(endpoint)
assert len(max_concurrent_job) == len(endpoint)
max_concurrent_job = {e: m for e, m in zip(endpoint, max_concurrent_job)}
# set prompt
width = int(os.getenv("WIDTH", 512))
height = int(os.getenv("HEIGHT", 512))
wait_sec = int(os.getenv("WAIT_M_SEC", 1))


def initialize_cache():
    _url2count = {e: 0 for e in endpoint}
    _input_data_queue = {}
    _output_data_queue = {}
    return _url2count, _input_data_queue, _output_data_queue


url2count, input_data_queue, output_data_queue = initialize_cache()


def generate_image(input_image: Image.Image, image_id: int) -> None:
    if len(input_data_queue) < max_queue:
        image_hex = image2bytes(input_image)
        input_data_queue[image_id] = {"id": image_id, "image_hex": image_hex, "image_pil": input_image}
    urls = sorted([(e, url2count[e]) for e in endpoint if url2count[e] < max_concurrent_job[e]], key=lambda x: x[1])
    if len(urls) != 0:
        image_id = sorted(input_data_queue.keys())[0]
        data = input_data_queue.pop(image_id)
        input_image_pil = data.pop("image_pil")
        url = urls[0][0]
        url2count[url] += 1
        logger.info(f"[generate_image][id={image_id}] generate_image to {url}")
        with requests.post(f"{url}/generate_image", json=data) as r:
            assert r.status_code == 200, r.status_code
            response = r.json()
        image = bytes2image(response["image_hex"])
        logger.info(f"[generate_image][id={image_id}] complete (time: {response['time']}, size: {image.size})")
        output_data_queue[image_id] = [image, input_image_pil]
        url2count[url] -= 1
    else:
        logger.info(f"[generate_image] endpoint full")


def main():
    # set window
    cv2.namedWindow("original")
    cv2.namedWindow("original_aligned")
    cv2.namedWindow("generated")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # start main loop
    frame_index = 0
    latest_frame_index = 0
    flag, frame = vc.read()
    elapsed = time()
    global url2count
    global input_data_queue
    global output_data_queue
    generated_frames = []
    source_frames = []
    while flag:
        frame_index += 1
        flag, frame = vc.read()
        frame_pil = resize_image(frame, width=width, height=height)
        logger.info(f"[new frame]\n\t- image_id {frame_index}\n\t- queue {len(input_data_queue)}\n\t- size: {frame.size}")
        cv2.imshow("original", np.array(frame_pil))
        Thread(target=generate_image, args=[frame_pil, frame_index]).start()
        if len(output_data_queue) > 0:
            tmp_frame_index = sorted(output_data_queue.keys())[0]
            generated_frame_pil, original_frame_pil = output_data_queue.pop(tmp_frame_index)
            generated_frames.append([tmp_frame_index, generated_frame_pil])
            source_frames.append([tmp_frame_index, original_frame_pil])
            # if tmp_frame_index is from old frame, skip it
            if latest_frame_index < tmp_frame_index:
                cv2.imshow("generated", np.array(generated_frame_pil))
                cv2.imshow("original_aligned", np.array(original_frame_pil))
                latest_frame_index = tmp_frame_index
            else:
                logger.info("no image to show: conflict in generation")
        else:
            logger.info("no image to show: no output queue")
        if time() - elapsed > refresh_rate_sec:
            if len(input_data_queue) > max_queue - 1:
                logger.info("refresh cache")
                url2count, input_data_queue, _ = initialize_cache()
            elapsed = time()
        wait_key = cv2.waitKey(wait_sec)  # mil-sec
        if wait_key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("original")
    cv2.destroyWindow("generated")


if __name__ == '__main__':
    main()
