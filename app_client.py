import logging
import traceback
import os
import requests
from threading import Thread
from dataclasses import dataclass, field

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# launch app
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


class Data(BaseModel):
    id: int
    image_hex: str

    def to_dict(self):
        return {"id": self.id, "image_hex": self.image_hex}


@dataclass()
class ImageQueue:
    input_data_queue: dict = field(default_factory=lambda: {})
    output_data_queue: dict = field(default_factory=lambda: {})

    def add_input(self, input_data: Data) -> None:
        self.input_data_queue[input_data.id] = input_data.to_dict()

    def pop_input(self) -> dict | None:
        key = sorted(self.input_data_queue.keys())
        if len(key) == 0:
            return None
        return self.input_data_queue.pop(key[0])

    def add_output(self, output_data: Data) -> None:
        self.output_data_queue[output_data.id] = output_data.to_dict()

    def pop_output(self) -> dict | None:
        key = sorted(self.output_data_queue.keys())
        if len(key) == 0:
            return None
        return self.output_data_queue.pop(key[0])


@app.post("/add_input")
async def add_input(item: Data):
    try:
        image_queue.add_input(item)
        return JSONResponse(content={"input_queue": list(image_queue.input_data_queue.keys())})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.get("/pop_output")
async def pop_output():
    try:
        output = image_queue.pop_output()
        if not output:
            return JSONResponse(content={})
        return JSONResponse(content=output)
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())

image_queue = ImageQueue()


@dataclass()
class Endpoint:
    url: str
    max_concurrent_job: int = 2
    n_current_job: int = 0

    @property
    def is_available(self) -> bool:
        return self.n_current_job < self.max_concurrent_job


@dataclass()
class MultipleEndpoints:
    endpoints: dict[str, Endpoint]

    def is_available(self, url: str) -> bool:
        return self.endpoints[url].is_available

    def get_url(self) -> str | None:
        key = [k for k in self.endpoints.keys() if self.is_available(k)]
        if not key:
            return None
        return key[0]

    def increase_job_count(self, url: str) -> int:
        self.endpoints[url].n_current_job += 1
        return self.endpoints[url].n_current_job

    def decrease_job_count(self, url: str) -> int:
        self.endpoints[url].n_current_job -= 1
        return self.endpoints[url].n_current_job


raw_endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
endpoints = {e: Endpoint(e) for e in raw_endpoint.split(",")}
multiple_endpoints = MultipleEndpoints(endpoints)


def generate_image() -> None:
    url = multiple_endpoints.get_url()
    if not url:
        return
    data = image_queue.pop_input()
    if not data:
        return
    job_count = multiple_endpoints.increase_job_count(url)
    logger.info(f"[generate_image][id={data['id']}][url={url}] job requested ({job_count} jobs running)")
    with requests.post(f"{url}/generate_image", json=data) as r:
        assert r.status_code == 200, r.status_code
        response = r.json()
    logger.info(f"[generate_image][id={data['id']}][url={url}] job complete (time: {response['time']})")
    image_queue.add_output(Data(id=response["id"], image_hex=response["image_hex"]))
    multiple_endpoints.decrease_job_count(url)


@app.put("/process_input")
async def process_input():
    try:
        Thread(target=generate_image).start()
        return JSONResponse(content={})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())

