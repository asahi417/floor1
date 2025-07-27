import os
import requests
from typing import Dict, Union

import gradio as gr

endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
endpoint = endpoint.split(",")


def validate_dict(d_1: Dict[str, Union[int, float, str]], d_2: Dict[str, Union[int, float, str]]) -> None:
    for k in d_1.keys():
        assert d_1[k] == d_2[k], f"{k} has different values: {d_1[k]}, {d_2[k]}"


def get_config() -> Dict[str, Union[int, float, str]]:
    config = None
    for e in endpoint:
        print(e)
        with requests.get(f"{e}/get_config") as r:
            assert r.status_code == 200, r.status_code
            tmp_config = r.json()
            if config is not None:
                validate_dict(config, tmp_config)
            config = tmp_config
    return config


default_config = get_config()


def update_config(
        prompt,
        noise_scale_latent_image,
        noise_scale_latent_prompt,
) -> Dict[str, Union[int, float, str]]:
    config = None
    for e in endpoint:
        with requests.post(
                f"{e}/update_config",
                json={
                    "prompt": prompt,
                    "noise_scale_latent_image": noise_scale_latent_image,
                    "noise_scale_latent_prompt": noise_scale_latent_prompt,
                }
        ) as r:
            assert r.status_code == 200, r.status_code
            tmp_config = r.json()
            if config is not None:
                validate_dict(config, tmp_config)
            config = tmp_config
    return config


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Tuning Img2Img Generation")
        prompt = default_config["prompt"]
        component_prompt = [
            gr.Text(label="Prompt 1", max_lines=1, placeholder="Prompt 1", value=prompt[0]),
            gr.Text(label="Prompt 2", max_lines=1, placeholder="Prompt 2", value=prompt[1]),
            gr.Text(label="Prompt 3", max_lines=1, placeholder="Prompt 3", value=prompt[2])
        ]


        component_noise_scale_latent_image = gr.Slider(
            label="Noise Scale (Image)",
            minimum=0.0,
            maximum=2.0,
            step=0.01,
            value=float(default_config["noise_scale_latent_image"])
        )
        component_noise_scale_latent_prompt = gr.Slider(
            label="Noise Scale (Prompt)",
            minimum=0.0,
            maximum=10.0,
            step=0.01,
            value=float(default_config["noise_scale_latent_prompt"])
        )
        component_seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=1_000_000,
            step=1,
            value=int(default_config["seed"])
        )
        run_button = gr.Button("Run", scale=0)
        result = gr.JSON(label="Configuration")
        gr.on(
            triggers=[run_button.click],
            fn=update_config,
            inputs=[
                component_prompt,
                component_noise_scale_latent_image,
                component_noise_scale_latent_prompt,
            ],
            outputs=[result]
        )
demo.launch(server_name="0.0.0.0")
