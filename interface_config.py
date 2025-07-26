import os
import requests
from typing import Dict, Union
import gradio as gr


endpoint = os.getenv("ENDPOINT", None)
if endpoint is None:
    raise ValueError("Endpoint not set.")
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
        negative_prompt,
        noise_scale_latent_image,
        noise_scale_latent_prompt,
        alpha
) -> Dict[str, Union[int, float, str]]:
    config = None
    for e in endpoint:
        with requests.post(
                f"{e}/update_config",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "noise_scale_latent_image": noise_scale_latent_image,
                    "noise_scale_latent_prompt": noise_scale_latent_prompt,
                    "alpha": alpha
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
        component_prompt = gr.Text(
            label="Prompt",
            max_lines=2,
            placeholder="Enter your prompt.",
            value=default_config["prompt"]
        )
        component_negative_prompt = gr.Text(
            label="Negative Prompt",
            max_lines=2,
            placeholder="Enter your negative prompt.",
            value=default_config["negative_prompt"]
        )
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
        component_alpha = gr.Slider(
            label="Alpha",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=float(default_config["alpha"])
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
                component_negative_prompt,
                component_noise_scale_latent_image,
                component_noise_scale_latent_prompt,
                component_alpha
            ],
            outputs=[result]
        )
demo.launch(server_name="0.0.0.0")
