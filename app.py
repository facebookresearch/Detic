import gradio
import subprocess
import os
from enum import Enum
import urllib.request


DeticModels = [
    {
        "model": "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size",
        "path": "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size.pth",
        "url": "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
    }
]


def inference(image):
    subprocess.run(
        [
            "python",
            "demo.py",
            "--config-file",
            "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            "--input",
            "desk.jpg",
            "--output",
            "out.jpg",
            "--vocabulary",
            "lvis",
            "--opts",
            "MODEL.WEIGHTS",
            "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        ]
    )


def prepare():
    for model in DeticModels:
        if not os.path.exists("models"):
            os.mkdir("models")
        if not os.path.exists(model["path"]):
            urllib.request.urlretrieve(model["url"], model["path"])


if __name__ == "__main__":
    prepare()
    # app = gradio.Interface(fn=inference, inputs=["image"], outputs=["image"])
    # app.launch()
