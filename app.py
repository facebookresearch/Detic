import gradio
import subprocess
import os
from enum import Enum
import urllib.request
from PIL import Image


DeticModels = [
    {
        "model": "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size",
        "path": "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        "url": "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
    }
]


def inference(image):
    Image.fromarray(image.astype("uint8"), "RGB").save("input.jpg")
    subprocess.run(
        [
            "python",
            "demo.py",
            "--config-file",
            "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            "--input",
            "input.jpg",
            "--output",
            "out.jpg",
            "--vocabulary",
            "lvis",
            "--opts",
            "MODEL.WEIGHTS",
            "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
            "MODEL.DEVICE",
            "cpu"
        ]
    )
    return Image.open('out.jpg')


def prepare():
    for model in DeticModels:
        if not os.path.exists("models"):
            os.mkdir("models")
        if not os.path.exists(model["path"]):
            urllib.request.urlretrieve(model["url"], model["path"])


if __name__ == "__main__":
    prepare()
    app = gradio.Interface(fn=inference, inputs=["image"], outputs=["image"])
    app.launch()
