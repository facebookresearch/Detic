import gradio
import subprocess


def greet(name):
    return "Hello " + name + "!"


def inference():
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


app = gradio.Interface(fn=greet, inputs="text", outputs="text")

app.launch()
