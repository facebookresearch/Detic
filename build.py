"""Install """
import subprocess
from typing import Dict, Any

__all__ = ("build",)


def build_detectron2(setup_kwargs: Dict[str, Any]) -> None:
    subprocess.call(["pip3", "install", "git+https://github.com/facebookresearch/detectron2.git@v0.6"])

if __name__ == "__main__":
    build_detectron2({})
