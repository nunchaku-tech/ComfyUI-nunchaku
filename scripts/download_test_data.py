# isort: off
import os

import wget
from tqdm import tqdm

# isort: on

if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    filenames = ["logo.png", "robot.png", "strawberry.png", "strawberry-mask.png", "mushroom_depth.webp"]
    for filename in tqdm(filenames):
        if os.path.exists(os.path.join("input", filename)):
            print(f"File {filename} already exists, skipping download.")
            continue
        print(f"Downloading {filename}...")
        wget.download(
            f"https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/{filename}",
            out=os.path.join("input", filename),
        )
