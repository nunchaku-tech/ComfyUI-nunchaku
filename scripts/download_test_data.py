import os

import wget
from tqdm import tqdm

if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    filenames = ["logo.png", "robot.png", "strawberry.png"]

    for filename in tqdm(filenames):
        wget.download(
            f"https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/{filename}",
            out=os.path.join("input", filename),
        )
