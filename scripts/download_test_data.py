import os
from pathlib import Path

import wget
import yaml
from tqdm import tqdm


def main():
    with open(Path(__file__).resolve().parent.parent / "test_data" / "images.yaml", "r") as f:
        config = yaml.safe_load(f)
    os.makedirs("input", exist_ok=True)
    for group in config.get("images", []):
        base_url = group["base_url"]
        for filename in tqdm(group["files"], desc="Downloading"):
            output_path = os.path.join("input", filename)
            if os.path.exists(output_path):
                print(f"File {filename} already exists, skipping download.")
                continue
            print(f"Downloading {filename}...")
            full_url = base_url.format(filename=filename)
            wget.download(full_url, out=output_path)


if __name__ == "__main__":
    main()
