import os
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


def main():
    with open(Path(__file__).resolve().parent.parent / "test_data" / "inputs.yaml", "r") as f:
        config = yaml.safe_load(f)
    for group in config.get("inputs", []):
        base_url = group["base_url"]
        download_dir = group.get("download_dir", "input")
        for filename in tqdm(group["files"], desc="Downloading"):
            output_path = os.path.join(download_dir, filename)
            if os.path.exists(output_path):
                print(f"File {filename} already exists, skipping download.")
                continue
            print(f"Downloading {filename}...")
            full_url = base_url.format(filename=filename)

            # Download with requests - more reliable than wget
            response = requests.get(full_url, stream=True, timeout=30)
            response.raise_for_status()

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


if __name__ == "__main__":
    main()
