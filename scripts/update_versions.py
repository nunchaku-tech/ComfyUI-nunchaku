import json
import re
import sys
import urllib.request
from typing import Dict, List, Set, Tuple

# Required for version string parsing and sorting
from packaging.version import parse as parse_version

# --- Source URLs to query ---
GITHUB_API_URL = "https://api.github.com/repos/nunchaku-tech/nunchaku"
HF_API_URL = "https://huggingface.co/api/models/nunchaku-tech/nunchaku/tree/main"
MODEL_SCOPE_API_URL = (
    "https://modelscope.cn/api/v1/models/nunchaku-tech/nunchaku/repo/files?Revision=master&PageSize=500"
)
OUTPUT_FILENAME = "nunchaku_versions.json"


def _get_json_from_url(url: str) -> List[Dict] | Dict:
    """Fetches and parses JSON data from a given URL."""
    try:
        headers = {"User-Agent": "Nunchaku-Version-Updater-Workflow"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                return json.loads(response.read())
        print(f"Warning: Received status code {response.status} from {url}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error fetching data from {url}: {e}", file=sys.stderr)
        return {}


def get_nunchaku_versions_from_sources() -> Tuple[Set[str], Set[str]]:
    """Retrieves all unique Nunchaku versions from GitHub, HuggingFace, and ModelScope."""
    official_tags, dev_tags = set(), set()
    wheel_regex = re.compile(r"nunchaku-([^-+]+)")

    # Source 1: GitHub Releases
    releases = _get_json_from_url(f"{GITHUB_API_URL}/releases")
    if isinstance(releases, list):
        for release in releases:
            for asset in release.get("assets", []):
                if asset.get("name", "").endswith(".whl"):
                    match = wheel_regex.search(asset["name"])
                    if match:
                        version_str = match.group(1)
                        if "dev" in version_str:
                            dev_tags.add(version_str)
                        else:
                            official_tags.add(version_str)
                        break  # One wheel per release is enough to get the version

    # Sources 2 & 3: Hugging Face & ModelScope
    sources = {"huggingface": (HF_API_URL, "path"), "modelscope": (MODEL_SCOPE_API_URL, "Name")}
    for source_name, (url, path_key) in sources.items():
        api_response = _get_json_from_url(url)
        if not api_response:
            continue
        file_list = []
        if source_name == "modelscope" and isinstance(api_response, dict):
            file_list = api_response.get("Data", {}).get("Files", [])
        elif source_name == "huggingface" and isinstance(api_response, list):
            file_list = api_response

        for file_info in file_list:
            filename = file_info.get(path_key)
            if filename and filename.endswith(".whl"):
                match = wheel_regex.search(filename)
                if match:
                    version_str = match.group(1)
                    if "dev" in version_str:
                        dev_tags.add(version_str)
                    else:
                        official_tags.add(version_str)

    return official_tags, dev_tags


def main():
    """Main function to generate and save the versions config file."""
    print("Fetching all available versions from sources...")
    official_versions, dev_versions = get_nunchaku_versions_from_sources()

    if not official_versions and not dev_versions:
        print("Error: Could not fetch any version information. Aborting.", file=sys.stderr)
        sys.exit(1)

    # This structure must match the one expected by installers.py
    config = {
        "versions": sorted(list(official_versions), key=parse_version, reverse=True),
        "dev_versions": sorted(list(dev_versions), key=parse_version, reverse=True),
        "supported_torch": ["torch2.5", "torch2.6", "torch2.7", "torch2.8", "torch2.9"],
        "supported_python": ["cp310", "cp311", "cp312", "cp313"],
        "filename_template": "nunchaku-{version}+{torch_version}-{python_version}-{python_version}-{platform}.whl",
        "url_templates": {
            "github": "https://github.com/nunchaku-tech/nunchaku/releases/download/{version_tag}/{filename}",
            "huggingface": "https://huggingface.co/nunchaku-tech/nunchaku/resolve/main/{filename}",
            "modelscope": "https://modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/{filename}",
        },
    }

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(
        f"Successfully generated '{OUTPUT_FILENAME}' with {len(official_versions)} official and {len(dev_versions)} dev versions."
    )


if __name__ == "__main__":
    main()
