# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

project = "ComfyUI-nunchaku"
copyright = "2025, Nunchaku Team"
author = "Nunchaku Team"

cur_path = Path(__file__)
toml_path = cur_path.parent.parent.parent / "pyproject.toml"
with open(toml_path, "rb") as f:
    data = tomllib.load(f)
    version = data["project"]["version"]

if "dev" in version:
    version = version.replace("dev", "")

release = version
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx.ext.extlinks",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Include global link definitions -----------------------------------------
with open(Path(__file__).parent / "links.rst", encoding="utf-8") as f:
    rst_epilog = f.read()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

extlinks = {
    "nunchaku-issue": ("https://github.com/mit-han-lab/nunchaku/issues/%s", "nunchaku#%s"),
    "comfyui-issue": ("https://github.com/mit-han-lab/ComfyUI-nunchaku/issues/%s", "ComfyUI-nunchaku#%s"),
}

intersphinx_mapping = {
    "nunchaku": ("https://nunchaku.tech/docs/nunchaku", None),
}
