# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
import tomllib
from pathlib import Path

project = "ComfyUI-nunchaku"
copyright = "2025, Nunchaku Team"
author = "Nunchaku Team"

cur_path = Path(__file__)
toml_path = cur_path.parent.parent.parent / "pyproject.toml"
with open(toml_path, "rb") as f:
    data = tomllib.load(f)
    version = data["project"]["version"]

if "dev" in version:
    version = re.sub(r"dev\d*", "", version)

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
links_dir = Path(__file__).parent / "links"
rst_epilog = ""
if links_dir.exists() and links_dir.is_dir():
    for link_file in sorted(links_dir.glob("*.txt")):
        with open(link_file, encoding="utf-8") as f:
            rst_epilog += f.read()

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

html_theme_options = {
    "repository_url": "https://github.com/nunchaku-tech/ComfyUI-nunchaku",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    # "announcement": "🔥 Nunchaku v1.2 released!",
}

html_favicon = "_static/nunchaku.ico"
