"""Build the user guide and live API from the same source distribution."""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "DL Complexity"
author = "Алябушев, Дементьев, Курдюков, Чирков"
copyright = "2026, DL Complexity team"
release = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]["version"]
version = release
language = "ru"
root_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# These remain GitHub-readable reference documents. The site has its own
# navigation and a live API, avoiding duplicate object descriptions.
exclude_patterns = ["_build", "README.md", "API_REFERENCE.md", "CODE_WALKTHROUGH.md"]
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
html_theme_options = {"navigation_depth": 3}
html_context = {
    "display_github": True,
    "github_user": "intsystems",
    "github_repo": "dl-complexity",
    "github_version": "master",
    "conf_py_path": "/docs/",
}
