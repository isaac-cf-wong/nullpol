from __future__ import annotations

import os
import sys

import nullpol

sys.path.insert(0, os.path.abspath('../../nullpol'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nullpol'
copyright = '2025, Isaac C. F. Wong'
author = 'Isaac C. F. Wong'
fullversion = nullpol.__version__
version = nullpol.__version__
release = fullversion
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.graphviz",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "sphinx_tabs.tabs",
    "sphinx_multiversion",
    "sphinx_copybutton",
    "autoapi.extension",
]

templates_path = ['_templates']
exclude_patterns = ["build", ".DS_store", "requirements.txt"]

pygments_style = "sphinx"

# settings for autoapi generation
autoapi_dirs = ["../../nullpol"]
autoapi_template_dir = "../_templates"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True

# copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "display_version": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

numpydoc_show_class_members = False

html_static_path = ['_static']

# Multiversion options
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"^(v1.*|1.*|0.3.12)$"

# Only include master and the current branch
smv_branch_whitelist = r"^(master|" + os.environ.get("CI_COMMIT_REF_NAME", "") + r")$"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = r"^(origin|upstream)$"

# Format for versioned output directories inside the build directory
smv_outputdir_format = "{ref.name}"

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False
