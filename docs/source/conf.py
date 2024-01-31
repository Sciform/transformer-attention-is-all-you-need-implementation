# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#   :show-inheritance:
#   :inherited-members:
#    :no-inherited-members:

import os
import sys
# Source code dir relative to this file
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# pylint: disable=redefined-builtin, invalid-name
project = 'Sciform MT Transformer Implementation'
copyright = '2024, Ursula Maria Mayer, Sciform GmbH'
author = 'Ursula Maria Mayer, Sciform GmbH'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    # Link to other project's documentation (see mapping below)
    'sphinx.ext.intersphinx',
    # Add a link to the Python source code for classes, functions etc.
    'sphinx.ext.viewcode',
    'sphinx_copybutton',  # Add a copy button to code examples
    # Automatically document param types (less noise in class signature)
    'sphinx_autodoc_typehints',
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    'sphinx_design',
    'sphinx_favicon',
    'IPython.sphinxext.ipython_console_highlighting',
    'myst_parser',
    "docs.source._extension.gallery_directive"
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True
nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# Pydata theme
html_theme = "pydata_sphinx_theme"
html_logo = "_static/LogoCantarell64_PlainCompanyName.png"
html_theme_options = {
    "show_prev_next": False,
    # "announcement": "Here's a Announcement!</a>",
    "external_links": [
        {"name": "Sciform", "url": "https://sciform.com"}
    ],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/Sciform/transformer-attention-is-all-you-need-implementation",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "navigation_with_keys": True,
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark"
}


html_static_path = ['_static']
html_css_files = ['pydata-custom.css']
