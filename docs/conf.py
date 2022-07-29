
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))

from sphinx_automodapi import automodsumm
from sphinx_automodapi.utils import find_mod_objs

def find_mod_objs_patched(*args, **kwargs):
    return find_mod_objs(args[0], onlylocals=True)

def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    automodsumm.find_mod_objs = find_mod_objs_patched

def setup(app):
    app.connect("builder-inited", patch_automodapi)
    # gen_color()
    # app.add_css_file('tablefix.css')
    # app.add_css_file('_static/sinebow.css')
    # app.add_css_file('_static/custom.css')
    # app.add_css_file("custom.css")

    

# -- Project information -----------------------------------------------------

project = 'omnipose'
copyright = '2022, Kevin Cutler, University of Washington'
author = 'Kevin Cutler'

# The full version, including alpha/beta/rc tags
release = '0.3.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Dark mode toggle
# extensions = ['sphinx_rtd_dark_mode','nbsphinx']
# default_dark_mode = True
extensions = [
    'sphinx.ext.autodoc',  # include documentation from docstrings
    'sphinx.ext.intersphinx', # external links
    'sphinx.ext.mathjax', # LaTeX style math
    'sphinx.ext.viewcode', # view code links
    'sphinx.ext.napoleon', # for NumPy style docstrings
    'sphinx.ext.autosummary',  # autosummary directive
    'sphinx.ext.autosectionlabel',  # use :ref:`Heading` for any heading
    'sphinx_copybutton',
    'sphinx_automodapi.automodapi',
    'myst_nb',
]

autoapi_dirs = ['../omnipose']
# autosectionlabel_prefix_document = True
# source_suffix=['.rst','.md']

# nb_custom_formats = {
#     '.ipynb': ['nbformat.reads', {'as_version': 4}, True],
# }

html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"] #plotly
nb_execution_mode = 'off'
render_figure_options = {'align':'center'}
nb_render_image_options = {'align':'center'}
nb_number_source_lines = True

myst_enable_extensions = ["dollarmath", "amsmath"]

master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '_build', '**.ipynb_checkpoints', 'links.rst', 'sinebow.rst']
rst_epilog =""
# Read link all targets from file
with open('links.rst') as f:
     rst_epilog += f.read()

# -- Options for HTML output -------------------------------------------------
# html_logo = '_static/favicon.ico'
# html_favicon = '_static/favicon.ico'
# html_logo = '_static/logo3.png'
html_logo = '_static/logo.png'
html_favicon = '_static/icon.png'


html_theme = 'furo'
# html_theme = 'sphinx_rtd_theme'
# html_theme_path = ["_themes", ]
# html_theme = 'theme' # use the theme in subdir 'theme'
# html_theme_path = ['.'] # make sphinx search for themes in current dir

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# furo
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    # '/_static/hacks.css',
    '/_static/sinebow.css',
    '/_static/custom.css',
    

]


from omnipose.utils import sinebow
import colour
N = 42
c = sinebow(N)
colors = [colour.rgb2hex(c[i]) for i in range(1,N+1)]
colordict = {}
for i in range(N):
    colordict['sinebow'+'%0d'%i] = colors[i]
    
dark = {
        "h2-color": "#aaa",

        # "color-brand-primary": "#6322f9",
        # "color-brand-primary": "#c8b600",
        # "color-brand-primary": "#04d8a3",
        "color-brand-primary": "#999",
        
        "color-brand-content": "#0de989",
        # "color-highlight-on-target":"#fe5130",
        "color-problematic":"#818181",
        # "color-highlight-on-target":"#c8b600",
        # "color-background-hover":"#8f0ae5",
        "color-api-name":"#f0147a",
        "color-api-pre-name":"#8f0ae5",
        # "color-api-paren":"#04a3d8",
        "color-api-keyword":"#04a3d8",
        
        "color-foreground-primary": "#ffffffcc", # for main text and headings
        "color-foreground-secondary": "#a0a0a0", # for secondary text
        "color-foreground-muted": "#818181", # for muted text
        "color-foreground-border": "#666666", # for content borders

        "color-background-primary": "#131313", # for content
        "color-background-secondary": "#191919", # for navigation + ToC, also the default for code block
        # because I can't seem to change the code block background, just make it consistent: 202020
        
        "color-background-hover": "#202020ff", # for navigation-item hover
        "color-background-hover--transparent": "#20202000",
        "color-background-border": "#333", # for UI borders
        "color-background-item": "#444", # for "background" items (eg: copybutton)

        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eee",

        # // Highlighted text (search)
        # --color-highlighted-background: #083563;

        # GUI Labels
        "color-guilabel-background": "#08356380",
        "color-guilabel-border": "#13395f80",

        # // API documentation
        # "color-api-keyword: var(--color-foreground-secondary);
        # "color-highlight-on-target: #333300;

        # Admonitions
        "color-admonition-background": "#181818",

        # Cards
        # "color-card-border: var(--color-background-secondary);
        "color-card-background": "#181818",
        # "color-card-marginals-background": var(--color-background-hover);


        #Highlighted text
        "color-highlight-on-target": "#1a1a1a", #only to suppress coloring when jumping to new page 
        # "color-link": "#888", # defaults to the brand color, that is fine
        "color-link--hover": "#f0147a",
        "color-link-underline": "#0000",
        "color-link-underline--hover": "#0000",
        # "color-sidebar-link-text": "#fff", 
        # "color-sidebar-link-text--top-level": "#f0147a", #arrow?  defaults to primary brand 
        

        # "color-card-marginals-background": "red",
        # 'color-sidebar-link-text--top-level': '#0000',
        # 'text-color': '#0000',
    }

light = {
        "h2-color": "#333",

        "color-brand-primary": "#000",
        
        "color-brand-content": "#0de989",
        # "color-highlight-on-target":"#fe5130",
        "color-problematic":"#ff4040",
        # "color-highlight-on-target":"#c8b600",
        # "color-background-hover":"#8f0ae5",
        "color-api-name":"#f0147a",
        "color-api-pre-name":"#8f0ae5",
        # "color-api-paren":"#04a3d8",
        "color-api-keyword":"#04a3d8",
        
        "color-foreground-primary": "black", # for main text and headings
        "color-foreground-secondary": "#5a5a5a", # for secondary text
        "color-foreground-muted": "#646464", # for muted text
        "color-foreground-border": "#878787", # for content borders

        "color-background-primary": "white", # for content
        "color-background-secondary": "#f9f9f9", # for navigation + ToC, also ALMOST default for code block
        
        "color-background-hover": "#efefefff", # for navigation-item hover
        "color-background-hover--transparent": "#efefef00",
        "color-background-border": "#eee", # for UI borders
        "color-background-item": "#ccc", # for "background" items (eg: copybutton)

        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eee",

    }

dark.update(colordict)
light.update(colordict)


html_theme_options = {
    # "externalrefs":True,
    # "sidebartextcolor": "cyan",
    "sidebar_hide_name": True,
    "top_of_page_button": "edit",
    "dark_css_variables": dark,
    # "extra_navbar": '<a href="installation.html" class="w3-bar-item w3-button"><span class="sinebow11">dfsfsdfs</span></a>',
    "light_css_variables": light.update(colordict),
    "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/kevinjohncutler/omnipose",
                "html": "",
                "class": "fa-brands fa-github",
            },
        ],
        
}





# Generate stub pages whenever ::autosummary directive encountered
# This way don't have to call sphinx-autogen manually
# autosummary_generate = True

# Use automodapi tool, created by astropy people. See:
# https://sphinx-automodapi.readthedocs.io/en/latest/automodapi.html#overview
# Normally have to *enumerate* function names manually. This will document
# them automatically. Just be careful, if you use from x import *, to exclude
# them in the automodapi:: directive
automodapi_toctreedirnm = 'api'  # create much better URL for the page
automodsumm_inherited_members = False

# One of 'class', 'both', or 'init'
# The 'both' concatenates class and __init__ docstring
# See: http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autoclass_content = 'both'

# Copybutton configuration
# See: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'
# pygments_dark_style = "monokai"

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

 
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# html_context = {
#     'css_files': [
#         '_static/custom.css',  # overrides for wide tables in RTD theme
#         ],
#     }