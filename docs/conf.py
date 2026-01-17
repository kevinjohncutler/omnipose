
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
import sys, os, re
import subprocess
from pathlib import Path
# Disable Numba JIT during doc builds—avoids compilation errors when
# Sphinx imports modules that use @njit functions.
# os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# sys.path.insert(0, os.path.abspath('.'))

# Ensure imports work regardless of working directory by anchoring to conf.py location
conf_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(conf_dir, '..', 'src', 'omnipose')))
sys.path.insert(0, os.path.abspath(os.path.join(conf_dir, '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(conf_dir, '..')))

# Avoid docutils traverse deprecation warning crash in Sphinx 8 + myst-nb.
from docutils import nodes
def _traverse_no_warn(self, *args, **kwargs):
    return self.findall(*args, **kwargs)
nodes.Node.traverse = _traverse_no_warn

# Add all the modules that can't be installed in the RTD environment
from dependencies import install_deps, gui_deps, distributed_deps
autodoc_mock_imports = install_deps + gui_deps + distributed_deps
autodoc_mock_imports += ["cv2", "tqdm", "skimage", "numba", "torch", 
                         "sklearn", #this one in particular is a problem because it registers different than the package name 
                         "torchvision", # may remove from imports 
                         ]

print("Mocking imports for autodoc:", autodoc_mock_imports)

# Function to strip version specifiers from package names
def strip_versions(dep_list):
    # Updated function to correctly process and strip version specifiers from package names
    stripped_list = []
    for dep in dep_list:
        # Split the dependency string on version specifiers and take the first part (the package name)
        dep_name = re.split(r'>=|==|<|<=|>', dep)[0]
        stripped_list.append(dep_name)
    return stripped_list

# Apply the corrected function to autodoc_mock_imports
autodoc_mock_imports = strip_versions(autodoc_mock_imports)
autodoc_mock_imports = [
    dep for dep in autodoc_mock_imports
    if dep not in {"numpy", "matplotlib"}
]
autodoc_mock_imports += ["colour"]

# Pre-mock heavy dependencies for automodapi imports.
from unittest.mock import MagicMock
from types import ModuleType

def _mock_module(name: str) -> None:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub in sys.modules:
            continue
        mod = ModuleType(sub)
        mod.__getattr__ = lambda _name: MagicMock()
        sys.modules[sub] = mod
        if i > 1:
            parent = sys.modules[".".join(parts[:i - 1])]
            setattr(parent, parts[i - 1], mod)

for dep in autodoc_mock_imports:
    if "-" in dep:
        continue
    _mock_module(dep)

# Common submodules imported at module import time.
for dep in ("numba.core", "numba.core.errors", "scipy.ndimage", "tqdm.auto"):
    _mock_module(dep)

# pygments
sys.path.append(os.path.abspath(os.path.join(conf_dir, "_pygments")))
pygments_style = 'style.CustomStyle'
pygments_dark_style = 'style.CustomStyle'

from sphinx_automodapi import automodsumm
from sphinx_automodapi.utils import find_mod_objs


def find_mod_objs_patched(*args, **kwargs):
    return find_mod_objs(args[0], onlylocals=True)

def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    automodsumm.find_mod_objs = find_mod_objs_patched

def setup(app):
    # app.add_css_file("custom.css") loaded above
    # app.add_js_file('custom.js') loaded above 

    app.connect("builder-inited", patch_automodapi)
    # gen_color()
    # app.add_css_file('tablefix.css')
    # app.add_css_file('_static/sinebow.css')
    # app.add_css_file('_static/custom.css')

    

# -- Project information -----------------------------------------------------
import datetime
current_year = datetime.datetime.now().year

# Set the copyright string
project = 'omnipose'
copyright = f'{current_year}, Kevin Cutler, University of Washington'
author = 'Kevin Cutler'

# The full version, including alpha/beta/rc tags
release = re.sub('^v', '', os.popen('git describe --tags').read().strip())
# The short X.Y version.
version = release


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
    'sphinx_design',
    'myst_nb',
    'sphinxarg.ext',
    # 'sphinxcontrib.autoprogram',
    # 'sphinxcontrib.programoutput',
    # 'sphinxcontrib.fulltoc'
]

# Avoid autodoc warnings for mocked module attributes.
autodoc_default_options = {
    "exclude-members": "nn",
}

MINIMAL_DOCS = os.environ.get("OMNIPOSE_DOCS_MINIMAL") == "1"
if MINIMAL_DOCS:
    extensions = [ext for ext in extensions if not ext.startswith("sphinx_automodapi")]



# autoapi_dirs = ['../omnipose']
autoapi_dirs = [os.path.abspath(os.path.join('..', 'src', 'omnipose'))]
autosectionlabel_prefix_document = True
# source_suffix=['.rst','.md']

# nb_custom_formats = {
#     '.ipynb': ['nbformat.reads', {'as_version': 4}, True],
# }

# html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"] #plotly
html_js_files = [
    'custom.js',
]

nb_execution_mode = 'off'
render_figure_options = {'align':'center'}
nb_render_image_options = {'align':'center'}#,'width':'100%'}
nb_number_source_lines = True

# myst_enable_extensions = ["dollarmath", "amsmath"]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_heading_anchors = 2 # add anchors to all headings

master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# Set the canonical URL for the latest version
html_baseurl = 'https://omnipose.readthedocs.io/'

# Configure HTML context for the canonical URL
html_context = {
    'canonical_url': html_baseurl
}

# Add the SEO template to your HTML templates
html_extra_path = ['_templates/seo.html']



# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'build',
    '_build',
    '**.ipynb_checkpoints',
    'links.rst',
    'sinebow.rst',
    '._*',
    '**/._*',
]
if MINIMAL_DOCS:
    exclude_patterns.extend(['api/**', 'readme_full.rst'])

# Exclude untracked docs sources so local scratch files are ignored by RTD.
def _git_tracked_paths(repo_root: Path):
    try:
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo_root)
    except Exception:
        return None
    tracked = set()
    for raw in output.decode().split("\0"):
        if raw:
            tracked.add(Path(raw))
    return tracked

docs_dir = Path(__file__).resolve().parent
repo_root = docs_dir.parent
tracked_paths = _git_tracked_paths(repo_root)
if tracked_paths:
    source_suffixes = {".rst", ".md", ".ipynb"}
    api_generated_dir = docs_dir / "api" / "api"
    for path in docs_dir.rglob("*"):
        if not path.is_file() or path.suffix not in source_suffixes:
            continue
        if api_generated_dir in path.parents:
            continue
        rel_repo = path.relative_to(repo_root)
        if rel_repo not in tracked_paths:
            exclude_patterns.append(path.relative_to(docs_dir).as_posix())
rst_epilog =""
# Read link all targets from file
with open('links.rst') as f:
     rst_epilog += f.read()

# -- Options for HTML output -------------------------------------------------
# html_logo = '_static/favicon.ico'
# html_favicon = '_static/favicon.ico'
# html_logo = '_static/logo3.png'
html_logo = '_static/logo.png' 
html_favicon = '_static/icon.ico'


html_theme = 'furo'
# html_theme = 'default'
# html_theme = 'sinebow'

# html_theme = 'sphinx_rtd_theme'
# html_theme_path = ["_themes", ]
# html_theme = 'theme' # use the theme in subdir 'theme'
# html_theme_path = ['. /' # make sphinx search for themes in current dir

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_static_path = []


# furo
html_css_files = [
#     '/_static/sinebow.css',
    '/_static/custom.css',
    
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    # '/_static/hacks.css',


]


# from omnipose.plot import sinebow
# for simplicity, just copy the function here
import numpy as np
def sinebow(N,bg_color=[0,0,0,0], offset=0):
    """ Generate a color dictionary for use in visualizing N-colored labels. Background color 
    defaults to transparent black. 
    
    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)
        
    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the  dictionary.
    
    Returns
    --------------
    Dictionary with entries {int:RGBA array} to map integer labels to RGBA colors. 
    
    """
    colordict = {0:bg_color}
    for j in range(N): 
        k = j+offset
        angle = k*2*np.pi / (N) 
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict


from matplotlib.colors import rgb2hex
N = 42
c = sinebow(N)
colors = [rgb2hex(c[i]) for i in range(1,N+1)]
colordict = {}
for i in range(N):
    colordict['sinebow'+'%0d'%i] = colors[i]

shared = {"color-problematic": "#818181",
          # "color-highlight-on-target":"#c8b600",
          # "color-background-hover":"#8f0ae5",
          "color-api-name": "#f0147a",
          "color-api-pre-name": "#8f0ae5",
        #   "pst-font-size-h5": "1.2rem",
#           --font-size--normal: 100%;
#   --font-size--small: 87.5%;
#   --font-size--small--2: 81.25%;
#   --font-size--small--3: 75%;
#   --font-size--small--4: 62.5%;
#             // API documentation
#   --color-api-background: var(--color-background-secondary);
#   --color-api-background-hover: var(--color-background-hover);
#   --color-api-overall: var(--color-foreground-secondary);
#   --color-api-name: var(--color-problematic);
#   --color-api-pre-name: var(--color-problematic);
#   --color-api-paren: var(--color-foreground-secondary);
#   --color-api-keyword: var(--color-foreground-primary);
#   --color-highlight-on-target: #ffffcc;



          # "color-api-paren":"#04a3d8",
          "color-api-keyword": "#04a3d8",

          #Highlighted text
          "color-highlight-on-target": "#0000", #only to suppress coloring when jumping to new page 
          # "color-link": "#888", # defaults to the brand color, that is fine
          "color-link--hover": "#f0147acc",
          "color-link-underline": "#0000",
          "color-link-underline--hover": "#0000",        
          # discovering a lot more by inspecting the css 
        #   "color-code-background": "#ff0000",      # not sure what this does 
        #   "color-code-foreground":"#0000ff", # text color saying "copied!"

            "color-toc-background": "#0000",
            "sd-color-shadow": "#0000",    
            # --color-toc-item-text--active, --color-toc-item-text
            "color-highlighted-background": "#0000", # search 
            "color-sidebar-item-background--current": "#0000", # do not keep it highlighted 
            
            "color-sidebar-search-background": '#0000',
            "color-sidebar-search-background--focus": '#0000',
        }

dark = {
        "h2-color": "#aaa",

        # "color-brand-primary": "#6322f9",
        # "color-brand-primary": "#c8b600",
        # "color-brand-primary": "#04d8a3",
        "color-brand-primary": "#666666",
        
        "color-brand-content": "#0de989",
        # "color-highlight-on-target":"#fe5130",

        
        "color-foreground-primary": "#ffffff", # for main text and headings
        "color-foreground-secondary": "#a0a0a0", # for secondary text
        "color-foreground-muted": "#818181", # for muted text
        "color-foreground-border": "#333", # for content borders

        "color-background-primary": "#111111", # for content
        "color-background-secondary": "#30303030", # for navigation + ToC, also the default for code block

        "mystnb-source-bg-color":"#30303030", # works when iv.cell div.cell_input, div.cell details.above-input>summary { has background-color none
        
        "color-background-hover": "#30303030", # for navigation-item hover
        "color-background-hover--transparent": "#0000",
        "color-card-background": "#30303030", # cards
        "color-background-border": "#333", # for UI borders
        "color-background-item": "#444", # foreground for "background" items (eg: copybutton)

        # Announcements
        "color-announcement-background": "#000000",
        "color-announcement-text": "#eee",

        # // Highlighted text (search)
        # --color-highlighted-background: #083563;

        # GUI Labels
        "color-guilabel-background": "#ff000080",
        "color-guilabel-border": "#00ff0080",

        # // API documentation
        # "color-api-keyword: var(--color-foreground-secondary);
        # "color-highlight-on-target: #333300;

        # Admonitions
        "color-admonition-background": "#181818",


        # "color-card-marginals-background": var(--color-background-hover);


        # "color-sidebar-link-text": "#fff", 
        # "color-sidebar-link-text--top-level": "#f0147a", #arrow?  defaults to primary brand 
        
        # "color-card-marginals-background": "red",
        # 'color-sidebar-link-text--top-level': '#0000',
        # 'text-color': '#0000',
        
        "color-sidebar-item-background": "#30303030", # maybe for readthedocs version
        "color-inline-code-background": "#30303030", # for inline code
        # --sd-color-shadow) coulduse this for shadow control

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
        "color-foreground-border": "#ddd", # for content borders

        "color-background-primary": "#ffff", # for content
        "color-background-secondary": "#eeeeee99", # for navigation + ToC, 
        "color-background-hover": "#eeeeee99", # for navigation-item hover
        "color-background-hover--transparent": "#0000",
        "mystnb-source-bg-color":"#eeeeee99",     # code input 
        
        
        "color-card-background": "#eeeeee99", # cards
    
        "color-background-border": "#ddd", # for UI borders
        "color-background-item": "#ccc", # foreground for "background" items (eg: copybutton)


        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eee",



    }

dark.update(colordict)
dark.update(shared)
light.update(colordict)
light.update(shared)


html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_button": "edit",
    "dark_css_variables": dark,
    "light_css_variables": light,
    "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/kevinjohncutler/omnipose",
                "html": "",
                "class": "fa-brands fa-github",
            },
        ],
        
}

js_vars = {
    "highlight_on_scroll": False,
}


# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True


# Generate stub pages whenever ::autosummary directive encountered
# This way don't have to call sphinx-autogen manually
autosummary_generate = True

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
# pygments_dark_style = "monokai"
# pygments_dark_style = "friendly_grayscale"

# pygments_style = 'custom'

# fix em dash being converted to double dash, straight to curly quotes, 
# and other headaches!
smartquotes = False

highlight_language = 'python'

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
        # "sidebar/ethical-ads.html",
        "ethicalads.html",
    ]
}
