from .imports import *

import subprocess
def explore_object(obj):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets is not installed. Installing now...")
        subprocess.check_call(["pip", "install", "ipywidgets"])
        import ipywidgets as widgets
        from IPython.display import display

    output = widgets.Output()

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            output.clear_output()
            with output:
                try:
                    next_obj = getattr(obj, change['new'])
                    print(f"Selected: {change['new']}")
                    print(dir(next_obj))
                    if hasattr(next_obj, '__dict__'):
                        explore_object(next_obj)
                except Exception as e:
                    print(str(e))

    dropdown = widgets.Dropdown(
        options=[attr for attr in dir(obj) if not attr.startswith("__")],
        description='Attributes:',
    )

    dropdown.observe(on_change)
    display(widgets.HBox([dropdown, output]))