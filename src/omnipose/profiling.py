# renamed from profile.py to avoid conflict with the pyinstrument package
from pyinstrument import Profiler
import functools

def pyinstrument_profile(func):
    """Decorator that runs `func` under pyinstrument Profiler and prints time as percentages of total runtime."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler(async_mode='disabled')

        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()

        # Show time as percentage of total runtime
        print(profiler.output_text(unicode=True, color=True, time='percent_of_total'))

        return result

    return wrapper