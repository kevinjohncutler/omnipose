from pyinstrument import Profiler
import functools

def pyinstrument_profile(func):
    """Decorator that runs `func` under pyinstrument Profiler and prints results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler(async_mode='disabled')

        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()

        # Print human-readable summary to stdout
        print(profiler.output_text(unicode=True, color=True))

        return result

    return wrapper