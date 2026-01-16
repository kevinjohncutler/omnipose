from .imports import *


def logger_setup(verbose=False):
    omni_dir = pathlib.Path.home().joinpath('.omnipose')
    omni_dir.mkdir(exist_ok=True)
    log_file = omni_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except Exception:
        print('creating new log file')
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    return logger, log_file
