from . import gpu # initialize torch 
from cellpose_omni.__main__ import main as cellpose_omni_main
from .cli import get_arg_parser
import traceback
import sys

parser = get_arg_parser()

def main():
    args = parser.parse_args() 
    args.omni = True
    cellpose_omni_main(args)

if __name__ == '__main__':

    print('running main')
    
    try:
        main()
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
    
