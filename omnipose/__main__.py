from cellpose_omni import main as cellpose_omni_main
from .cli import get_arg_parser

parser = get_arg_parser()

def main():
    args = parser.parse_args() 
    args.omni = True
    cellpose_omni_main(args)

if __name__ == '__main__':
    main()
    
