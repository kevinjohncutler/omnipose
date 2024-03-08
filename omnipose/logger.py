import logging
import sys
import os

# get rid of that annoying xmlschema warning
logging.getLogger('xmlschema').setLevel(logging.WARNING) 

# LOGGER_FORMAT = "%(asctime)-20s\t[%(levelname)-5s]\t[%(filename)-10s %(lineno)-5d%(funcName)-18s]\t%(message)s"
LOGGER_FORMAT = "%(asctime)-20s\t[%(levelname)-5s]\t%(message)s"

def hex_to_ansi(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = [int((x/255.0)*5) for x in rgb]
    ansi_color = 16 + (r*36) + (g*6) + b
    return f'\033[38;5;{ansi_color}m'          
                  
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'core': hex_to_ansi('#5c9edc'),  # Blue color for core.py
        'models': hex_to_ansi('#5cd97c'),  # Green color for models.py
        'ENDC': '\033[0m'
    }

    last_module = None
    last_funcName = None

    def format(self, record):
        colored_record = record
        module_name = os.path.splitext(os.path.basename(colored_record.pathname))[0]
        func_name = colored_record.funcName

        # Add asctime attribute to the record
        colored_record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

        # Set a fixed width for the module and function names
        mwidth = 8

        # Crop the module name if it's longer than the fixed width
        if len(module_name) > mwidth:
            module_name = module_name[:mwidth//2-2] + '...' + module_name[-mwidth//2+1:]

        fwidth = 12

        # Crop the function name if it's longer than the fixed width
        if len(func_name) > fwidth:
            func_name = func_name[:fwidth//2-2] + '...' + func_name[-fwidth//2+1:]

        # Only print the module name when it changes
        if module_name != self.last_module:
            self.last_module = module_name
            prefix_module = f'{module_name:<{mwidth}}'
        else:
            prefix_module = ' ' * (mwidth )  # width + '.py'

        # Only print the function name when it changes
        if func_name != self.last_funcName:
            self.last_funcName = func_name
            prefix_func = f'{func_name:.<{fwidth}}()'
        else:
            prefix_func = ' ' * (fwidth)  # width + '() line '

        # Always print the line number
        line_info = f'\t line {colored_record.lineno}'

        if module_name in self.COLORS:
            message_color = self.COLORS[module_name] + colored_record.asctime + '\t[' + colored_record.levelname + ']\t' + prefix_module + prefix_func + line_info + '\t' + colored_record.getMessage() + self.COLORS['ENDC']
            colored_record.msg = message_color

        return colored_record.msg

    def formatMessage(self, record):
        return record.msg
        
def setup_logger(name=__name__):
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = ColoredFormatter()
    ch.setFormatter(formatter)

    # Remove all handlers if any exist
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the new handler
    root_logger.addHandler(ch)

    # Set the formatter for the root logger
    root_logger.handlers[0].setFormatter(formatter)

    # Return a logger with the specified name
    return logging.getLogger(name)
    
