import logging
import sys
import os
from datetime import datetime
import textwrap
import re

# get rid of that annoying xmlschema warning
logging.getLogger('xmlschema').setLevel(logging.WARNING)

logging.getLogger('bfio').setLevel(logging.ERROR) # probably from cv2
logging.getLogger('OpenGL').setLevel(logging.ERROR) # not sure what uses this
logging.getLogger('qdarktheme').setLevel(logging.ERROR)


# LOGGER_FORMAT = "%(asctime)-20s\t[%(levelname)-5s]\t[%(filename)-10s %(lineno)-5d%(funcName)-18s]\t%(message)s"
LOGGER_FORMAT = "%(asctime)-20s\t%(levelname)-7s\t%(message)s"

def hex_to_ansi(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = [int((x / 255.0) * 5) for x in rgb]
    ansi_color = 16 + (r * 36) + (g * 6) + b
    return f'\033[38;5;{ansi_color}m'

def replace_url(message):
    home_path = os.path.expanduser("~")
    url_pattern = re.compile(r'file://[^\s]+')
    urls = url_pattern.findall(message)
    for url in urls:
        short_url = url.replace("file://", "")#.replace(home_path, "~")
        message = short_url
        
        # short_url = '/model_dir'
        # hyperlink = f'\033]8;;{url}\033\\{short_url}\033]8;;\033\\'
        # message = message.replace(url, hyperlink)
    return message

class ColoredFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

        self.COLORS = {
            'core': hex_to_ansi('#5c9edc'),  # Blue color for core.py
            'models': hex_to_ansi('#5cd97c'),  # Green color for models.py
            '__main__': hex_to_ansi('#fff44f'),  # Yellow color for __main__.py
            'io': hex_to_ansi('#ff7f0e'),  # Orange color for io.py
            'gpu': hex_to_ansi('#ff0055'),  # Pink color for gpu.py
            # 'model': hex_to_ansi('#ff00ff'),  # 
            'ENDC': '\033[0m'
        }

        self.last_module = None
        self.last_funcName = None

    def format(self, record):
        colored_record = record
        module_name = os.path.splitext(os.path.basename(colored_record.pathname))[0]

        # Get the immediate directory containing the file
        immediate_directory = os.path.basename(os.path.dirname(colored_record.pathname))

        # Combine the immediate directory and file name
        module_name_full = os.path.join(immediate_directory, module_name) + '.py'
        
        func_name = colored_record.funcName

        # Add asctime attribute to the record
        colored_record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]

        # Set a fixed width for the module and function names
        mwidth = 30

        # Crop the module name if it's longer than the fixed width
        if len(module_name_full) > mwidth:
            module_name_full = module_name_full[:mwidth // 2 - 2] + '...' + module_name_full[-mwidth // 2 + 1:]
        
        fwidth = 12

        # Crop the function name if it's longer than the fixed width
        if len(func_name) > fwidth:
            func_name = func_name[:fwidth // 2 - 2] + '...' + func_name[-fwidth // 2 + 1:]

        # Only print the module name when it changes
        if module_name != self.last_module:
            self.last_module = module_name
            prefix_module = f'{module_name_full:<{mwidth}}'
        else:
            prefix_module = ' ' * mwidth  # width + '.py'

        # Only print the function name when it changes
        if func_name != self.last_funcName:
            self.last_funcName = func_name
            prefix_func = f'{func_name:.<{fwidth}}()'
        else:
            prefix_func = ' ' * fwidth  # width + '() line '

        # Always print the line number
        line_info = f'\t line {colored_record.lineno:>3}'

        # Determine log level color and padding
        level_name = colored_record.levelname
        level_name_padded = f'[{level_name}]'.ljust(11)  # Adjust padding here to match the longest log level name

        # Replace URLs in the message
        message = replace_url(colored_record.getMessage())

        # Wrap the message
        # width = 150
        # indent = len(colored_record.asctime) + 1 + 11 + 1 + len(prefix_module) + len(prefix_func) + len(line_info) + 2
        # message_wrapped = textwrap.fill(message, width=width, subsequent_indent=' ' * indent)
        message_wrapped = message

        ansi_color = self.COLORS[module_name] if module_name in self.COLORS else '\033[90m' # gray if not specified above
        colored_record.msg = (
            ansi_color +
            colored_record.asctime + '\t' +
            level_name_padded +
            prefix_module + prefix_func + line_info + '\t' +
            message_wrapped + self.COLORS['ENDC']
        )

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