from .imports import *

# helper function to check for a path; if it doesn't exist, make it 
def check_dir(path):
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path,exist_ok=True)

                
def findbetween(s,string1='[',string2=']'):
    """Find text between string1 and string2."""
    return re.findall(str(re.escape(string1))+"(.*)"+str(re.escape(string2)),s)[0]

def getname(path,prefix='',suffix='',padding=0):
    """Extract the file name."""
    return os.path.splitext(Path(path).name)[0].replace(prefix,'').replace(suffix,'').zfill(padding)


import platform
def adjust_file_path(file_path):
    """
    Adjust the file path based on the operating system.
    On macOS, replace '/home/user' with '/Volumes'.
    On Linux, replace '/Volumes' with the home directory path.
    On Windows, map either '/home/user' or '/Volumes' to the user home and
    normalize separators to the platform default.

    Args:
        file_path (str): The original file path.

    Returns:
        str: The adjusted file path.
    """
    system = platform.system()
    if system == 'Darwin':  # macOS
        adjusted_path = re.sub(r'^/home/\w+', '/Volumes', file_path)
    elif system == 'Linux':  # Linux
        home_dir = os.path.expanduser('~')
        adjusted_path = re.sub(r'^/Volumes', home_dir, file_path)
    elif system == 'Windows':  # Windows
        home_dir = os.path.expanduser('~')
        # Map WSL-like or macOS-style mounts to the Windows home and clean slashes.
        replace_with_home = lambda _match: home_dir
        adjusted_path = re.sub(r'^/home/[^/]+', replace_with_home, file_path)
        adjusted_path = re.sub(r'^/Volumes', replace_with_home, adjusted_path)
        adjusted_path = os.path.normpath(adjusted_path)
    else:
        print(f"No defined transformation for {system}")
        adjusted_path = file_path
    return adjusted_path


def find_files(directory, suffix, exclude_suffixes=[]):
    """
    Find files in a directory matching a suffix, excluding specific suffixes.

    Parameters:
    - directory: Path to the directory to search.
    - suffix: Suffix to match for file names.
    - exclude_suffixes: List of suffixes to exclude.

    Returns:
    - A list of matching file paths.
    """
    matching_files = []  # List to store matching files
    for root, dirs, files in os.walk(directory):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if name.endswith(suffix) and not any(name.endswith(exclude) for exclude in exclude_suffixes):
                filename = os.path.join(root, basename)
                matching_files.append(filename)  # Collect the matching file
    return matching_files