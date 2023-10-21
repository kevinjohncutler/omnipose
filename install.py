import os
import sys

packages = ['.', 'cellpose-omni']
extras = ''

# Check if 'gui' was passed as a command line argument
if len(sys.argv) > 1 and sys.argv[1] == 'gui':
    extras = '[gui]'

for package in packages:
    os.system(f'pip install -e {package}')
    if extras and package != '.':
        # Save the current directory
        current_dir = os.getcwd()
        
        # Change to the package directory
        os.chdir(os.path.join(current_dir, package))
        
        # Install the extras
        os.system(f'pip install .{extras}')
        
        # Change back to the original directory
        os.chdir(current_dir)
