"""
TensorBoard viewer for offline visualization of training history.

This module provides utilities to visualize training progress from saved
loss history files, even when the model is not actively being trained.

Usage:
    # From command line:
    python -m omnirefactor.cli.tensorboard_viewer /path/to/model_loss_history.json

    # Or with a save directory (finds all loss history files):
    python -m omnirefactor.cli.tensorboard_viewer /path/to/save_dir --port 6007

    # Programmatically:
    from omnirefactor.cli.tensorboard_viewer import view_loss_history
    view_loss_history('/path/to/model_loss_history.json')
"""

import json
import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path


def create_tensorboard_events(loss_history, output_dir, run_name='training'):
    """
    Create TensorBoard event files from a loss history dictionary.

    Parameters
    ----------
    loss_history : dict
        Loss history dictionary with keys: epoch, batch, train_loss,
        epoch_loss, learning_rate, timestamp, raw_losses
    output_dir : str
        Directory to write TensorBoard event files
    run_name : str
        Name for this run (subdirectory in output_dir)

    Returns
    -------
    str
        Path to the created run directory
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError('TensorBoard not available. Install with: pip install tensorboard')

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)

    steps_per_epoch = None
    last_epoch = -1

    # First pass: determine steps per epoch and compute overall means
    epochs = loss_history.get('epoch', [])
    if epochs:
        from collections import Counter
        epoch_counts = Counter(epochs)
        if epoch_counts:
            steps_per_epoch = max(epoch_counts.values())

    if steps_per_epoch is None:
        steps_per_epoch = 1

    # Compute min/max for each loss term (for min-max normalization)
    loss_min = {}
    loss_max = {}
    raw_losses_list = loss_history.get('raw_losses', [])
    for entry in raw_losses_list:
        if entry is not None:
            for name, value in entry.items():
                if value is not None:
                    val_f = float(value)
                    if name not in loss_min:
                        loss_min[name] = val_f
                        loss_max[name] = val_f
                    else:
                        loss_min[name] = min(loss_min[name], val_f)
                        loss_max[name] = max(loss_max[name], val_f)

    n_entries = len(loss_history.get('train_loss', []))

    for i in range(n_entries):
        epoch = loss_history['epoch'][i] if i < len(loss_history['epoch']) else 0
        batch = loss_history['batch'][i] if i < len(loss_history['batch']) else i
        train_loss = loss_history['train_loss'][i]

        # Calculate global step
        global_step = epoch * steps_per_epoch + (batch if batch < steps_per_epoch else i % steps_per_epoch)

        # Log batch loss
        if train_loss is not None:
            writer.add_scalar('Loss/batch', train_loss, global_step)

        # Log epoch loss (once per epoch change)
        epoch_loss = loss_history['epoch_loss'][i] if i < len(loss_history.get('epoch_loss', [])) else None
        if epoch_loss is not None and epoch != last_epoch:
            writer.add_scalar('Loss/epoch_avg', epoch_loss, global_step)
            last_epoch = epoch

        # Log learning rate
        lr = loss_history['learning_rate'][i] if i < len(loss_history.get('learning_rate', [])) else None
        if lr is not None:
            writer.add_scalar('LearningRate', lr, global_step)

        # Log min-max normalized losses (0-1 scale)
        if i < len(raw_losses_list) and raw_losses_list[i] is not None:
            raw_losses = raw_losses_list[i]
            norm_dict = {}
            for name, value in raw_losses.items():
                if value is None:
                    continue
                val_f = float(value)
                lo = loss_min.get(name, val_f)
                hi = loss_max.get(name, val_f)
                rng = hi - lo
                norm_dict[name] = (val_f - lo) / rng if rng > 1e-12 else 0.5
            if norm_dict:
                writer.add_scalars('0_Loss', norm_dict, global_step)

    writer.close()
    return run_dir


def load_loss_history(path):
    """
    Load loss history from a JSON file.

    Parameters
    ----------
    path : str
        Path to JSON file or directory containing *_loss_history.json files

    Returns
    -------
    dict or list of tuples
        If path is a file: returns the loss history dict
        If path is a directory: returns list of (name, loss_history) tuples
    """
    path = Path(path)

    if path.is_file():
        with open(path, 'r') as f:
            return json.load(f)

    elif path.is_dir():
        histories = []
        # Look for loss history files
        for json_file in path.glob('*_loss_history.json'):
            name = json_file.stem.replace('_loss_history', '')
            with open(json_file, 'r') as f:
                histories.append((name, json.load(f)))

        # Also check in subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir():
                for json_file in subdir.glob('*_loss_history.json'):
                    name = f"{subdir.name}/{json_file.stem.replace('_loss_history', '')}"
                    with open(json_file, 'r') as f:
                        histories.append((name, json.load(f)))

        return histories

    else:
        raise FileNotFoundError(f'Path not found: {path}')


def view_loss_history(path, port=6006, host='localhost', launch_browser=True,
                      output_dir=None, keep_files=False):
    """
    View training loss history in TensorBoard.

    Creates TensorBoard event files from saved loss history and launches
    TensorBoard to view them.

    Parameters
    ----------
    path : str
        Path to loss history JSON file or directory containing them
    port : int
        Port to run TensorBoard on (default: 6006)
    host : str
        Host to bind TensorBoard to (default: localhost)
    launch_browser : bool
        Whether to open browser automatically (default: True)
    output_dir : str, optional
        Directory to write TensorBoard files. If None, uses a temp directory.
    keep_files : bool
        If True and output_dir is None, don't delete temp files on exit

    Returns
    -------
    str
        URL of TensorBoard server
    """
    import subprocess
    import webbrowser

    # Load history
    history_data = load_loss_history(path)

    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='omnirefactor_tb_')
        cleanup = not keep_files
    else:
        os.makedirs(output_dir, exist_ok=True)
        cleanup = False

    try:
        # Create event files
        if isinstance(history_data, dict):
            # Single file
            run_name = Path(path).stem.replace('_loss_history', '')
            create_tensorboard_events(history_data, output_dir, run_name)
            print(f'Created TensorBoard events for: {run_name}')
        else:
            # Multiple files
            for name, history in history_data:
                create_tensorboard_events(history, output_dir, name)
                print(f'Created TensorBoard events for: {name}')

        # Launch TensorBoard
        url = f'http://{host}:{port}'
        print(f'\nStarting TensorBoard at {url}')
        print(f'Log directory: {output_dir}')
        print('Press Ctrl+C to stop\n')

        if launch_browser:
            webbrowser.open(url)

        # Run TensorBoard
        cmd = [
            sys.executable, '-m', 'tensorboard.main',
            '--logdir', output_dir,
            '--port', str(port),
            '--host', host,
        ]

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print('\nTensorBoard stopped.')

    finally:
        if cleanup and os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='View training loss history in TensorBoard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # View a single loss history file:
    python -m omnirefactor.cli.tensorboard_viewer model_loss_history.json

    # View all loss histories in a directory:
    python -m omnirefactor.cli.tensorboard_viewer /path/to/save_dir

    # Use a different port:
    python -m omnirefactor.cli.tensorboard_viewer model_loss_history.json --port 6007

    # Save TensorBoard files instead of using temp directory:
    python -m omnirefactor.cli.tensorboard_viewer model_loss_history.json --output-dir ./tb_logs
'''
    )
    parser.add_argument('path', help='Path to loss history JSON file or directory')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard port (default: 6006)')
    parser.add_argument('--host', default='localhost', help='TensorBoard host (default: localhost)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--output-dir', '-o', help='Directory to save TensorBoard files')
    parser.add_argument('--keep-files', action='store_true',
                       help='Keep temporary files after TensorBoard exits')

    args = parser.parse_args()

    view_loss_history(
        args.path,
        port=args.port,
        host=args.host,
        launch_browser=not args.no_browser,
        output_dir=args.output_dir,
        keep_files=args.keep_files,
    )


if __name__ == '__main__':
    main()
