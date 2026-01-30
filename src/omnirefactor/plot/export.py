"""Video and animation export utilities."""

import os
import subprocess

import numpy as np
from scipy import ndimage

from ..utils.arrays import to_8_bit, to_16_bit


def export_gif(frames, basename, basedir, scale=1, fps=15, loop=0, bounce=True):
    """
    Export frame sequence to GIF using ffmpeg with palette generation.

    Parameters:
        frames: Array of frames (T, Y, X, C) or (T, Y, X) for grayscale
        basename: Base name for output file
        basedir: Directory to save to
        scale: Scale factor for frames (applied via scipy.ndimage.zoom)
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        bounce: If True, append reversed frames for ping-pong effect
    """
    if scale != 1:
        frames = ndimage.zoom(frames, [1, scale, scale, 1], order=0)
        # scaling is not working with ffmpeg, so I will just scale the frames with ndimage
        # should be timepoints x Y x X x channels
    try:
        if frames.ndim == 4:
            frame_width, frame_height, nchan = frames.shape[-3:]
            if nchan == 3:
                pixel_format = 'rgb24'
            else:
                pixel_format = 'rgba'
        else:
            frame_width, frame_height = frames.shape[-2:]
            pixel_format = 'gray'
            # turns out this gives the same size, maybe I would need to specify the palette too

        file = os.path.join(basedir, basename + '_{}_fps_scale_{}.gif'.format(fps, scale))

        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
                              '-f', 'rawvideo', '-vcodec',
                              'rawvideo', '-s', '{}x{}'.format(frame_height, frame_width), '-pix_fmt', pixel_format,
                              '-r', str(fps), '-i', '-', '-an',
                              '-filter_complex', '[0:v]palettegen=stats_mode=full[pal],[0:v][pal]paletteuse=dither=none',
                              '-vcodec', 'gif', '-loop', str(loop),
                              file], stdin=subprocess.PIPE)

        # loop over the frames
        frames_8_bit = to_8_bit(frames)
        if bounce:
            frames_8_bit = np.concatenate((frames_8_bit, frames_8_bit[::-1]), axis=0)
        for frame in frames_8_bit:
            # write frame to pipe
            p.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # close the pipe
        p.stdin.close()
        p.wait()


def export_movie(frames, basename, basedir, scale=1, fps=15):
    """
    Export frame sequence to MP4 video using ffmpeg.

    Parameters:
        frames: Array of frames (T, Y, X, C) - must have 3 or 4 channels
        basename: Base name for output file
        basedir: Directory to save to
        scale: Scale factor for output video
        fps: Frames per second
    """
    frame_width, frame_height, nchan = frames.shape[-3:]
    if nchan == 3:
        pixel_format = 'rgb48le'
    else:
        pixel_format = 'rgba64le'

    file = os.path.join(basedir, basename + '_{}_fps.mp4'.format(fps))

    p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
                          'rawvideo', '-s', '{}x{}'.format(frame_height, frame_width), '-pix_fmt', pixel_format,
                          '-r', str(fps), '-i', '-', '-f', 'lavfi', '-i', 'anullsrc', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale, scale),
                          '-shortest', '-c:v', 'mpeg4', '-q:v', '0',
                          file], stdin=subprocess.PIPE)

    # loop over the frames
    for frame in to_16_bit(frames):
        # write frame to pipe
        p.stdin.write(frame.tobytes())

    # close the pipe
    p.stdin.close()
    p.wait()
