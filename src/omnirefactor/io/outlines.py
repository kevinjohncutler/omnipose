from .imports import *


def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')
