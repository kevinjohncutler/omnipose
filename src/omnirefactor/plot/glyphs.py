import numpy as np

def vector_to_arrow(vectors, flip_y: bool = False):
    """
    Convert one or many 2-D vectors (in y,x order) into Unicode arrow glyphs.

    Parameters
    ----------
    vectors : array-like
        Either a single (dy, dx) pair or an iterable of such pairs.
    flip_y : bool, default False
        If True, invert the y-sign (dy) before mapping.

    Returns
    -------
    str | list[str]
        A single glyph when a single vector is provided,
        otherwise a list of glyphs.
    """
    arr = np.asarray(vectors, dtype=float)

    # Ensure shape (N, 2)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("Expected a single 2-vector.")
        arr = arr.reshape(1, 2)
    elif arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Each vector must have exactly two components.")

    # Extract signed directions
    dy = np.sign(arr[:, 0]).astype(int)
    dx = np.sign(arr[:, 1]).astype(int)

    # Optionally flip only the y sign
    if flip_y:
        dy = -dy

    signs = list(zip(dy, dx))

    arrow_map = {
        ( 0,  0): '•',
        ( 0,  1): '→',
        ( 0, -1): '←',
        (-1,  0): '↓',
        ( 1,  0): '↑',
        (-1,  1): '↘',
        ( 1,  1): '↗',
        (-1, -1): '↙',
        ( 1, -1): '↖',
    }

    glyphs = [arrow_map.get(sig, '?') for sig in signs]
    return glyphs[0] if len(glyphs) == 1 else glyphs