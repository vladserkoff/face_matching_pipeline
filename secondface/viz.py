# pylint: disable = R0914, C0103
"""
Visualization tools
"""

from typing import List, Tuple

import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_as_grid(images: List[Image.Image],
                 ncols: int = 10,
                 padding: int = 2,
                 figsize: Tuple[int, int] = (15, 15)) -> mimg.AxesImage:
    """
    Draw a grid of images.
    Numpy translation of `torchvision.utils.make_grid`

    Parameters
    ----------
    images : list of `PIL.Image.Image`s
        Must all be of the same size.
    ncols : int, default 10
        Number of columns in the resulting image. Number of
        rows will be infered automatically.
    padding : int, default 2
        Space between images, in pixels.
    figsize : tuple, default (15, 15)
        Parameter to pass to `plt.subplots`.

    Returns
    -------
    image : matplotlib.image.AxesImage
        Resulting image.
    """
    array = np.stack([np.array(x) for x in images])
    nmaps = array.shape[0]
    xmaps = min(ncols, nmaps)
    ymaps = int(np.ceil(nmaps / xmaps))
    height, width = int(array.shape[1] + padding), int(array.shape[2] +
                                                       padding)
    grid = np.zeros(
        (height * ymaps + padding, width * xmaps + padding, 3), dtype=np.uint8)
    for y_coord in range(ymaps):
        for x_coord in range(xmaps):
            y_start, y_end = y_coord * height + padding, (y_coord + 1) * height
            x_start, x_end = x_coord * width + padding, (x_coord + 1) * width
            img_idx = (x_coord + y_coord * ncols)
            if img_idx >= nmaps:
                break
            grid[y_start:y_end, x_start:x_end, :] = array[img_idx]
    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    image = ax.imshow(grid, interpolation='nearest')
    return image
