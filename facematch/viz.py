# pylint: disable = R0914, C0103
"""
Visualization tools
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.font_manager as fmgr
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_default_font() -> ImageFont:
    fm = fmgr.FontManager()
    djv = [x for x in fm.ttffiles if x.endswith('DejaVuSans.ttf')][0]
    font = ImageFont.truetype(djv, 18)
    return font


DEFAULT_FONT = _get_default_font()
DEFAULT_RESIZE = (1024, 1024)


def resize(image: Image.Image,
           new_size: Tuple[int, int] = DEFAULT_RESIZE) -> Image.Image:
    """
    Fit image into new size, respecting the aspect ratio.

    Parameters
    ----------
    image : `PIL.Image.Image`
        Original image.
    new_size : tuple
        Resize image to fit into this box, (W x H)

    Returns
    -------
    resize_image : `PIL.Image.Image`
        Resized copy of the original image.
    """
    image = image.copy()
    image.thumbnail(new_size)
    return image


def show_as_grid(images: List[Image.Image],
                 ncols: int = 10,
                 padding: int = 2,
                 resize_to: Tuple[int, int] = DEFAULT_RESIZE) -> Image.Image:
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
    resize_to : tuple, default (1024, 1024)
        Parameter to pass to `plt.subplots`.

    Returns
    -------
    image : Image.Image
        Resulting image.
    """
    array = np.stack([np.array(x) for x in images])
    nmaps = array.shape[0]
    xmaps = min(ncols, nmaps)
    ymaps = int(np.ceil(nmaps / xmaps))
    height, width = int(array.shape[1] + padding), int(array.shape[2] +
                                                       padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3),
                    dtype=np.uint8)
    for y_coord in range(ymaps):
        for x_coord in range(xmaps):
            y_start, y_end = y_coord * height + padding, (y_coord + 1) * height
            x_start, x_end = x_coord * width + padding, (x_coord + 1) * width
            img_idx = (x_coord + y_coord * ncols)
            if img_idx >= nmaps:
                break
            grid[y_start:y_end, x_start:x_end, :] = array[img_idx]
    image = Image.fromarray(grid)
    if resize_to:
        image = resize(image, resize_to)
    return image


def plot_boxes(image: Image.Image,
               detected_faces: List[Dict[str, Any]]) -> Image.Image:
    """
    Plot original image with boxes around identified faces.

    Parameters
    ----------
    image : `PIL.Image.Image` instance
        Image used in `find_faces` step
    detected_faces : list of dicts
        Dictionaries with `box` fields for every face found on the `image`

    Returns
    -------
    image_with_boxes : `PIL.Image.Image`
        Original image with boxes drawn around identified faces.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for face in detected_faces:
        draw.rectangle(face['box'])
    return image


def overlay_text(image: Image,
                 text: Union[str, float, int, np.float, np.int]) -> Image:
    """
    Draw a text in the top-left corner of the image.

    Parameters
    ----------
    image : `PIL.Image.Image`
        Original image.
    text : str or float or int
        Text to draw on the image. If float, will be rounded to two
        decimal points.

    Returns
    -------
    img : `PIL.Image.Image`
        Copy of original image with overlayed text.
    """
    if isinstance(text, (float, np.floating)):
        text = round(text, 2)
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), str(text), font=DEFAULT_FONT)
    return image
