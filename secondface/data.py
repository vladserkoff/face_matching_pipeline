"""
Data reading routines
"""

import glob
import random
from io import BytesIO
from typing import Iterable

from PIL import Image


def read_image(img_bytes: bytes) -> Image.Image:
    """
    Load image from a bytes string

    Parameters
    ----------
    img_bytes : bytes
        Image read as bytes

    Returns
    -------
    image : PIL.Image.Image
    """
    image = Image.open(BytesIO(img_bytes))
    return image


def load_images(pattern: str, shuffle: bool = True) -> Iterable[Image.Image]:
    """
    Lazily load a collection of images.

    Parameters
    ----------
    pattern : str
        Path pattern to load images.

    Returns
    -------
    images : generator of `PIL.Image.Image`s
        Collection of images.
    """
    paths = glob.glob(pattern)
    if shuffle:
        random.shuffle(paths)
    else:
        paths.sort()
    for path in paths:
        with open(path, 'rb') as file:
            img_bytes = file.read()
        image = read_image(img_bytes)
        yield image
