"""
Data reading routines
"""

import os
import random
from copy import deepcopy
from io import BytesIO
from typing import Iterable, Iterator

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def load_image(path: str) -> Image.Image:
    """
    Read and load image from file as a pillow image.

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : PIL.Image.Image
    """
    with open(path, 'rb') as file:
        img_bytes = file.read()
    image = read_image(img_bytes)
    return image


def read_image(img_bytes: bytes) -> Image.Image:
    """
    Load image from a bytes string.

    Parameters
    ----------
    img_bytes : bytes
        Image read as bytes

    Returns
    -------
    image : PIL.Image.Image
    """
    image = Image.open(BytesIO(img_bytes))
    if image.mode != 'RGB':
        image = image.convert(mode='RGB')
    return image


class ImageDataset(Iterable):
    """
    Lazily load a collection of images.
    """

    def __init__(self, root_dir: str, pattern: str = '',
                 shuffle: bool = True) -> None:
        """
        Parameters
        ----------
        root_dir : str
            Path to a root directory with images.
        pattern : str, default ''
            Only read images that contain `pattern` in its' names.
        shuffle : bool, default True
            Wither to shuffle images or return them in alphabetical order.
        """
        self.root_dir = root_dir
        self.pattern = pattern
        self.shuffle = shuffle
        self.paths = self._find_images()

    def _find_images(self):
        paths = [
            os.path.join(directory, file)
            for directory, _, files in os.walk(self.root_dir) for file in files
            if (os.path.splitext(file)[1].lower() in IMG_EXTENSIONS) and (
                self.pattern in file)
        ]
        if self.shuffle:
            random.shuffle(paths)
        else:
            paths.sort()
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        """
        Parameters
        ----------
        index : int
            Index of an image

        Returns
        -------
        image : `PIL.Image.Image`
            An image with particular index.
        """
        path = self.paths[index]
        if isinstance(index, slice):
            subset = deepcopy(self)
            subset.paths = path
            return subset
        image = load_image(path)
        return image

    def __iter__(self) -> Iterator[Image.Image]:
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        fmt_str = 'Image Dataset' + '\n'
        fmt_str += '    Root directory: "{}"\n'.format(self.root_dir)
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        return fmt_str
