"""
Face object
"""

from io import BytesIO
from typing import List

from PIL import Image

from secondface.viz import overlay_text


class Face:
    """
    Face object, that at least contains following fields:
        * `box`: list of face box coordinates
        * `image`: `PIL.Image.Image`, image of the face, cropped and squared
        * `confidence`: float, model's confidence in the image being a face
    """

    def __init__(self, box: List[int], image: Image.Image,
                 confidence: float) -> None:
        """
        Parameters
        ----------
        box: list
            [x_left, y_top, x_right, y_bottom] of the face area
        image: `PIL.Image.Image`
            Cropped face, resized for facenet model
        confidence: float
            Model's confidence in the image being a face
        """
        self.box = box
        self.image = image
        self.confidence = confidence

    def __repr__(self):
        stub = 'Face at {}, confidence: {:.2f}'
        return stub.format(str(self.box), self.confidence)

    def _repr_png_(self):
        """
        ipython display hook support
        """
        stub = '{}\nconf: {:.2f}'
        text = stub.format(str(self.box), self.confidence)
        img = overlay_text(self.image, text)
        buff = BytesIO()
        img.save(buff, 'PNG')
        return buff.getvalue()
