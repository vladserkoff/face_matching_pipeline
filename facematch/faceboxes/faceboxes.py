# pylint: disable = E1129
"""
Face detection based on "FaceBoxes: A CPU Real-time Face Detector with High Accuracy"
as implemented in https://github.com/TropComplique/FaceBoxes-tensorflow
"""

from typing import Tuple

import numpy as np
import tensorflow as tf


class FaceBoxes:
    """
    Face detection with FaceBoxes
    """

    def __init__(self, model_path: str) -> None:
        """
        Parameters
        ----------
        model_path : str
            Path to a frozen graph.
        """
        self.model_path = model_path
        self.ready = False
        self._session = None  # type: tf.Session

    def load_model(self) -> None:
        """
        Load model from frozengraph file.
        """
        with tf.gfile.GFile(self.model_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='import')
        self._session = tf.Session(graph=graph)
        self.ready = True

    def forward(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward pass given the inputs.

        Parameters
        ----------
        image : np.ndarray
            A numpy uint8 array with shape (B x H x W x C),
            that represents a RGB image.

        Returns
        -------
        boxes : np.ndarray
            A float numpy array of shape (num_faces, 4).
            Coordinates are `y_top`, `x_left`, `y_bottom`, `x_right`.
        scores : np.ndarray
            A float numpy array of shape (num_faces,).
        """
        if not self.ready:
            raise AttributeError('You must first call `load_model`')

        input_image = self._session.graph.get_tensor_by_name(
            'import/image_tensor:0')
        output_ops = [
            self._session.graph.get_tensor_by_name('import/boxes:0'),
            self._session.graph.get_tensor_by_name('import/scores:0')
        ]

        boxes, scores = self._session.run(
            output_ops, feed_dict={input_image: image})
        return boxes, scores
