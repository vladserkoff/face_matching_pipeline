# pylint: disable = E1129
"""
Facenet implementation based on https://github.com/davidsandberg/facenet
"""

import numpy as np
import tensorflow as tf


class FaceNet:
    """
    Encode faces in vector space.
    """

    def __init__(self, frozen_graph_path: str) -> None:
        """
        Parameters
        ----------
        frozen_graph_path : str
            Path to a frozen tensorflow model
        """
        self.frozen_graph_path = frozen_graph_path
        self.ready = False
        self._session = None  # type: tf.Session

    def load_model(self) -> None:
        """
        Load model from frozengraph file.
        """
        with tf.gfile.GFile(self.frozen_graph_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        self._session = tf.Session(graph=graph)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run forward pass given the inputs

        Parameters
        ----------
        inputs : np.ndarray
            Array with dimensions (B x H x W x C)

        Returns
        -------
        embeddings : np.ndarray
            Array if inputs embeddings with dimensions (B x emb_size)
        """
        if not self.ready:
            AttributeError('You must first call `load_model`')
        input_placeholder = self._session.graph.get_tensor_by_name("input:0")
        embeddings_placeholder = self._session.graph.get_tensor_by_name(
            "embeddings:0")
        phase_train_placeholder = self._session.graph.get_tensor_by_name(
            "phase_train:0")
        feed_dict = {input_placeholder: inputs, phase_train_placeholder: False}
        embeddings = self._session.run(
            embeddings_placeholder, feed_dict=feed_dict)
        return embeddings
