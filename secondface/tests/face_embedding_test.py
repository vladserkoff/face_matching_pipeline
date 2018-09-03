#pylint: disable = C0111, R0903, R0201
"""
Tests that all faces are detected, and processed as expected
"""
import os
import numpy as np

from secondface.data import load_images
from secondface.face_encoding import FaceEncoder

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')
TEST_IMGS_PAT = os.path.join(TEST_DIR_PATH, 'face*.jpg')

IMAGES = load_images(TEST_IMGS_PAT)


class Test:
    def test_encoding(self):
        encoder = FaceEncoder()
        embeddings = encoder.calculate_embeddings(IMAGES)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 128)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1)
