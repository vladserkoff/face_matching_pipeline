#pylint: disable = C0111, R0903, R0201
"""
Tests distance computation
"""
import numpy as np
from secondface.face_matching import find_closest


class Test:
    def test_closest(self):
        embeddings = np.array([[1, 2], [2, 4], [0, 1]], dtype=np.float)
        references = np.array([[0, 1], [2, 4], [0, .1], [.99, .99]])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        references /= np.linalg.norm(references, axis=1, keepdims=True)
        best_matches, distances = find_closest(embeddings, references)
        assert len(best_matches) == len(distances) == len(embeddings)
        assert np.all(best_matches == [1, 1, 0])
        assert np.all(np.isclose(distances, 1))
