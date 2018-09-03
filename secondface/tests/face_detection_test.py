#pylint: disable = C0111, R0903, R0201
"""
Tests that all faces are detected, and processed as expected
"""
import os

from secondface.data import read_image
from secondface.face_detection import FaceDetector

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')
TEST_IMG_PATH = os.path.join(
    TEST_DIR_PATH, '11_Meeting_Meeting_11_Meeting_Meeting_11_633.jpg')

with open(TEST_IMG_PATH, 'rb') as file:
    IMAGE = read_image(file.read())


class Test:
    def test_detection(self):
        detector = FaceDetector()
        faces = detector.find_faces(IMAGE)
        assert len(faces) == 3
