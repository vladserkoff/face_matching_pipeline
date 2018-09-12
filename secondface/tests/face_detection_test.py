#pylint: disable = C0111, R0903, R0201, C1801
"""
Tests that all faces are detected, and processed as expected
"""
import os

from PIL import Image

from secondface.face_detection import FaceDetector
from secondface.face import Face

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')
TEST_IMG_PATH_FACES = os.path.join(
    TEST_DIR_PATH, '11_Meeting_Meeting_11_Meeting_Meeting_11_633.jpg')
TEST_IMG_PATH_NO_FACES = os.path.join(TEST_DIR_PATH,
                                      '14_Traffic_Traffic_14_55.jpg')


class Test:
    def test_detection(self):
        detector = FaceDetector()
        image = Image.open(TEST_IMG_PATH_FACES)
        faces = detector.find_faces(image)
        assert len(faces) == 3
        face = faces[0]
        assert isinstance(face, Face)
        assert hasattr(face, 'box')
        assert len(face.box) == 4
        assert hasattr(face, 'image')
        assert hasattr(face, 'confidence')

    def test_no_detection(self):
        detector = FaceDetector()
        image = Image.open(TEST_IMG_PATH_NO_FACES)
        faces = detector.find_faces(image)
        assert len(faces) == 0

    def test_detection_batch(self):
        detector = FaceDetector()
        images = [
            Image.open(x)
            for x in [TEST_IMG_PATH_NO_FACES, TEST_IMG_PATH_FACES]
        ]
        faces = detector.find_faces_batch(images)
        assert len(faces) == 2
