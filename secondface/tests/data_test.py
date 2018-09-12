#pylint: disable = C0111, R0903, R0201
"""
Test that data readers return expected results
"""
import os

from PIL import Image

from secondface import data

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')
TEST_IMG_PATH = os.path.join(
    TEST_DIR_PATH, '11_Meeting_Meeting_11_Meeting_Meeting_11_633.jpg')


class Test:
    def test_file_read(self):
        with open(TEST_IMG_PATH, 'rb') as file:
            img = data.read_image(file.read())
        assert isinstance(img, Image.Image)
        assert img.size == (512, 428)

    def test_file_load(self):
        img = data.load_image(TEST_IMG_PATH)
        assert isinstance(img, Image.Image)
        assert img.size == (512, 428)

    def test_dir_read(self):
        pattern = 'face'
        images = data.ImageDataset(TEST_DIR_PATH, pattern)
        assert len(images) == 3
        assert isinstance(images[0], Image.Image)
