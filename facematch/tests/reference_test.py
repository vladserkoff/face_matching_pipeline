#pylint: disable = C0111, R0201
"""
Tests reference database construction.
"""

import os
import sqlite3
import tempfile

import numpy as np
from PIL import Image

from facematch import reference

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')


class Test:
    def test_db_creation(self):
        _, tmp_db_path = tempfile.mkstemp()
        connection = reference.init_db(tmp_db_path)
        cursor = connection.cursor()
        cursor.execute("""
            select
                count(*)
            from
                sqlite_master
            where
                name = 'reference_faces'
            """)
        assert cursor.fetchone() == (1, )
        os.remove(tmp_db_path)

    def test_name_from_path(self):
        path1 = 'test_test_0001.jpg'
        path2 = 'Test Test.1.jpg'
        name1 = reference.get_name_from_path(path1)
        name2 = reference.get_name_from_path(path2)
        assert name1 == name2 == 'test test'

    def test_construction(self):
        _, tmp_db_path = tempfile.mkstemp()
        connection = reference.construct_references(tmp_db_path, TEST_DIR_PATH)
        assert isinstance(connection, sqlite3.Connection)
        cursor = connection.cursor()
        result = cursor.execute("""
            SELECT
                name,
                image,
                embedding
            FROM
                reference_faces
            """)
        names, face_images, embeddings = zip(*result)
        assert len(names) == 2
        assert isinstance(face_images[0], Image.Image)
        assert isinstance(embeddings[0], np.ndarray)
        embeddings = np.stack(embeddings)
        assert embeddings.shape == (2, 512)
        os.remove(tmp_db_path)
