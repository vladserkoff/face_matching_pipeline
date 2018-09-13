"""
Reference faces database
"""

import io
import os
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List

import numpy as np
from PIL import Image

from facematch.data import ImageDataset
from facematch.face import Face
from facematch.face_detection import FaceDetector
from facematch.face_encoding import FaceEncoder


class Reference:
    """
    Reference database
    """

    def __init__(self, db_path: str, images_path: str = None) -> None:
        """
        Parameters
        ----------
        db_path : str
            Path to sqlite database file
        images_path : str, default None
            Path to a directory with reference photos.

        Note
        ----
        If both `db_path` and `images_path` are not None
        and there exists `db_path` file, `images_path` will be ignored.
        """
        self.db_path = db_path
        self.images_path = images_path
        self.load_references()

    def load_references(self) -> None:
        """
        Load from file reference database, that contains names, images
        and embeddings of reference people.
        """
        # Register the new adapters
        _register_sqlite_converters()
        if not os.path.isfile(self.db_path):
            if not self.images_path:
                raise ValueError(
                    "Database at '%s' is not present, `images_path` must be provided"
                    % self.db_path)
            connection = construct_references(self.db_path, self.images_path)
        else:
            connection = sqlite3.connect(
                self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
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
        embeddings = np.stack(embeddings)
        connection.close()

        self.names = names
        self.images = face_images
        self.embeddings = embeddings

    def __repr__(self):
        return '%i reference faces with %iD embeddings' % (len(
            self.names), self.embeddings.shape[-1])


def _register_sqlite_converters():
    def adapt_array(array):
        """
        Using the np.save function to save a binary version of the array,
        and BytesIO to catch the stream of data and convert it into a sqlite3.Binary.
        """
        out = io.BytesIO()
        np.save(out, array)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(blob):
        """
        Using BytesIO to convert the binary version of the array back into a numpy array.
        """
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)

    def convert_image(blob):
        """
        Using BytesIO to convert the binary version of the array back into a Image.Image.
        """
        out = io.BytesIO(blob)
        out.seek(0)
        img_array = np.load(out)
        return Image.fromarray(img_array)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter('array', convert_array)
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter('image', convert_image)


def construct_references(db_path: str, images_path: str) -> sqlite3.Connection:
    """
    Construct reference database, that contains names, images
    and embeddings of reference people.

    Parameters
    ----------
    db_path : str
        Path to sqlite database file
    images_path : str, default None
        Path to a directory with reference photos.

    Returns
    -------
    connection : sqlite3.Connection
        Database connection
    """
    _register_sqlite_converters()
    connection = init_db(db_path)
    detector = FaceDetector(min_confidence=0.5)
    encoder = FaceEncoder()
    images = ImageDataset(images_path, shuffle=False)

    detected = detector.find_faces_batch(images)

    main_faces = {
        i: select_main_face(all_faces)
        for i, all_faces in enumerate(detected) if all_faces
    }
    print('Found %i faces in total' % len(main_faces))

    persons = defaultdict(list)  # type: Dict[str, List[int]]
    for idx, path in enumerate(images.paths):
        if idx in main_faces:
            name = get_name_from_path(path)
            persons[name].append(idx)
    print('Found %i persons' % len(persons))

    print('Generating embeddings')
    persons_embeddings = {}
    for i, (name, img_indexes) in enumerate(persons.items()):
        embedding = get_person_embedding(main_faces, img_indexes, encoder)
        persons_embeddings[name] = embedding
        print(
            '\r{} of {} embeddings generated'.format(i + 1, len(persons)),
            end='')
    print()

    cursor = connection.cursor()
    for i, (name, emb) in enumerate(persons_embeddings.items()):
        image = main_faces[persons[name][0]].image
        cursor.execute(
            """
            INSERT INTO reference_faces
                (name, image, embedding)
            VALUES
                (?, ?, ?)
            """, [name, np.array(image), emb])
    connection.commit()
    cursor.close()
    return connection


def init_db(db_path) -> sqlite3.Connection:
    """
    Intitialize temporary reference database.

    Returns
    -------
    conn : sqlite3.Connection
        Connection to the newly created db.
    """
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reference_faces
        (
            name text,
            image image,
            embedding array
        )
        """)
    conn.commit()
    cursor.close()
    return conn


def select_main_face(detected_faces: List[Face]) -> Face:
    """
    If a reference image contains multiple faces,
    assume that largest face is the one,
    i.e. the face with the largest area.

    Parameters
    ----------
    detected_faces : list of `facematch.face.Face`s
        All the faces detected on the image.

    Returns
    -------
    main_face : `facematch.face.Face`
        Main face in the image
    """
    if len(detected_faces) == 1:
        return detected_faces[0]
    boxes = np.stack([face.box for face in detected_faces], axis=-1)
    # [x_left, y_top, x_right, y_bottom]
    areas = (boxes[0] - boxes[2]) * (boxes[1] - boxes[3])
    largest_idx = areas.argmax()
    return detected_faces[largest_idx]


def get_name_from_path(path: str) -> str:
    """
    Extract name from filename (assumes that files are named as `person_name.?.ext`)
    """
    _, filename = os.path.split(path)
    name = filename.lower()
    name = re.sub(r'_[0-9]{4}\..*', '', name)
    name = re.sub(r'\.?[0-9]*?\.[a-z]*$', '', name)
    name = name.replace('_', ' ')
    return name


def get_person_embedding(faces: Dict[int, Face], img_indexes: List[int],
                         encoder: FaceEncoder) -> np.ndarray:
    """
    Calculate person embedding based on (possibly)
    several images.

    Parameters
    ----------
    faces : list of `Face`s
        Face objects for which to compute embeddings.
    img_indexes : list of ints
        Indexes of images in `faces`
    encoder : FaceEncoder

    Returns
    -------
    embedding : np.ndarray
        Person's face embedding.
    """
    images = [faces[i].image for i in img_indexes]
    embeddings = encoder.calculate_embeddings(images, report_progress=False)
    embedding = average_embeddings(embeddings)
    return embedding


def average_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate normalized average of 2+ vectors.
    Vectors must be of the same length.
    """
    average = np.mean(embeddings, axis=0)
    average_norm = average / np.linalg.norm(average, axis=0, keepdims=True)
    return average_norm
