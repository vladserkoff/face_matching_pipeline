"""
Prepare LFW dataset
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
from typing import Tuple
from urllib.request import urlretrieve

from secondface.reference import construct_references

LFW_URL = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
LFW_MD5 = 'a17d05bd522c52d84eca14327a23d494'
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

DEFAULT_LFW_PATH = './data/lfw'
DEFAULT_DB_PATH = os.path.join(DEFAULT_LFW_PATH, 'reference.sqlite')


def fetch_remote_lfw() -> str:
    """
    Downloads the archive.

    Returns
    -------
    filepath : str
        Path to the downloaded archive.
    """
    filepath, _ = urlretrieve(LFW_URL)
    if not check_md5_checksum(filepath, LFW_MD5):
        raise IOError("{} has an md5 checksum differing from expected, "
                      "file may be corrupted.".format(LFW_URL))
    return filepath


def check_md5_checksum(filepath: str, expected_md5: str) -> bool:
    """
    Checks if file md5 checksum matches with expexted sum.

    Parameters
    ----------
    filepath : str
        Path to a file that needs to be checked.
    expected : str
        Expected md5 checksum of the file

    Returns
    -------
    is_ok : bool
        Weather the checksum matched.
    """
    md5 = hashlib.md5(open(filepath, 'rb').read())
    is_ok = md5.hexdigest() == expected_md5
    return is_ok


def prepare_lfw_data(lfw_dir: str) -> Tuple[str, str]:
    """
    Download lfw dataset and prepare it for the matching pipeline.

    Parameters
    ----------
    lfw_dir : str
        Destination directory for lfw dataset.

    Returns
    -------
    (references_dir, candidates_dir) : tuple
        Paths to reference and candidate images.

    Notes
    -----
    If there already exists `lfw_dir`/lfw directory, its' contents
    would be used instead of downloading from the internet.
    """
    raw_dir = os.path.join(lfw_dir, 'lfw')

    if not os.path.exists(raw_dir):
        path = fetch_remote_lfw()
        lfw_dir = os.path.abspath(lfw_dir)
        with tarfile.open(path) as archive:
            archive.extractall(lfw_dir)
        os.remove(path)

    references_dir = os.path.join(lfw_dir, 'references')
    os.makedirs(references_dir)
    candidates_dir = os.path.join(lfw_dir, 'candidates')
    os.makedirs(candidates_dir)

    images = [
        os.path.join(directory, file)
        for directory, _, files in os.walk(raw_dir) for file in files
        if os.path.splitext(file)[1].lower() in IMG_EXTENSIONS
    ]
    for img in images:
        if '0001' in img:
            shutil.copy(img, references_dir)
        else:
            shutil.copy(img, candidates_dir)

    return references_dir, candidates_dir


def make_reference_database(lfw_dir: str = DEFAULT_LFW_PATH,
                            db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Make LFW reference database

    Parameters
    ----------
    lfw_dir : str, default './data/lfw'
        Destination directory for lfw dataset.
        If it's empty, dataset is downloaded from the internet.
    db_path : str, default './data/lfw/reference.sqlite'
        Path to the resulting database.

    Returns
    -------
    candidates_dir : str
        Path to candidate images.
    """
    references_dir, candidates_dir = prepare_lfw_data(lfw_dir)
    connection = construct_references(db_path, references_dir)
    connection.close()
    return candidates_dir


def main(args):
    """
    Make dataset
    """
    candidates_dir = make_reference_database(args.lfw_dir, args.db_path)
    msg = "Created reference database from LFW dataset at '{}'\n"
    msg += "You could test matching with images from '{}'"
    print(msg.format(args.db_path, candidates_dir))


def parse_arguments(argv):
    """
    Parse cli argments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lfw_dir',
        type=str,
        default=None,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument(
        '--db_path',
        type=str,
        default=None,
        help='Path to the data directory containing aligned LFW face patches.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
