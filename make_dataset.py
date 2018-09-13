"""
Prepare custom dataset
"""

import argparse
import sys

from facematch.reference import construct_references


def make_reference_database(reference_dir: str, db_path: str) -> None:
    """
    Make custom reference database

    Parameters
    ----------
    reference_dir : str
        Directory with face images for reference database.
    db_path : str
        Path to the resulting database.
    """
    connection = construct_references(db_path, reference_dir)
    connection.close()


def main(args) -> None:
    """
    Make dataset
    """
    make_reference_database(args.reference_dir, args.db_path)
    msg = "Created reference database at '{}'".format(args.db_path)
    print(msg)


def parse_arguments(argv):
    """
    Parse cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference_dir',
        type=str,
        help='Path to the directory with face images for reference database.')
    parser.add_argument(
        '--db_path', type=str, help='Path to the resulting database.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
