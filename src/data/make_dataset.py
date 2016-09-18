import subprocess
import os

from src.features import build_features


def main():
    build_features()
    R_script = os.path.join(os.path.dirname(__file__), '..', 'features',
                            'merge_features.R')
    subprocess.Popen("RScript " + R_script, shell=True)


if __name__ == '__main__':
    main()
