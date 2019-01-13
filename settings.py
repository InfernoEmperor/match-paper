from os.path import join, abspath, dirname

PROJ_DIR = abspath(dirname(__file__))
DATA_DIR = join(PROJ_DIR, 'data')
PAPER_DIR = join(DATA_DIR, 'papers')
TRAIN_DIR = join(PROJ_DIR, 'trained')

