import os

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory += os.sep if not directory[-1] == os.sep else ''
    return directory