import os


def get_unique_name(path: str, name: str):
    # use name_1 etc
    files = os.listdir(path)
    i = 1
    while True:
        new_name = f"{name}_{i}"
        if new_name not in files:
            return os.path.join(path, new_name)
        i += 1
