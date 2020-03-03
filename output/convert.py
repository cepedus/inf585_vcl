import os
from tqdm import tqdm

path = os.getcwd()
d = 4

list_of_files = {}
for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
    for filename in filenames:
        if filename.startswith('file') and filename.endswith('png'):
            # list_of_files[filename] = os.sep.join([dirpath, filename])/
            old_path = os.path.join(dirpath, filename)
            # print(old_path)
            num = filename.split("_")[1].split(".")[0]
            # print(num)
            nz = d - len(num)
            new_name = filename[:-3] + ".png"
            new_path = os.path.join(dirpath, new_name)
            os.rename(old_path, new_path)
            # print(os.path.join(dirpath, new_name))

            # exit()