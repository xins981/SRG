import os

def list_files_with_suffix(directory, suffix):
    file_list = [file.replace(suffix, '') for file in os.listdir(directory) if file.endswith(suffix)]
    return file_list

directory_path = 'resources/objects/ycb'
suffix = '.png'

file_list = list_files_with_suffix(directory_path, suffix)
print(len(file_list))
print(file_list)