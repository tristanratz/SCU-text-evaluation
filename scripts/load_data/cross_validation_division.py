import os
import json

def txt_to_dict(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = {}
        for i, line in enumerate(lines):
            data[line.strip()] = i
    return data


def process_txt_files(folder_path):
    counter = 0
    id_dict = txt_to_dict('../../Lite2_3Pyramid/data/REALSumm/ids.txt')
    file_dict = {}
    for i in range(1, 6):
        file_path = os.path.join(folder_path, f"fold{i}.id")
        with open(file_path, "r") as f:
            file_content = f.readlines()
            file_content = [line.strip() for line in file_content]
            ids = []
            for value in file_content:
                ids.append(id_dict[value])
            file_dict[i] = ids

    with open("../../eval_interface/src/data/realsumm/fold_split.json", "w") as json_file:
        json.dump(file_dict, json_file)


process_txt_files('../../Lite2_3Pyramid/data/REALSumm/by_examples')
