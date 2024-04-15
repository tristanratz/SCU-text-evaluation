import os
import json


def load_files_into_dict(folder_path):
    files_dict = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, encoding="ISO-8859-1") as f:
                file_content = f.read()
                files_dict[filename] = file_content
    return files_dict


def create_json(data_dict, filename):
    # create a new dictionary to store modified keys and values
    new_dict = {}
    for key, value in data_dict.items():
        # extract instance id from first 7 digits of key
        instance_id = key[:7]
        # extract data after last '.' in key as key in new_dict
        new_key = key.split('.')[-1]
        # check if instance_id already in new_dict
        if int(new_key) < 58:
            if instance_id in new_dict:
                # add value to existing instance_id key
                new_dict[instance_id][new_key] = value
            else:
                # create new instance_id key with value
                new_dict[instance_id] = {new_key: value}
    output_array = []
    for key, value in new_dict.items():
        temp_dict = {"instance_id": key}
        for k, v in value.items():
            temp_dict[k] = v
        output_array.append(temp_dict)

    # save new_dict as JSON file
    with open(filename, 'w') as outfile:
        json.dump(output_array, outfile, indent=4)


def file_to_json(list_of_filepath, output_file):
    data = {}
    if len(list_of_filepath) == 2:
        split_char = '\t'
    else:
        split_char = " "

    for filepath in list_of_filepath:
        with open(filepath, 'r') as f:
            counter = 0
            for line in f:
                line_parts = line.strip().split(split_char)
                key = line_parts[1]
                num = line_parts[0].split("-")
                if num[1] in "B":
                    inner_key = ((int(num[0][-2:]) - 1) * 2) + 1
                else:
                    inner_key = (int(num[0][-2:]) - 1) * 2
                inner_value = float(line_parts[2])
                if key not in data:
                    data[key] = {}
                data[key][inner_key] = float(inner_value)
                counter += 1
    with open(output_file, 'w') as f:
        json.dump(data, f)


tac08_system_summarys = load_files_into_dict("data/tac08/UpdateSumm08_eval/ROUGE/peers")
# create_json(tac08_system_summarys, "eval_interface/src/data/tac08/tac08-system-summary.json")

file_to_json(["data/tac08/UpdateSumm08_eval/manual/manual.peer"],
             "eval_interface/src/data/tac08/tac08-golden-labels.json")
# print(tac08_system_summarys.keys())


tac08_system_summarys = load_files_into_dict("data/tac09/UpdateSumm09_eval/ROUGE/peers")
# create_json(tac08_system_summarys, "eval_interface/src/data/tac09/tac09-system-summary.json")

file_to_json(["data/tac09/UpdateSumm09_eval/manual/manual.peer.A", "data/tac09/UpdateSumm09_eval/manual/manual.peer.B"],
             "eval_interface/src/data/tac09/tac09-golden-labels.json")
