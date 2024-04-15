import json
import os
import numpy as np


def open_all_files(folder, file_endings):
    # The folder containing the text files
    # folder = '/path/to/folder'

    # The list to store the contents of the text files
    text_files = []
    file_names = []

    # List the files in the folder
    for file in os.listdir(folder):
        # Check if the file is a text file
        if file.endswith(file_endings):
            # Store the file name
            file_names.append(file)
            # The list to store the lines of the text file
            lines = []
            # Open the file and read the contents
            with open(os.path.join(folder, file), 'r') as f:
                for line in f:
                    lines.append(line)
            # Append the list of lines to the list of text files
            text_files.append(lines)

    return file_names, text_files  # A list containing the contents of the text files


def create_gold_evaluation(file_names, summarys, name_of_output):
    outputDict = {}
    for i, summary in enumerate(summarys):
        output_Temp = []
        for j, values in enumerate(summary):
            # Add to dict for json
            label_scu = str(values).replace('\n', '')
            label_scu_array = label_scu.split('\t')

            result = list(map(int, map(lambda x: x.strip('"'), label_scu_array)))

            output_Temp.append(np.mean(result))
        outputDict[file_names[i].replace('.label', '')] = dict(
            zip(range(len(output_Temp)), map(lambda x: x, output_Temp)))

    jsonString = json.dumps(outputDict)
    jsonFile = open(name_of_output, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


# PyrXSum dataset
def gold_evaluate_pyrxsum():
    file_names, list_of_labels = open_all_files('../../data/PyrXSum(Source)/labels', '.label')
    create_gold_evaluation(file_names, list_of_labels,
                           '../../eval_interface/src/data/pyrxsum/pyrxsum-golden-labels.json', )
    print("PyrXSum done!")


# REALSumm dataset !!! stu realsumm-70 has "." and smu realsumm-69 has "iii","****" and realsumm-97 has "most."
def gold_evaluate_realsumm():
    file_names, list_of_labels = open_all_files('../../data/REALSumm(Source)/labels', '.label')
    create_gold_evaluation(file_names, list_of_labels,
                           '../../eval_interface/src/data/realsumm/realsumm-golden-labels.json', )
    print("REALSumm done!")


# Tac2008 dataset
def gold_evaluate_tac08():
    # nli_evaluation_dataset(smus, 'eval_interface/src/data/tac08/tac08-nli.json')
    print("Tac2008 done!")


# Tac2009 dataset !!! stu d0913-A has "?" devided by 0 error
def gold_evaluate_tac09():
    # nli_evaluation_dataset(smus, 'eval_interface/src/data/tac09/tac09-nli.json')
    print("Tac2009 done!")


def gold_evaluate_data(labels, result_path):
    gold_evaluation_dataset(labels, result_path)
    print(f"nli evaluation of {result_path} done!")


gold_evaluate_pyrxsum()
gold_evaluate_realsumm()
# gold_evaluate_tac08()
# gold_evaluate_tac09()
