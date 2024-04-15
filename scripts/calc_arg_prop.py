import json
import numpy
import matplotlib.pyplot as plt
import numpy as np


def open_json_file(filename):
    with open(filename) as f:
        data_json = json.load(f)
        return data_json


def create_json(pred_args, name_of_output):
    # Calc probability
    outputDict = {}
    list_sum = []
    for key in pred_args:
        sum_of_values = sum(pred_args[key].values())

        output_Temp = {}
        for k in pred_args[key]:
            # Add to dict for json
            output_Temp[k] = pred_args[key][k] / sum_of_values
        list_sum.append(sum_of_values)

        outputDict[key] = dict(sorted(output_Temp.items(), key=lambda item: item[1], reverse=True))

    print(f"Total sum: {sum(list_sum)}")
    print(f"List of sum: {list_sum}")
    print(f"mean: {numpy.mean(list_sum)}")
    print(f"median: {numpy.median(list_sum)}")
    x = numpy.zeros(len(list_sum))
    plt.boxplot(list_sum)
    plt.show()

    # get 80 % of the most used features
    outDict = {}
    for key in outputDict:
        commutative_sum = 0
        output_Temp = []
        for k in outputDict[key]:
            if commutative_sum <= 0.8:
                output_Temp.append(k)
            commutative_sum += outputDict[key][k]
        outDict[key] = output_Temp

    jsonString = json.dumps(outDict)
    jsonFile = open(name_of_output, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def create_core_roles(pred_args, name_of_output):
    # Calc probability
    prepDict = {}
    outputDict = {}
    list_sum = []
    for key in pred_args:
        output_Temp = {}
        for k in pred_args[key]:
            # check if there are more than 2 values and if the second is not an arg
            arg_list = k.split(' ')
            if len(arg_list) == 1:
                #if "ARG" in k:
                output_Temp[k] = pred_args[key][k]
            elif len(arg_list) == 2:
                if "ARG" not in arg_list[1]:
                    if arg_list[0] in output_Temp:
                        output_Temp[arg_list[0]] = output_Temp[arg_list[0]] + pred_args[key][k]
                    else:
                        output_Temp[arg_list[0]] = pred_args[key][k]
                else:
                    output_Temp[k] = pred_args[key][k]
            else:
                if "ARG" not in arg_list[1]:
                    if arg_list[0] in output_Temp:
                        output_Temp[arg_list[0]] = output_Temp[arg_list[0]] + pred_args[key][k]
                    else:
                        output_Temp[arg_list[0]] = pred_args[key][k]
                else:
                    temp_key = arg_list[0] + ' ' + arg_list[1]
                    if temp_key in output_Temp:
                        output_Temp[temp_key] = output_Temp[temp_key] + pred_args[key][k]
                    else:
                        output_Temp[temp_key] = pred_args[key][k]


        outputDict[key] = dict(sorted(output_Temp.items(), key=lambda item: item[1], reverse=True))

    create_json(outputDict, name_of_output)

create_core_roles(open_json_file('data/pred_args_fine.json'), 'data/pred_args_core.json')
