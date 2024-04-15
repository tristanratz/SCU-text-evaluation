import json


def open_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines


def create_json(summary, name_of_sxus, sxus, name_of_output, name_of_instance):
    outputDict = []
    for i, l in enumerate(summary):
        list_of_sxus = sxus[i].replace('\n', '').split("\t")
        remove_list = [sxu for sxu in list_of_sxus if not sxu.__contains__(" ")]
        if remove_list:
            for r in remove_list:
                list_of_sxus.remove(r)
        outputDict.append(
            {
                'instance_id': name_of_instance + "-" + str(i),
                'summary': l.replace('\n', ''),#.replace('\"', '').replace('\u2019', '`').replace('\u00a3', 'Pound'),
                name_of_sxus: list_of_sxus, #.replace('\"', '').replace('\u2019', '`').replace('\u00a3', 'Pound').split("\t"),
            }
        )

    jsonString = json.dumps(outputDict)
    jsonFile = open(name_of_output, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

# Create Json out of source PyrXSum source data
summary = open_file('../../eval_interface/src/data/PyrXSum(Source)/references.txt')
stus = open_file('../../eval_interface/src/data/PyrXSum(Source)/STUs.txt')
scus = open_file('../../eval_interface/src/data/PyrXSum(Source)/SCUs.txt')
create_json(summary, 'stus', stus, 'eval_interface/src/data/pyrxsum/pyrxsum-stus.json', 'pyrxsum')
create_json(summary, 'scus', scus, 'eval_interface/src/data/pyrxsum/pyrxsum-scus.json', 'pyrxsum')

# Create Json out of source REALSumm data
summary = open_file('../../eval_interface/src/data/REALSumm(Source)/references.txt')
stus = open_file('../../eval_interface/src/data/REALSumm(Source)/STUs.txt')
scus = open_file('../../eval_interface/src/data/REALSumm(Source)/SCUs.txt')
create_json(summary, 'stus', stus, 'eval_interface/src/data/realsumm/realsumm-stus.json', 'realsumm')
create_json(summary, 'scus', scus, 'eval_interface/src/data/realsumm/realsumm-scus.json', 'realsumm')
