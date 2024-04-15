import json


def open_jsonl_file(filename):
    with open(filename) as f:
        data_json = [json.loads(jline) for jline in f.read().splitlines()]
        return data_json


def create_stus_json_file(output_file_name, json_data):
    outputDict = []
    list_of_stus = []
    list_of_summary = ''

    for i, l in enumerate(json_data):
        # print(l['instance_id'])

        list_of_summary = list_of_summary + str(l['summary'])
        for stu in l['stus']:
            if stu.__contains__(" "):
                list_of_stus.append(stu)
            # print(stu)
        if (i + 1) % 4 == 0 and i != 0:
            # print(i)
            outputDict.append({
                'instance_id': l['instance_id'],
                'summary': list_of_summary,
                'stus': list_of_stus,
            })
            list_of_stus = []
            list_of_summary = ''

    jsonString = json.dumps(outputDict)
    jsonFile = open(output_file_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


# Tac2008 dataset
data_json = open_jsonl_file('../../eval_interface/src/data/tac08/tac2008.stus.coref_true.jsonl')
create_stus_json_file('../../eval_interface/src/data/tac08/tac2008-stus.json', data_json)

# Tac2009 dataset
data_json = open_jsonl_file('../../eval_interface/src/data/tac09/tac2009.stus.coref_true.jsonl')
create_stus_json_file('../../eval_interface/src/data/tac09/tac2009-stus.json', data_json)
