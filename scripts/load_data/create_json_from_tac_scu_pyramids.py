import json


def open_jsonl_file(filename):
    with open(filename) as f:
        data_json = [json.loads(jline) for jline in f.read().splitlines()]
        return data_json


def create_scu_json_file(output_file_name, json_data):
    outputDict = []

    for l in json_data:
        # print(l['instance_id'])
        list_of_scus = []
        summary_total = ''
        for scu in l['scus']:
            list_of_scus.append(scu['label'])
            # print(scu['label'])
        for summary in l['summaries']:
            summary_total = summary_total + summary

        outputDict.append({
            'instance_id': l['instance_id'],
            'summary': summary_total,
            'scus': list_of_scus,
        })

    jsonString = json.dumps(outputDict)
    jsonFile = open(output_file_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


# Tac2008 dataset
data_json = open_jsonl_file('../../eval_interface/src/data/tac08/tac2008.pyramids.jsonl')
create_scu_json_file('../../eval_interface/src/data/tac08/tac2008-scus.json', data_json)

# Tac2009 dataset
data_json = open_jsonl_file('../../eval_interface/src/data/tac09/tac2009.pyramids.jsonl')
create_scu_json_file('../../eval_interface/src/data/tac09/tac2009-scus.json', data_json)
