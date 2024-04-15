import json
import numpy as np
import torch

from Lite2_3Pyramid.reproduce.utils import system_level_correlation
from Lite2_3Pyramid.reproduce.utils import summary_level_correlation
from Lite2_3Pyramid.metric.score import score


experiment = "-stus"
sxu = "stu"

def open_json_file(filename, sxu_name=None):
    with open(filename) as f:
        data_json = json.load(f)
        if sxu_name == "sgu":
            return data_json["data"]
        return data_json


def calc_corr_summary_and_system(results, golden, cross_validation=False):
    if cross_validation:

        fold_division = open_json_file('eval_interface/src/data/realsumm/fold_split.json')
        system_pearson_array = []
        system_spearman_array = []
        for key_fold, value_fold in fold_division.items():
            results_temp = {}
            golden_temp = {}
            print(value_fold)
            for key, value in results.items():
                results_temp[key] = {k: value[k] for k in [str(n) for n in value_fold] if k in value}

            for key, value in golden.items():
                golden_temp[key] = {k: value[k] for k in [str(n) for n in value_fold] if k in value}

            system_pearson, system_spearman = system_level_correlation(golden_temp, results_temp)
            system_pearson_array.append(system_pearson)
            system_spearman_array.append(system_spearman)

        system_pearson = np.mean(system_pearson_array)
        system_spearman = np.mean(system_spearman_array)
    else:
        system_pearson, system_spearman = system_level_correlation(golden, results)

    summary_pearson, summary_spearman = summary_level_correlation(golden, results)

    return [system_pearson, system_spearman, summary_pearson, summary_spearman]


def nli_evaluation_from_paper(summarys, smus, model_type):
    if "D08" in summarys[0]['instance_id'] or "D09" in summarys[0]['instance_id']:
        sorted_smus = sorted(smus, key=lambda x: x['instance_id'])
        sorted_summaries = sorted(summarys, key=lambda x: x['instance_id'])
        all_sorted_summaries = [dict([('instance_id', d['instance_id'])] + sorted([(k, v) for k, v in d.items() if k != 'instance_id'])) for d in sorted_summaries]#[dict(sorted(d.items())) for d in sorted_summaries]
    else:
        all_sorted_summaries = summarys
        sorted_smus = smus
    outputDict = {}
    all_summarys = list(map(lambda x: [], range(len(all_sorted_summaries[0]) - 1)))
    all_sxus = []
    name_of_system = []
    for i, data in enumerate(all_sorted_summaries):
        for j, value in enumerate(data.items()):
            if "instance_id" in value[0]:
                continue
            if i == 0:
                name_of_system.append(value[0])
            all_summarys[j - 1].append(value[1])
        all_sxus.append(sorted_smus[i][sxu + 's'])

    for i in range(len(all_summarys)):
        print(f"System summary: {name_of_system[i]} ( {i + 1} / {len(all_summarys)} )")

        scores = score(all_summarys[i], all_sxus, detail=True, device=torch.device("mps"), model_type=model_type)['l3c']

        outputDict[name_of_system[i].replace('.summary', '')] = dict(
            zip([str(n) for n in range(len(scores[1]))], map(lambda x: x, scores[1])))

    return outputDict


def write_to_json(list_of_results, output_file):
    outputDict = []
    list_of_datasets = ['pyrxsum', 'realsumm', 'tac09'] # 'tac08',
    for i, result in enumerate(list_of_results):
        output_Temp = {'instance_id': list_of_datasets[i],
                       'pearson_system': result[0],
                       'spearman_system': result[1],
                       'pearson_summary': result[2],
                       'spearman_summary': result[3]
                       }
        outputDict.append(output_Temp)

    jsonString = json.dumps(outputDict)
    jsonFile = open(output_file, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_dict_to_json(outputDict, file_name):
    jsonString = json.dumps(outputDict)
    jsonFile = open(file_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def corr_evaluate_realsumm(load_data=False):
    print("REALSumm start!")

    if load_data:
        result_Dict = open_json_file('eval_interface/src/data/realsumm/realsumm-nli-score-smu' + experiment + '.json')
    else:
        result_Dict = nli_evaluate_data(open_json_file('eval_interface/src/data/realsumm/realsumm-system-summary.json'),
                                        open_json_file('eval_interface/src/data/realsumm/realsumm' + experiment + '.json', sxu), "shiyue/roberta-large-tac08")

        save_dict_to_json(result_Dict, 'eval_interface/src/data/realsumm/realsumm-nli-score-smu' + experiment + '.json')

    return calc_corr_summary_and_system(result_Dict,
                                        open_json_file(
                                            'eval_interface/src/data/realsumm/realsumm-golden-labels.json'))

def corr_evaluate_pyrxsum(load_data=False):
    print("PyrXSum start!")
    if load_data:
        result_Dict = open_json_file('eval_interface/src/data/pyrxsum/pyrxsum-nli-score-smu' + experiment + '.json')
    else:
        result_Dict = nli_evaluate_data(open_json_file('eval_interface/src/data/pyrxsum/pyrxsum-system-summary.json'),
                                        open_json_file('eval_interface/src/data/pyrxsum/pyrxsum' + experiment + '.json', sxu), "shiyue/roberta-large-tac08")

        save_dict_to_json(result_Dict, 'eval_interface/src/data/pyrxsum/pyrxsum-nli-score-smu' + experiment + '.json')

    return calc_corr_summary_and_system(result_Dict,
                                        open_json_file(
                                            'eval_interface/src/data/pyrxsum/pyrxsum-golden-labels.json'))

def corr_evaluate_tac08(load_data=False):
    print("tac08 start!")

    if load_data:
        result_Dict = open_json_file('eval_interface/src/data/tac08/tac08-nli-score-smu' + experiment + '.json')
    else:
        result_Dict = nli_evaluate_data(open_json_file('eval_interface/src/data/tac08/tac08-system-summary.json'),
                                        open_json_file('eval_interface/src/data/tac08/tac08' + experiment + '.json', sxu), "shiyue/roberta-large-tac09")

        save_dict_to_json(result_Dict, 'eval_interface/src/data/tac08/tac08-nli-score-smu' + experiment + '.json')

    return calc_corr_summary_and_system(result_Dict,
                                        open_json_file(
                                            'eval_interface/src/data/tac08/tac08-golden-labels.json'))

def corr_evaluate_tac09(load_data=False):
    print("tac09 start!")

    if load_data:
        result_Dict = open_json_file('eval_interface/src/data/tac09/tac09-nli-score-smu' + experiment + '.json')
    else:
        result_Dict = nli_evaluate_data(open_json_file('eval_interface/src/data/tac09/tac09-system-summary.json'),
                                        open_json_file('eval_interface/src/data/tac09/tac09' + experiment + '.json', sxu), "shiyue/roberta-large-tac08")

        save_dict_to_json(result_Dict, 'eval_interface/src/data/tac09/tac09-nli-score-smu' + experiment + '.json')

    return calc_corr_summary_and_system(result_Dict,
                                        open_json_file(
                                            'eval_interface/src/data/tac09/tac09-golden-labels.json'))



def nli_evaluate_data(summarys, smus, model_type):
    return nli_evaluation_from_paper(summarys, smus, model_type)


def corr_evaluation_datase():
    list_of_results = []

    list_of_results.append(corr_evaluate_pyrxsum())
    list_of_results.append(corr_evaluate_realsumm())
    list_of_results.append(corr_evaluate_tac08())
    list_of_results.append(corr_evaluate_tac09())

    write_to_json(list_of_results, 'data/extrinsic_evaluation' + experiment + '.json')


if __name__ == '__main__':
    corr_evaluation_datase()
