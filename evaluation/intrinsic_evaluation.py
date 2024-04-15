from summ_eval.rouge_metric import RougeMetric
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.mover_score_metric import MoverScoreMetric
import bert_score
import torch
import json
from transformers import logging
from itertools import islice
import numpy

logging.set_verbosity_error()

data_dir = '{data_dir}'
device = "cuda" if torch.cuda.is_available() else "cpu"

def open_json_file(filename):
    with open(filename) as f:
        data_json = json.load(f)
        return data_json


def open_jsonl_file(filename):
    with open(filename) as f:
        data_json = [json.loads(jline) for jline in f.read().splitlines()]
        return data_json


def simple_evaluation(scus, sxus):
    rouge = RougeMetric()
    # print(sxus[0])
    if sxus[0] is None:
        return 0
    return rouge.evaluate_batch(scus, sxus, False)['rouge']['rouge_1_f_score']


def evaluation_Bert_one(scus, sxus):

    scu_for_eval = []
    sxu_for_eval = []
    for c in scus:
        for t in sxus:
            scu_for_eval.append(c)
            sxu_for_eval.append(t)

    # evaluate bert score
    all_preds, _ = bert_score.score(scu_for_eval, sxu_for_eval, lang='en', model_type='bert-base-uncased',
                                            num_layers=8, verbose=False, idf=False, \
                                            nthreads=4, batch_size=64, rescale_with_baseline=False, return_hash=True,
                                            device=torch.device(device))
    length_to_split = []
    for i in range(len(scus)):
        length_to_split.append(len(sxus))
    sxu_accs = []
    sxu_pos = []
    Inputt = iter(all_preds[2].tolist())
    Output = [list(islice(Inputt, elem))
              for elem in length_to_split]
    for i in range(len(scus)):
        temp_max = max(Output[i])
        sxu_accs.append(temp_max)
        sxu_pos.append([int(numpy.argmax(Output[i])), temp_max])

    acc_sxu = (1 / len(sxu_accs)) * sum(sxu_accs)
    
    return acc_sxu, sxu_pos


def evaluation_Bert_two(scus, stus, smus):

    scu_for_eval = []
    sxu_for_eval = []
    for c in scus:
        for t in stus:
            scu_for_eval.append(c)
            sxu_for_eval.append(t)
        for m in smus:
            scu_for_eval.append(c)
            sxu_for_eval.append(m)
    # evaluate bert score
    all_preds, hash_code = bert_score.score(scu_for_eval, sxu_for_eval, lang='en', model_type='bert-base-uncased',
                                            num_layers=8, verbose=False, idf=False, \
                                            nthreads=4, batch_size=64, rescale_with_baseline=False, return_hash=True,
                                            device=torch.device(device))
    length_to_split = []
    for i in range(len(scus)):
        length_to_split.append(len(stus))
        length_to_split.append(len(smus))
    stu_accs = []
    stu_pos = []
    smu_accs = []
    smu_pos = []
    Inputt = iter(all_preds[2].tolist())
    Output = [list(islice(Inputt, elem))
              for elem in length_to_split]
    for i in range(len(scus) * 2):
        if i % 2 == 0:
            temp_max = max(Output[i])
            stu_accs.append(temp_max)
            stu_pos.append([int(numpy.argmax(Output[i])), temp_max])
        else:
            temp_max = max(Output[i])
            smu_accs.append(temp_max)
            smu_pos.append([int(numpy.argmax(Output[i])), temp_max])

    acc_stu = (1 / len(stu_accs)) * sum(stu_accs)
    acc_smu = (1 / len(smu_accs)) * sum(smu_accs)

    return acc_stu, acc_smu, stu_pos, smu_pos 


def simple_evaluation_Mover(scus, sxus):
    mover = MoverScoreMetric()

    if sxus[0] is None:
        return 0
    return mover.evaluate_batch(scus, sxus)['mover_score']


def easiness_sent_evaluation(scus, sxus):
    rouge = RougeMetric()
    if sxus[0] is None:
        return 0
    # Easiness_sent
    # Get acc of every scu
    list_of_acc = []
    list_of_pos = []
    for scu in scus:
        r1_f1_score = []
        for sxu in sxus:
            
            if sxu != "&" and sxu != "$":
                r1_f1_score.append(rouge.evaluate_example(scu, sxu)['rouge']['rouge_1_f_score'])
        temp_max = max(r1_f1_score)
        list_of_acc.append(temp_max)
        list_of_pos.append([int(numpy.argmax(r1_f1_score)), temp_max])

    # calculate average of acc list
    acc_sxu = (1 / len(list_of_acc)) * sum(list_of_acc)
    return acc_sxu, list_of_pos


def easiness_sent_evaluation_Mover(scus, sxus):
    mover = MoverScoreMetric()
    if sxus[0] is None:
        return 0
    # Easiness_sent
    # Get acc of every scu
    list_of_acc = []
    for scu in scus:
        score = []
        for sxu in sxus:
            score.append(mover.evaluate_example(scu, sxu)['mover_score'])
        list_of_acc.append(max(score))
    # calculate average of acc list
    acc_sxu = (1 / len(list_of_acc)) * sum(list_of_acc)
    return acc_sxu


def evaluate_summaries(scus, stus, smus, output_file, rouge, bert, mover):
    # rouge = RougeMetric()

    summaries = ["This is one summary", "This is another summary"]
    references = ["This is one reference", "This is another"]

    outputDict = []

    for i, scu in enumerate(scus):  # [21:22]):
        #i = 33
        print(scu['instance_id'])
        output_Temp = {'instance_id': scu['instance_id']}
        # Evaluate
        if rouge:
            # Easiness_sent
            stus_evaluation_rouge, stus_pos_rouge = easiness_sent_evaluation(scu['scus'], stus[i]['stus'])
            smus_evaluation_rouge, smus_pos_rouge = easiness_sent_evaluation(scu['scus'], smus[i]['smus'])

            # Add to dict for json
            output_Temp['easiness-stus-acc-rouge'] = stus_evaluation_rouge
            output_Temp['easiness-smus-acc-rouge'] = smus_evaluation_rouge
            output_Temp['stus-pos-rouge'] = stus_pos_rouge
            output_Temp['smus-pos-rouge'] = smus_pos_rouge
            print("ROUGE done!")
        if bert:
            # Easiness_sent
            if stus[i]['stus'][0] is None:
                smus_evaluation_bert = evaluation_Bert_one(scu['scus'], smus[i]['smus'])
                stus_evaluation_bert = 0
            elif smus[i]['smus'][0] is None:
                stus_evaluation_bert = evaluation_Bert_one(scu['scus'], stus[i]['stus'])
                smus_evaluation_bert = 0
            else:
                stus_evaluation_bert, smus_evaluation_bert, stus_pos_bert, smus_pos_bert = evaluation_Bert_two(scu['scus'], stus[i]['stus'],
                                                                                 smus[i]['smus'])

            # Add to dict for json
            output_Temp['easiness-stus-acc-bert'] = stus_evaluation_bert
            output_Temp['easiness-smus-acc-bert'] = smus_evaluation_bert
            output_Temp['stus-pos-bert'] = stus_pos_bert
            output_Temp['smus-pos-bert'] = smus_pos_bert
            print("BERT done!")

        if mover:
            # Simple evaluation by a collection of sentences
            stus_evaluation_mover_total = simple_evaluation_Mover(scu['scus'], stus[i]['stus'])
            smus_evaluation_mover_total = simple_evaluation_Mover(scu['scus'], smus[i]['smus'])

            # Easiness_sent
            stus_evaluation_mover = easiness_sent_evaluation_Mover(scu['scus'], stus[i]['stus'])
            smus_evaluation_mover = easiness_sent_evaluation_Mover(scu['scus'], smus[i]['smus'])

            # Add to dict for json
            output_Temp['batch-stus-acc-mover'] = stus_evaluation_mover_total
            output_Temp['batch-smus-acc-mover'] = smus_evaluation_mover_total
            output_Temp['easiness-stus-acc-mover'] = stus_evaluation_mover
            output_Temp['easiness-smus-acc-mover'] = smus_evaluation_mover
            print("MOVER done!")

        # Save to json file
        outputDict.append(output_Temp)

    jsonString = json.dumps(outputDict)
    jsonFile = open(output_file, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


# PyrXSum dataset
def evaluate_pyrxsum(rouge=True, bert=False, mover=False):
    smus = open_json_file(f'{data_dir}pyrxsum/pyrxsum-smus-sg4-plus-v10.json')
    stus = open_json_file(f'{data_dir}pyrxsum/pyrxsum-stus.json')
    scus = open_json_file(f'{data_dir}pyrxsum/pyrxsum-scus.json')

    evaluate_summaries(scus, stus, smus, '{data_dir}pyrxsum/pyrxsum-acc-sg4-plus-v10.json', rouge, bert, mover)
    print("PyrXSum done!")


# REALSumm dataset !!! stu realsumm-70 has "." and smu realsumm-69 has "iii","****" and realsumm-97 has "most."
def evaluate_realsumm(rouge=True, bert=False, mover=False):
    smus = open_json_file(f'{data_dir}realsumm/realsumm-smus.json')
    stus = open_json_file(f'{data_dir}realsumm/realsumm-stus.json')
    scus = open_json_file(f'{data_dir}realsumm/realsumm-scus.json')

    evaluate_summaries(scus, stus, smus, '{data_dir}realsumm/realsumm-acc.json', rouge, bert, mover)
    print("REALSumm done!")


# Tac08 dataset
def evaluate_tac08(rouge=True, bert=False, mover=False):
    smus = open_json_file(f'{data_dir}tac08/tac08-smus-sg4-plus-v10.json')
    stus = open_json_file(f'{data_dir}tac08/tac08-stus.json')
    scus = open_json_file(f'{data_dir}tac08/tac08-scus.json')

    evaluate_summaries(scus, stus, smus, '{data_dir}tac08/tac08-acc.json', rouge, bert, mover)
    print("Tac08 done!")


# Tac09 dataset !!! stu d0913-A has "?" devided by 0 error
def evaluate_tac09(rouge=True, bert=False, mover=False):
    smus = open_json_file(f'{data_dir}tac09/tac09-smus-sg4-plus-v10.json')
    stus = open_json_file(f'{data_dir}tac09/tac09-stus.json')
    scus = open_json_file(f'{data_dir}tac09/tac09-scus.json')

    evaluate_summaries(scus, stus, smus, '{data_dir}tac09/tac09-acc.json', rouge, bert, mover)
    print("Tac09 done!")

def evaluate_data(smus, stus, scus, result_path, rouge=True, bert=False, mover=False):

    evaluate_summaries(scus, stus, smus, result_path, rouge, bert, mover)
    print(f"File {result_path} done!")



def debug():
    # debug realsumm
    bert = BertScoreMetric()

    scus = open_json_file(f'{data_dir}realsumm/realsumm-scus.json')
    stus = open_json_file(f'{data_dir}realsumm/realsumm-stus.json')

    scu = scus[0]['scus']
    stu = stus[0]['stus']

    print(scu[2])
    print(stu[3])

    stus_evaluation_bert = bert.evaluate_example(scu[2], stu[3])['bert_score_f1']
    print(stus_evaluation_bert)


if __name__ == '__main__':
    evaluate_pyrxsum(True, True, False)
    evaluate_realsumm(True, True, False)
    evaluate_tac08(True, True, False)
    evaluate_tac09(True, True, False)
