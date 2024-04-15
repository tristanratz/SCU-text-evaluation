import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm


FULL = True

def open_jsonl_file(filename):
    with open(filename) as f:
        data_json = [json.loads(jline) for jline in f.read().splitlines()]
        return data_json


def create_scu_json_file(output_file_name, json_data):
    outputDict = []
    tf = TfidfVectorizer(analyzer='word',stop_words= 'english')

    scu_count = 0
    avg_sim = 0
    low_sim = 1
    max_sim = 0

    for l in json_data:
        print("Sample", l['instance_id'])
        # print(l['instance_id'])
        list_of_scus = []
        summaries = []


        for summary in l['summaries']:
            summaries.append(summary)
            list_of_scus.append([]);

        # Train vectorizer on the given words
        # No additional/new vocab will be in the scus
        tf.fit(summaries)

        for scu in l['scus']:
            label = scu['label']
            lvec = tf.transform([label])

            scu_count = scu_count + 1

            maxInd = 0 # The ID of the summary to which it should be added
            maxSim = 0 # The Similarity value of the most important contributor
            maxLab = 0 # Label

            # Problem: To many calculators "contribute" to a given label
            # Though: Most times only one label really contributes
            # Solution: Take most likely contributor, look to which summary it belongs and add it to this summary
            for contr in scu["contributors"]:
                clabel = contr["label"]
                
                # Full will assign the SCU to every Summary that contributed
                if FULL:
                    list_of_scus[contr['summary_index']].append(label)
                else:
                    cvec = tf.transform([clabel])

                    # Calc the similarity between the general label and the contributor
                    cosine = cosine_similarity(lvec, cvec, dense_output=True) 

                    # Choose the summary which is most contributes most
                    if maxSim < cosine and cosine != 0:
                        maxSim = cosine
                        maxInd = contr["summary_index"]
                        maxLab = contr["label"]

            if not FULL:
                avg_sim = avg_sim + maxSim

                if max_sim < maxSim:
                    max_sim = maxSim
                if low_sim > maxSim:
                    low_sim = maxSim
                    print(label, maxSim, maxLab)

                list_of_scus[maxInd].append(label)

        for ind, summary in enumerate(summaries):
            outputDict.append({
                'instance_id': str(l['instance_id'] + "-" + str(ind)),
                'summary': summary,
                'scus': list_of_scus[ind],
            })

    jsonString = json.dumps(outputDict)
    jsonFile = open(output_file_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print("Average sim:", avg_sim/scu_count)
    print("Low sim:", low_sim)
    print("Max sim:", max_sim)


# Tac2008 dataset
data_json = open_jsonl_file("../../eval_interface/src/data/tac08/tac2008.pyramids.raw.jsonl")
create_scu_json_file("../../eval_interface/src/data/tac08/tac2008-scus-sp-full.json", data_json)

# Tac2009 dataset
data_json = open_jsonl_file("../../eval_interface/src/data/tac09/tac2009.pyramids.raw.jsonl")
create_scu_json_file("../../eval_interface/src/data/tac09/tac2009-scus-sp-full.json", data_json)
