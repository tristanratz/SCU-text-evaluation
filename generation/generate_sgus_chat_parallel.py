import openai
import json
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import traceback

ft_model = 'gpt-4' 
# ft_model = 'gpt-4' 
# 'davinci:ft-personal-2023-02-08-09-42-43' # ft-yWG8yb5cL8E1du9igUeaFk4O
data_path = "../../data"

def get_samplefile_path(ds, fn = None):
    print('Load sample data', ds,'...')
    return f"{data_path}" + ds + ".jsonl"

def get_path(ds, fn = None):
    print('Load data', ds,'...')
    return f"{data_path}/" + ds + "/" + (fn if fn != None else ds + "-scus") + ".json"

def get_save_path(ds, fn = None):
    print('Save data', ds,'...')
    return f"{data_path}/" + ds + "/" + (fn if fn != None else ds + f"-sgus-{ft_model}") + ".json"


def generate(ds, sample_file=None, nr_samples=1, save_name=None):
    fn = get_path(ds)
    with open(fn) as f:
        dataset = []
        print("Opening", fn)
        d = json.load(f)
        samples = []
        if sample_file is not None:
            with open(get_samplefile_path(sample_file), 'r') as sf:
                for sam in sf:
                    samples.append(json.loads(sam))
        
        print("Generating...")

        with mp.Pool(mp.cpu_count()) as pool:
            dataset = [pool.apply(generate_sgu, args=(s, idx, samples, nr_samples)) for idx, s in enumerate(d)]

        dump = {
            "generation_date": datetime.today().strftime('%d-%m-%Y'),
            "temperature": 0,
            "nr_samples": nr_samples,
            "model": ft_model,
            "samples_used": [s for s in samples[0:nr_samples]],
            "data": dataset
        }
        json.dump(dump, open(get_save_path(ds, save_name), "w"))


def generate_sgu(s, idx, samples, nr_samples):
    try:
        print("Processing sample NR.", idx, "-", " ".join(s["summary"].split(" ")[0:5]), "...")
        sample = { "summary": s["summary"], "instance_id": s["instance_id"] }
        prompt = s["summary"].replace('<t> ','').replace(' </t>','').replace(" . ", ". ")
        messages=[
            {"role": "system", "content": "You split the provided input in small sentences separated by an #. The split sentences represent subsentences of the original sentence."},
        ]

        for samp in samples[0:nr_samples]:
            messages.append({"role": "user", "content": samp["prompt"]})
            messages.append({"role": "assistant", "content": samp["completion"]})

        messages.append(
            {"role": "user", "content": prompt}
        )

        res = openai.ChatCompletion.create(
            model=ft_model, 
            temperature=0,
            messages=messages
        )
        sample['sgus'] = res['choices'][0]['message']["content"].replace(" END\n", "").lstrip().replace(" . ", ". ").split(" # ")
        sample['sample_nr'] = idx
        return sample
    except Exception as error:
        print("An error occurred:")
        print(error)
        traceback.print_exc()
        return generate_sgu(s, idx, samples, nr_samples)

if __name__ == '__main__':
    for dataset in ["tac09", "realsumm", "pyrxsum"]:
        generate(dataset, sample_file="gpt_training_chat_full", nr_samples=1, save_name=dataset + "_" + ft_model + "_ctx_oneshot")
