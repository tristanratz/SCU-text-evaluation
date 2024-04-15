import json

def get_average_sentences(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    total_sentences = 0
    total_records = 0
    
    for record in data:
        if 'summary' in record:
            summary = record['summary']
            sentences = summary.split('.')  # Assuming sentences are separated by periods
            
            total_sentences += len(sentences)
            total_records += 1
    
    if total_records == 0:
        return 0  # Avoid division by zero
    
    average_sentences = total_sentences / total_records
    return average_sentences

averagepyrx = get_average_sentences('eval_interface/src/data/pyrxsum/pyrxsum-scus.json')
print(f"Average number of PYRXsum sentences: {averagepyrx}")

averagereal = get_average_sentences('eval_interface/src/data/realsumm/realsumm-scus.json')
print(f"Average number of Realsumm sentences: {averagereal}")

averagetac = get_average_sentences('eval_interface/src/data/tac09/tac09-scus.json')
print(f"Average number of TAC09 sentences: {averagetac}")

def get_average_words(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    total_words = 0
    total_records = 0

    for record in data:
        if 'summary' in record:
            summary = record['summary']
            words = summary.split()  # Splitting by whitespace to count words

            total_words += len(words)
            total_records += 1

    if total_records == 0:
        return 0  # Avoid division by zero

    average_words = total_words / total_records
    return average_words

averagewpyrx = get_average_words('eval_interface/src/data/pyrxsum/pyrxsum-scus.json')
print(f"Average number of pyrxsum words: {averagewpyrx}")
averagewreal = get_average_words('eval_interface/src/data/realsumm/realsumm-scus.json')
print(f"Average number of realsumm words: {averagewreal}")
averagewtac = get_average_words('eval_interface/src/data/tac09/tac09-scus.json')
print(f"Average number of tac09 words: {averagewtac}")

print(f"Ratio of pyrxsum: {averagewpyrx/averagepyrx}")
print(f"Ratio of realsumm: {averagewreal/averagereal}")
print(f"Ratio of tac09: {averagewtac/averagetac}")

print()

def get_average_scus(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    total_scus = 0
    total_records = 0
    
    for record in data:
        if 'scus' in record:
            scus = record['scus']
            
            total_scus += len(scus)
            total_records += 1
    
    if total_records == 0:
        return 0  # Avoid division by zero
    
    average_scus = total_scus / total_records
    return average_scus

averagescupyrx = get_average_scus('eval_interface/src/data/pyrxsum/pyrxsum-scus.json')
print(f"Average number of pyrxsum scus: {averagescupyrx}")
averagescureal = get_average_scus('eval_interface/src/data/realsumm/realsumm-scus.json')
print(f"Average number of realsumm scus: {averagescureal}")
averagescutac = get_average_scus('eval_interface/src/data/tac09/tac09-scus.json')
print(f"Average number of tac09 scus: {averagescutac}")