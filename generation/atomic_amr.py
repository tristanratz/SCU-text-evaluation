import json
import csv
import multiprocessing

from tqdm import tqdm
import amrlib as amrlib
import penman
from penman.models import amr
from itertools import chain, combinations
import spacy
import re
from penman.graph import Graph

from amrlib.graph_processing.amr_fix import maybe_fix_unlinked_in_subgraph

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import numpy as np
import coreferee
import math

from Lite2_3Pyramid.metric.extract_stus import _get_and_replace_coref

spacy_o = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

nlp.add_pipe('coreferee')

data_path = '../data'

# Download and unzip models https://github.com/bjascob/amrlib-models
DIR_STOG_MODEL = 'model_parse_xfm_bart_large-v0_1_0'
DIR_GTOS_MODEL = 'model_generate_t5wtense-v0_1_0'
stog = amrlib.load_stog_model(DIR_STOG_MODEL)
gtos = amrlib.load_gtos_model(DIR_GTOS_MODEL)

hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
max_length = 256
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)


def sent_in_summary(summary, sentences, ranking=False):
    list_of_decisions = []
    for smu in sentences:
        tokenized_input_seq_pair = tokenizer.encode_plus(summary, smu,
                                                         max_length=max_length,
                                                         return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)

        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
        # Note:
        # "id2label": {
        #     "0": "entailment",
        #     "1": "neutral",
        #     "2": "contradiction"
        # },

        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

        if ranking:
            list_of_decisions.append(predicted_probability[0])
        else:
            list_of_decisions.append(predicted_probability[0] > 0.5)

    return list_of_decisions


def split_amr_meta(entry):
    meta_lines = []
    graph_lines = []
    for line in entry.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# ::'):
            meta_lines.append(line)
        elif line.startswith('#'):
            continue
        else:
            graph_lines.append(line)
    return meta_lines, graph_lines


def gstring_to_oneline(gstring):
    meta_lines, graph_lines = split_amr_meta(gstring)
    gstring = ' '.join(graph_lines)
    gstring = re.sub(' +', ' ', gstring)
    return gstring


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = iterable
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def read_csv(file):
    # Read CSV file
    with open(file) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]

        return data_read


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_source_docs(file_path, to_dict=False):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def path_var(path, node):
    var, branches = node
    for step in path[:-1]:
        var, branches = branches[step][1]
    return var


def get_subgraphs2(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)
    t = penman.configure(g)

    dict_variables = {}

    root_node = t.node[0]
    subgraphs = []
    paths_variables = {}
    for path, branch in t.walk():
        val_node = path_var(path, t.node)
        if val_node not in dict_variables:
            dict_variables[val_node] = branch
            continue


        role, target = branch

        if dict_variables[val_node] not in paths_variables:
            paths_variables[dict_variables[val_node]] = {}
            paths_variables[dict_variables[val_node]]['val_description'] = val_node
            paths_variables[dict_variables[val_node]]['triples'] = []
        paths_variables[dict_variables[val_node]]['triples'].append((role, target))

    for variables in paths_variables.keys():
        val_node = paths_variables[variables]['val_description']

        args_triples = []
        other_triples = []

        for arg in paths_variables[variables]['triples']:
            if 'ARG' in arg[0]:
                args_triples.append(arg)
            else:
                other_triples.append(arg)

        lists_args = []
        lists_args.append(args_triples)
        for arg in other_triples:
            lists_args.append(args_triples + [arg])

        for list_args in lists_args:

            filtered_list_args = []
            for arg in list_args:
                if 'purpose' in arg[0]:
                    continue
                filtered_list_args.append(arg)

            list_args = filtered_list_args

            count_args = 0
            check_arg2 = False
            for arg in list_args:
                if 'ARG' in arg[0]:
                    count_args += 1
                if 'ARG2' in arg[0]:
                    check_arg2 = True

            if list_args:
                if count_args > 1 or (check_arg2 and len(list_args) > 1):

                    linearized_graph = penman.format((val_node, [dict_variables[val_node], *list_args]))
                    subgraphs.append(linearized_graph)

    return subgraphs


def get_subgraphs3(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)

    list_of_t = []
    list_of_g_level_one = []

    base_node_tuple = g.triples[0]
    base_node = g.triples[0][0]
    for i, subtree in enumerate(g.triples):
        if i == 0:
            continue
        if subtree[0] == base_node or subtree[2] == base_node:
            list_of_t.append([base_node_tuple, subtree])
            list_of_g_level_one.append([subtree])
        else:
            list_of_t[-1].append(subtree)
            list_of_g_level_one[-1].append(subtree)

    # Split the trees of level one leaves
    for j, level_one_tree in enumerate(list_of_g_level_one):
        if len(level_one_tree) <= 3:
            continue
        base_node = level_one_tree[1][0]
        set_root = True
        for i, subtree in enumerate(level_one_tree):
            if i == 0:
                continue
            if (subtree[0] == base_node or subtree[2] == base_node) and set_root:
                set_root = False
                list_of_t.append([subtree])
            else:
                list_of_t[-1].append(subtree)

    # Split by every predicart
    temp_list_2 = []
    for i, node in enumerate(g.triples):
        if node[1] in ":instance" and "-0" in node[2]: 
            linked_notes = [node[0]]
            temp_list = []
            for j, inner_node in enumerate(g.triples[i:]):
                if inner_node[0] not in linked_notes:
                    break
                else:
                    if inner_node[1] not in ":instance":
                        linked_notes.append(node[2])
                    temp_list.append(inner_node)
            list_of_t.append(temp_list)
            temp_list_2.append(temp_list)

    for i, graph in enumerate(temp_list_2):
        base_node_tuple_temp = graph[0]
        base_node_temp = graph[0][0]
        for j, subtree in enumerate(graph):
            if j == 0:
                continue
            if subtree[0] == base_node_temp or subtree[2] == base_node_temp:
                list_of_t.append([base_node_tuple_temp, subtree])
            else:
                list_of_t[-1].append(subtree)

    # spliten after op, if it there are leaves of and
    # check and presence
    temp_graph_and = [[]]
    i = 0
    end_of_outer_loop = True
    while end_of_outer_loop:
        node = g.triples[i]
        for n in temp_graph_and:
            n.append(node)
        if node[2] == "and":
            end_of_inner_loop = True
            if i == 0:
                node_id_of_arg = ('XXX', 'XXX', 'XXX')
            else:
                node_id_of_arg = g.triples[i - 1]
            node_id_of_and = g.triples[i]
            subtree_graphs = []
            i += 1
            number_of_subtrees = len(temp_graph_and)
            while end_of_inner_loop:
                node = g.triples[i]
                if node[0] == node_id_of_and[0] and ":op" in node[1]:
                    for n in temp_graph_and:
                        tempp = n.copy()
                        tempp.append(node)
                        subtree_graphs.append(tempp)

                elif node[0] == node_id_of_arg[0] or i >= len(g.triples) - 1:
                    for j in range(number_of_subtrees):
                        subtree_graphs[-(j + 1)].append(node)
                    if i >= len(g.triples) - 1:
                        end_of_outer_loop = False
                    temp_graph_and = subtree_graphs
                    i -= 2
                    end_of_inner_loop = False


                else:
                    for j in range(number_of_subtrees):
                        subtree_graphs[-(j + 1)].append(node)
                i += 1

        if i >= len(g.triples) - 1:
            end_of_outer_loop = False
        else:
            i += 1
    if len(temp_graph_and) != 1:

        for graph in temp_graph_and:
            label_of_and = 'XXX'
            for i, node in enumerate(graph):
                if node[2] == "and" and i == 0:
                    label_of_and = graph[0][0]
                    graph.pop(0)
                    graph.pop(0)
                    new_label = graph[0][0]
                elif node[2] == "and":
                    label_of_and = node[0]
                    new_label = graph[i + 1][2]

                    if graph[i - 1][0] == label_of_and:
                        y = list(graph[i - 1])
                        y[0] = graph[i + 1][2]
                        graph[i - 1] = tuple(y)
                    else:
                        y = list(graph[i - 1])
                        y[2] = graph[i + 1][2]
                        graph[i - 1] = tuple(y)
                    graph.pop(i)
                    graph.pop(i)
                elif node[0] == label_of_and:
                    y = list(node)
                    y[0] = new_label
                    graph[i] = tuple(y)
                elif node[2] == label_of_and:
                    y = list(node)
                    y[2] = new_label
                    graph[i] = tuple(y)
        list_of_t.extend(temp_graph_and)

    # delete single properties like location and time
    list_of_deleted_nodes = []
    list_of_types = [':location', ':time', ':source', ':purpose', 'topic']
    temp_sub_graph = []
    list_of_nodes = []
    for i, _ in enumerate(g.triples):
        if g.triples[i] == g.triples[0]:
            list_of_nodes.append(g.triples[i][0])
        if g.triples[i][0] not in list_of_nodes:
            temp_node = list(g.triples[i])
            temp1 = g.triples[i][0]
            temp_node[0] = g.triples[i][2]
            temp_node[2] = temp1
            g.triples[i] = tuple(temp_node)
            list_of_nodes.append(g.triples[i][2])
        if g.triples[i] != g.triples[0]:
            if g.triples[i][1] != ':instance':
                list_of_nodes.append(g.triples[i][2])
        if g.triples[i][2] in list_of_deleted_nodes:
            list_of_deleted_nodes.append(g.triples[i][0])
        elif g.triples[i][1] in list_of_types:
            list_of_deleted_nodes.append(g.triples[i][2])
        elif g.triples[i][0] in list_of_deleted_nodes:
            if g.triples[i][1] not in ":instance":
                list_of_deleted_nodes.append(g.triples[i][2])
        else:
            temp_sub_graph.append(g.triples[i])
    list_of_t.append(temp_sub_graph)

    # check if length > 3 (at least 4 elements)
    tree_list = []
    for i in list_of_t:
        if len(i) > 3:
            tree_list.append(i)

        # check if graph is right
        list_of_t = []
        valide = True
        for tree in tree_list:
            linked_nodes_check = []
            valide = True
            for i, node in enumerate(tree):
                if i == 0:
                    linked_nodes_check.append(node[0])
                elif node[0] in linked_nodes_check:
                    linked_nodes_check.append(node[2])
                else:
                    # print(counter)
                    valide = False
            if valide:
                list_of_t.append(tree)
    list_of_t_result = list(map(lambda x: penman.format(penman.configure(Graph(x))), list_of_t))

    return list_of_t_result


def get_subgraphs4(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)
    list_of_t = []

    # Split by every predicate
    check_predicate = lambda string: any([char.isdigit() for char in string.split("-")[-2:]])
    left_set, right_set = set([elem[0] for elem in g.triples]), set([elem[2] for elem in g.triples])
    of_labels = list(left_set.difference(right_set))
    if g.triples[0][0] in of_labels:
        of_labels.remove(g.triples[0][0])

    for i, node in enumerate(g.triples):
        if node[1] in ":instance" and check_predicate(node[2]):
            linked_notes = [node[0]]
            temp_list = []
            for j, inner_node in enumerate(g.triples[i:]):
                if inner_node[0] not in linked_notes:
                    if inner_node[0] in of_labels and inner_node[2] in linked_notes:
                        linked_notes.append(inner_node[0])
                        temp_list.append(inner_node)
                    else:
                        break
                else:
                    if inner_node[1] not in ":instance":
                        linked_notes.append(inner_node[2])
                    temp_list.append(inner_node)
            list_of_t.append(temp_list)

    # delete single arguments like ARG
    split_strings = lambda strings: [string.split() for string in strings]

    list_of_types = open_json_file('data/pred_args_prob_fine_v2.json')

    list_of_t.reverse()
    result = []
    list_of_temp = list_of_t.copy()
    # list_of_t = []
    for graph in list_of_temp:
        left_set, right_set = set([elem[0] for elem in graph]), set([elem[2] for elem in graph])
        of_labels = list(left_set.difference(right_set))
        if graph[0][0] in of_labels:
            of_labels.remove(graph[0][0])

        if graph[0][2] in list_of_types:
            list_of_frequences = list_of_types[graph[0][2]]
            list_of_frequences = split_strings(list_of_frequences)
        else:
            # list_of_t.append(graph)
            break

        for i, freq in enumerate(list_of_frequences):
            array_position = 0
            linked_notes = []
            temp_list = []
            for j, node in enumerate(graph):
                if j == 0 or array_position == len(freq):
                    if j == 0:
                        temp_list.append(node)
                    continue
                if node[1] in freq[array_position] and node[0] in graph[0][0]:
                    array_position += 1
                    linked_notes.append(node[2])
                    temp_list.append(node)
                    for inner_node in graph[j + 1:]:
                        if inner_node[0] not in linked_notes:
                            if inner_node[0] in of_labels and inner_node[2] in linked_notes:
                                linked_notes.append(inner_node[0])
                                temp_list.append(inner_node)
                            else:
                                break
                        else:
                            if inner_node[1] not in ":instance":
                                linked_notes.append(inner_node[2])
                            temp_list.append(inner_node)

            if array_position == len(freq):
                list_of_t.append(temp_list)
                result.append(temp_list)

    list_of_types = [':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5', 'ARG6', 'ARG7', 'ARG8', 'ARG9']
    temp_list_of_t = list_of_t.copy()
    for graph in temp_list_of_t:
        temp_sub_graph = []
        list_of_nodes = []
        list_of_deleted_nodes = []
        for i, _ in enumerate(graph):
            if graph[i] == graph[0]:
                list_of_nodes.append(graph[i][0])
            if graph[i][0] not in list_of_nodes:
                temp_node = list(graph[i])
                temp1 = graph[i][0]
                temp_node[0] = graph[i][2]
                temp_node[2] = temp1
                graph[i] = tuple(temp_node)
                list_of_nodes.append(graph[i][2])
            if graph[i] != graph[0]:
                if graph[i][1] != ':instance':
                    list_of_nodes.append(graph[i][2])
            if graph[i][2] in list_of_deleted_nodes:
                list_of_deleted_nodes.append(graph[i][0])
            elif graph[i][1] in list_of_types:
                list_of_deleted_nodes.append(graph[i][2])
            elif graph[i][0] in list_of_deleted_nodes:
                if graph[i][1] not in ":instance":
                    list_of_deleted_nodes.append(graph[i][2])
            else:
                temp_sub_graph.append(graph[i])
        list_of_t.append(temp_sub_graph)

    # TODO put lst of t in there in a loop
    list_t_copy = list_of_t.copy()
    for t in list_t_copy:
        list_of_t.extend(split_at_and(t))

    # remove duplicates
    # no_duplicats = [item for i, item in enumerate(list_of_t) if item not in list_of_t[:i]]

    # check if length > 3 (at least 4 elements)
    tree_list = []
    for i in list_of_t:  # no_duplicats:
        if 3 < len(i):
            tree_list.append(i)
    if len(tree_list) == 0:
        tree_list.append(g.triples)
    tree_list.sort(key=len)
    filtered_list = list(map(lambda x: penman.format(penman.configure(Graph(x))), tree_list))

    return filtered_list


def get_subgraphs4_plus(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)
    list_of_t = []
    linked_nodes_check = []
    # check if graph is right
    for i, node in enumerate(g.triples):
        if i == 0:
            linked_nodes_check.append(node[0])
        elif node[0] in linked_nodes_check:
            linked_nodes_check.append(node[2])
        else:
            g.triples[i] = (node[2], node[1], node[0])
            linked_nodes_check.append(node[0])

    if g.triples[0][2] == "and":
        for id, node in enumerate(g.triples):
            if id == 0:
                continue
            elif "op" in node[1]:
                list_of_t.append([])
            else:
                list_of_t[-1].append(node)



    else:
        # Split by every predicate
        check_predicate = lambda string: any([char.isdigit() for char in string.split("-")[-2:]])
        left_set, right_set = set([elem[0] for elem in g.triples]), set([elem[2] for elem in g.triples])
        of_labels = list(left_set.difference(right_set))
        if g.triples[0][0] in of_labels:
            of_labels.remove(g.triples[0][0])

        for i, node in enumerate(g.triples):
            if node[1] in ":instance":  # and check_predicate(node[2]):
                linked_notes = [node[0]]
                temp_list = []
                for j, inner_node in enumerate(g.triples[i:]):
                    if inner_node[0] not in linked_notes:
                        if inner_node[0] in of_labels and inner_node[2] in linked_notes:
                            linked_notes.append(inner_node[0])
                            temp_list.append(inner_node)
                        else:
                            break
                    else:
                        if inner_node[1] not in ":instance":
                            linked_notes.append(inner_node[2])
                        temp_list.append(inner_node)
                list_of_t.append(temp_list)

        # define lambda function
        split_strings = lambda strings: [string.split() for string in strings]

        list_of_types = open_json_file('data/pred_args_core.json')

        result = []
        list_of_temp = list_of_t.copy()
        list_of_t = []
        for graph in list_of_temp:
            left_set, right_set = set([elem[0] for elem in graph]), set([elem[2] for elem in graph])
            of_labels = list(left_set.difference(right_set))
            if graph[0][0] in of_labels:
                of_labels.remove(graph[0][0])

            # create possible combination of args
            if graph[0][2] in list_of_types or True:
                graph_args = []
                for i_temp in graph:
                    if i_temp[1] not in graph_args and i_temp[1] not in [':instance']:
                        graph_args.append(i_temp[1])
                list_of_frequences = []
                for n in range(len(graph_args) + 1):  # _arg) + 1):
                    if n == 0:
                        continue
                    else:
                        list_of_frequences.extend(list(combinations(graph_args, n)))  # _arg, n)))
                temp_l = []
                for t_i in list_of_frequences:
                    temp_l.append(list(t_i))

                list_of_frequences = temp_l  # split_strings(temp_l)

            else:
                break

            # bring list in format of the level of the graph
            converted_tree = convert_tree(graph)

            if len(converted_tree) == 1 or not converted_tree[1]:
                continue
            else:

                for i, freq in enumerate(list_of_frequences):
                    arg_index = 0
                    right_freq = False
                    temp_g = {}
                    link_g = {}
                    root_nodes = []

                    for l, graph_level in enumerate(converted_tree):
                        for level_element in graph_level:
                            if len(level_element) == 1 and l != 1:
                                for key in temp_g:
                                    stop_loop = False
                                    for i_d, sub_tree in enumerate(temp_g[key]):
                                        if level_element[0][0] in link_g[key][i_d] and not stop_loop:
                                            sub_tree_links = link_g[key][i_d].copy()
                                            for index in range(3):
                                                if (level_element[0][0] == sub_tree[-1][2] or "name" in sub_tree[-2][
                                                    2]) and index == 0:
                                                    temp_g[key][i_d] = sub_tree + level_element
                                                    link_g[key][i_d].append(level_element[0][2])
                                                    break
                                                else:
                                                    sub_tree.pop(-1)
                                                    if index % 2 == 0:
                                                        sub_tree_links.pop(-1)

                                                    if level_element[0][0] == sub_tree[-1][2]:
                                                        temp_g[key].append(sub_tree + level_element)
                                                        link_g[key].append([sub_tree_links, level_element[0][2]])
                                                        stop_loop = True
                                                        break
                                else:
                                    continue
                            elif not right_freq and l == 0:
                                if freq[arg_index] == level_element[1][1]:
                                    arg_index += 1
                                    root_nodes.append(level_element)
                                    temp_g[level_element[1][1]] = []
                                    link_g[level_element[1][1]] = []
                            elif l == 1:
                                for n in root_nodes:
                                    if n[1][2] == level_element[0][0]:
                                        if n[1][1] in [":frequency", ":name", ":manner"]:
                                            if not temp_g[n[1][1]]:
                                                temp_g[n[1][1]].append(n + level_element)
                                            else:
                                                temp_g[n[1][1]] = [temp_g[n[1][1]][0] + level_element]
                                        else:
                                            temp_g[n[1][1]].append(n + level_element)
                                        if len(level_element) == 1:
                                            link_g[n[1][1]].append([n[1][2]])
                                        else:
                                            link_g[n[1][1]].append([n[1][2], level_element[1][2]])
                            else:
                                for key in temp_g:
                                    stop_loop = False
                                    for i_d, sub_tree in enumerate(temp_g[key]):
                                        if level_element[0][0] in link_g[key][i_d] and not stop_loop:
                                            sub_tree_temp = sub_tree.copy()
                                            sub_tree_links = link_g[key][i_d].copy()
                                            for index in range(3):
                                                if (level_element[0][0] == sub_tree[-1][2] or "name" in sub_tree[-2][
                                                    2]) and index == 0:
                                                    temp_g[key][i_d] = sub_tree + level_element
                                                    link_g[key][i_d].append(level_element[1][2])
                                                    break
                                                else:
                                                    sub_tree_temp.pop(-1)
                                                    if index % 2 == 0:
                                                        sub_tree_links.pop(-1)
                                                    if level_element[0][0] == sub_tree_temp[-1][2]:
                                                        temp_g[key].append(sub_tree_temp + level_element)
                                                        link_g[key].append(sub_tree_links + [level_element[1][2]])
                                                        stop_loop = True
                                                        break
                                else:
                                    continue
                            if arg_index == len(freq):
                                right_freq = True

                    joined_lists = join_lists(
                        list(temp_g.values())) 

                    if not joined_lists:
                        continue
                    else:
                        if right_freq:
                            for t_list in joined_lists:
                                list_of_t.append(remove_duplicates(t_list))

        # remove and in the tree
        list_t_copy = list_of_t.copy()
        for t in list_t_copy:
            delete_and_node(t)

    # check if length > 3 (at least 4 elements)
    tree_list = []
    for i in list_of_t:  # no_duplicats:
        if 3 < len(i):
            tree_list.append(i)
    if len(tree_list) == 0:
        tree_list.append(g.triples)

    # check if graph is right
    list_of_t = []
    valide = True
    for counter, tree in enumerate(tree_list):
        linked_nodes_check = []
        valide = True
        for i, node in enumerate(tree):
            if i == 0:
                linked_nodes_check.append(node[0])
            elif node[0] in linked_nodes_check:
                linked_nodes_check.append(node[2])
            else:
                # print(counter)
                valide = False
        if valide:
            list_of_t.append(tree)

    filtered_list = list(map(lambda x: penman.format(penman.configure(Graph(x))), list_of_t))

    return filtered_list


def join_lists(lists):
    if not lists:
        return [[]]
    result = []
    for item in lists[0]:
        for inner_list in join_lists(lists[1:]):
            result.append(item + inner_list)
    return result


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def delete_and_node(tree):
    for i, node in enumerate(tree):
        if i == 0:
            continue
        elif node[2] == "and" and i < (len(tree) - 1):
            y = list(tree[i - 1])
            y[2] = tree[i + 1][2]
            tree[i - 1] = tuple(y)
            tree.pop(i)
            tree.pop(i)
            break

    return tree


# not used
def get_subgraphs5(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)
    result_subtrees = []

    # bring list in format of the level of the graph
    converted_tree = convert_tree(g.triples)

    # check for persons and generate short sentences with the relation to the person
    person = []
    for level in converted_tree:
        for nodes in level:
            if "instance" in nodes[0][1] and "person" in nodes[0][2]:
                person.extend(nodes)

    return list(map(lambda x: penman.format(penman.configure(Graph(x))), result_subtrees))


def convert_tree(tree):
    converted_tree = []
    # connections = []
    result, subtrees = split_at_node(tree)
    converted_tree.append(result)

    while len(subtrees) != 0:
        temp_results = []
        temp_subtrees = []
        for t in subtrees:
            result, subt = split_at_node(t)
            temp_results.extend(result)
            temp_subtrees.extend(subt)
        converted_tree.append(temp_results)
        subtrees = temp_subtrees.copy()

    return converted_tree


def split_at_node(tree):
    base_node_tuple = tree[0]
    top_level_nodes = []
    list_of_subtrees = []
    new_subtree = False

    base_node = tree[0][0]
    for i, subtree in enumerate(tree):
        if i == 0:
            continue
        if subtree[0] == base_node:  # or subtree[2] == base_node:
            top_level_nodes.append([base_node_tuple, subtree])
            new_subtree = True
        else:
            if new_subtree:
                list_of_subtrees.append([subtree])
                new_subtree = False
            else:
                if not list_of_subtrees:
                    list_of_subtrees.append([subtree])
                else:
                    list_of_subtrees[-1].append(subtree)

    if not top_level_nodes:
        top_level_nodes = [[base_node_tuple]]
    return top_level_nodes, list_of_subtrees


def split_at_and(tree):
    # spliten after op, if it there are leaves of and

    # check if polarity after and
    # for i, elem in enumerate(tree):
    #     if elem[2] in "and":
    #         if (len(tree) -1) > i:
    #             if "polarity" in tree[i + 1][1]:
    #                 tree[i], tree[i + 1] = tree[i + 1], tree[i]
    #                 break

    # check and presence
    list_of_t = []
    temp_graph_and = [[]]
    i = 0
    end_of_outer_loop = True
    while end_of_outer_loop:
        node = tree[i]
        for n in temp_graph_and:
            n.append(node)
        if node[2] == "and":
            end_of_inner_loop = True
            if i == 0:
                node_id_of_arg = ('XXX', 'XXX', 'XXX')
            else:
                node_id_of_arg = tree[i - 1]
            node_id_of_and = tree[i]
            subtree_graphs = []
            i += 1

            number_of_subtrees = len(temp_graph_and)
            while end_of_inner_loop:
                node = tree[i]
                if node[0] == node_id_of_and[0] and ":op" in node[1]:
                    for n in temp_graph_and:
                        tempp = n.copy()
                        tempp.append(node)
                        subtree_graphs.append(tempp)

                elif node[0] == node_id_of_arg[0] or i >= len(tree) - 1:
                    for j in range(number_of_subtrees):
                        subtree_graphs[-(j + 1)].append(node)
                    if i >= len(tree) - 1:
                        end_of_outer_loop = False
                    temp_graph_and = subtree_graphs
                    i -= 2
                    end_of_inner_loop = False


                else:
                    for j in range(number_of_subtrees):
                        subtree_graphs[-(j + 1)].append(node)
                if len(tree) - 1 == i:
                    end_of_inner_loop = False
                    end_of_outer_loop = False
                else:
                    i += 1

        if i >= len(tree) - 1:
            end_of_outer_loop = False
        else:
            i += 1
    if len(temp_graph_and) != 1:

        for graph in temp_graph_and:
            label_of_and = 'XXX'
            for i, node in enumerate(graph):
                if node[2] == "and" and i == 0:
                    label_of_and = graph[0][0]
                    graph.pop(0)
                    graph.pop(0)
                    new_label = graph[0][0]
                elif node[2] == "and":
                    label_of_and = node[0]
                    new_label = graph[i + 1][2]

                    if graph[i - 1][0] == label_of_and:
                        y = list(graph[i - 1])
                        y[0] = graph[i + 1][2]
                        graph[i - 1] = tuple(y)
                    else:
                        y = list(graph[i - 1])
                        y[2] = graph[i + 1][2]
                        graph[i - 1] = tuple(y)
                    graph.pop(i)
                    graph.pop(i)
                elif node[0] == label_of_and:
                    y = list(node)
                    y[0] = new_label
                    graph[i] = tuple(y)
                elif node[2] == label_of_and:
                    y = list(node)
                    y[2] = new_label
                    graph[i] = tuple(y)
        list_of_t.extend(temp_graph_and)
    return list_of_t


def get_subgraphs(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)

    t = penman.configure(g)

    dict_variables = {}
    root_node = t.node[0]
    subgraphs = []
    for path, branch in t.walk():
        val_node = path_var(path, t.node)
        if val_node not in dict_variables:
            dict_variables[val_node] = branch

        if val_node != root_node:
            continue

        role, target = branch
        if isinstance(target, tuple):
            linearized_graph = penman.format((val_node, [dict_variables[val_node], (role, target)]))
            subgraphs.append(linearized_graph)
    return subgraphs


def get_concepts(g_tag):
    tokens = g_tag.split()
    dict_concepts = {}
    for t in tokens:
        if "~" in t:
            t = t.replace("(", "")
            t = t.replace(")", "")
            parts_t = t.split("~")
            dict_concepts[parts_t[0]] = t
    return dict_concepts


def replace_graph_with_tags(dict_tag, graph):
    for key, value in dict_tag.items():
        graph = graph.replace(key, value)
    return graph


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def open_json_file(filename):
    with open(filename) as f:
        data_json = json.load(f)
        return data_json


def open_jsonl_file(filename):
    with open(filename) as f:
        data_json = [json.loads(jline) for jline in f.read().splitlines()]
        return data_json


def bert_cluster(sentences):
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Encode sentences using BERT
    sentence_embeddings = []
    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(input_ids)[0].mean(dim=1).squeeze().numpy()
        sentence_embeddings.append(embeddings)

    # Use silhouette score to determine the optimal number of clusters
    sil_scores = []
    min_k = 3
    if len(sentences) <= 7:
        max_k = len(sentences)
        if len(sentences) <= 3:
            min_k = 2
    else:
        max_k = 7
    for k in range(min_k, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(sentence_embeddings)
        sil_scores.append(silhouette_score(sentence_embeddings, kmeans.labels_))
    optimal_k = np.argmax(sil_scores) + 2

    # Cluster sentences using KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(sentence_embeddings)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embeddings)

    # Get the best-represented sentence from each cluster
    best_sentences = [sentences[idx] for idx in closest]

    return best_sentences


def sentences_with_verbs(sentences):
    # Load English language model
    nlp = spacy.load("en_core_web_sm")

    # List to store sentences with verbs
    verb_sentences = []

    # Iterate over each sentence in the input list
    for sentence in sentences:
        # Parse sentence using spaCy
        doc = nlp(sentence)

        # Check if sentence contains at least one verb and one subject and one object
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_subject = False
        has_object = False
        for token in doc:
            if token.dep_ == "nsubj":
                has_subject = True
            elif token.dep_ == "dobj":
                has_object = True
        has_predicate = has_verb and has_subject and has_object

        # If sentence has a subject, verb, and object, or if it has at least one verb, add it to the output list
        if has_predicate or has_verb:
            verb_sentences.append(sentence)

    return verb_sentences


def inner_thread(inputs, top_k):
    s, g, g_tag = inputs
    summary_trees = [g]
    list_of_sents = []
    list_of_trees = []
    result_sents = []
    result_trees = []

    dict_tag = get_concepts(g_tag)
    temp_sent_list = []
    for sgf in [get_subgraphs3]: #4_plus]:  # , get_subgraphs, get_subgraphs2, get_subgraphs3, get_subgraphs4]:
        subgraphs = sgf(g)
        # Fallback okay ? --> Original sentence for default if too short?

        subgraphs_tag = []
        for sb in subgraphs:
            sb = maybe_fix_unlinked_in_subgraph(g, sb)
            list_of_trees.append(sb)
            sb = gstring_to_oneline(sb)
            sb = replace_graph_with_tags(dict_tag, sb)
            subgraphs_tag.append(sb)
            # print("-")

        sents, _ = gtos.generate_taged(subgraphs_tag, disable_progress=True)
        temp_sent_list.extend(sents)

    return summary_trees, temp_sent_list, list_of_trees


def run_amr(filename, data_json):
    outputDict = []
    duplicate_counter = 0

    for index_i, key in enumerate(data_json.keys()):

        se = data_json[key][0]

        # if "\u00a0" in se:
        #    se = se.replace("\u00a0", '')
        se = re.sub(r'\s+', ' ', se)
        if "realsumm" in filename:
            # # initializing tag
            # tag = "t"
            # # regex to extract required strings
            # reg_str = "<" + tag + ">(.*?)</" + tag + ">"
            # sentences = re.findall(reg_str, se)
            page_doc = spacy_o(se, disable=["tagger"])
            sentences = [se.text for se in page_doc.sents]
            top_k = 5
        elif "pyrxsum" in filename:
            sentences = [se]
            top_k = 5
        else:
            page_doc = spacy_o(se, disable=["tagger"])
            sentences = [se.text for se in page_doc.sents]
            top_k = 2

        # Filter the sentence if they contain words
        filtered_sentences = []
        for sentence in sentences:
            num_count = sum(c.isdigit() for c in sentence)
            num_char = sum(1 for c in sentence)
            if num_count / num_char < 0.5:
                sen = sentence.split(';')
                if len(sen) == 1:
                    filtered_sentences.append(sentence)
                else:
                    for s in sen:
                        if len(s) > 3:
                            filtered_sentences.append(s)

        graphs, graphs_tags = stog.parse_sents(filtered_sentences, add_metadata=True)

        print(key)
        list_of_sents = []
        list_of_trees = []
        result_sents = []
        result_trees = []
        summary_trees = []

        pool = multiprocessing.Pool(6)
        results = []

        for idx, (s, g, g_tag) in enumerate(zip(sentences, graphs, graphs_tags)):

            res = pool.apply_async(inner_thread, args=((s, g, g_tag), top_k))
            results.append(res)

        pool.close()
        pool.join()
        for res in results:
            s_t, r_s, r_t = res.get()
            summary_trees.extend(s_t)
            result_sents.extend(r_s)
            result_trees.extend(r_t)
            # do something with the results

        outputDict.append(
            {'instance_id': key,
                'summary': se,
                'summary_trees': summary_trees,
                'tree': result_trees,
                'smus': result_sents, }
        )

    jsonString = json.dumps(outputDict)
    jsonFile = open(filename, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    print(f"Number of duplicates: {duplicate_counter}")


'''
# Create smus out of Tac2008 data
def run_tac08_amr(scu_path, result_path):
    data_json = open_json_file(scu_path)
    run_amr(result_path, data_json)


# Create smus out of Tac2009 data
def run_tac09_amr(scu_path, result_path):
    data_json = open_json_file(scu_path)
    run_amr(result_path, data_json)


# Create smus out of PyrXSum data
def run_pyyrxsum_amr(scu_path, result_path):
    data_json = open_json_file(scu_path)
    run_amr(result_path, data_json)


# Create smus out of REALSumm data
def run_realsumm_amr(scu_path, result_path):
    data_json = open_json_file(scu_path)
    run_amr(result_path, data_json)
'''


def run_amr_data(scus, result_path):
    index = [value['instance_id'] for value in scus]
    summaries = [value['summary'] for value in scus]
    golden_summaries = _get_and_replace_coref(summaries, index, torch.device("mps"))
    run_amr(result_path, golden_summaries)


if __name__ == '__main__':
    run_amr_data(open_json_file(f'{data_path}/pyrxsum/pyrxsum-scus.json'),
                 f'{data_path}/pyrxsum/pyrxsum-smus-sg3.json')

    run_amr_data(open_json_file(f'{data_path}/realsumm/realsumm-scus.json'),
                 f'{data_path}/realsumm/realsumm-smus-sg3.json')

    run_amr_data(open_json_file(f'{data_path}/tac08/tac08-scus.json'),
                 f'{data_path}/tac08/tac2008-smus-sg3.json')

    run_amr_data(open_json_file(f'{data_path}/tac09/tac09-scus.json'),
                 f'{data_path}/tac09/tac2009-smus-sg3.json')
