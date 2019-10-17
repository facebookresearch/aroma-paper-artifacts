#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pickle
import random
import re
from collections import Counter, OrderedDict

from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


working_dir = None
config = None
vocab = None
options = None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--corpus",
        action="store",
        dest="corpus",
        default=None,
        help="Process raw ASTs, featurize, and store in the working directory.",
    )
    parser.add_argument(
        "-d",
        "--working-dir",
        action="store",
        dest="working_dir",
        help="Working directory.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file-query",
        action="append",
        dest="file_query",
        default=[],
        help="File containing the query AST of a query code as JSON.",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        action="append",
        dest="keywords",
        default=[],
        help="Keywords to search for.",
    )
    parser.add_argument(
        "-i",
        "--index-query",
        type=int,
        action="store",
        dest="index_query",
        default=None,
        help="Index of the query AST in the corpus.",
    )
    parser.add_argument(
        "-t",
        "--testall",
        dest="testall",
        action="store_true",
        default=False,
        help="Sample config.N_SAMPLES snippets and search.",
    )
    options = parser.parse_args()
    logging.info(options)
    return options


class Config:
    def __init__(self):
        self.MIN_MERGED_CODE = 3
        self.MIN_PRUNED_SCORE = 0.65
        self.N_PARENTS = 3
        self.N_SIBLINGS = 1
        self.N_VAR_SIBLINGS = 2
        self.NUM_SIMILARS = 100
        self.MIN_SIMILARITY_SCORE = 0.4
        self.VOCAB_FILE = "vocab.pkl"
        self.TFIDF_FILE = "tfidf.pkl"
        self.FEATURES_FILE = "features.json"
        self.NUM_FEATURE_MIN = 10
        self.DBSCAN_EPS = 0.1
        self.SAMPLE_METHOD_MIN_LINES = 12
        self.SAMPLE_METHOD_MAX_LINES = 7
        self.METHOD_MAX_LINES = 150
        self.SEED = 119
        self.N_SAMPLES = 100
        self.IGNORE_VAR_NAMES = True
        self.IGNORE_SIBLING_FEATURES = False
        self.IGNORE_VAR_SIBLING_FEATURES = False
        self.CLUSTER = True
        self.PRINT_SIMILAR = True
        self.USE_DBSCAN = True
        self.THRESHOLD1 = 0.9
        self.THRESHOLD2 = 1.5
        self.TOP_N = 5


class Vocab:
    def __init__(self):
        self.vocab = OrderedDict()
        self.words = []

    def get_word(self, i):
        if i <= config.NUM_FEATURE_MIN:
            return "#UNK"
        return self.words[i - 1 - config.NUM_FEATURE_MIN]

    def add_and_get_index(self, word):
        if not (word in self.vocab):
            self.words.append(word)
            self.vocab[word] = [0, len(self.vocab) + 1 + config.NUM_FEATURE_MIN]
        value = self.vocab[word]
        value[0] += 1
        return value[1]

    def get_index(self, word):
        if word in self.vocab:
            return self.vocab[word][1]
        else:
            return config.NUM_FEATURE_MIN

    def dump(self):
        with open(os.path.join(options.working_dir, config.VOCAB_FILE), "wb") as out:
            pickle.dump([self.vocab, self.words], out)
            logging.info(f"Dumped vocab with size {len(self.vocab)}")

    @staticmethod
    def load(init=False):
        tmp = Vocab()
        if not init:
            try:
                with open(
                    os.path.join(options.working_dir, config.VOCAB_FILE), "rb"
                ) as out:
                    [tmp.vocab, tmp.words] = pickle.load(out)
                logging.info(f"Loaded vocab with size {len(tmp.vocab)}")
            except:
                logging.info("Initialized vocab.")
                pass
        return tmp


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def ast_to_code_aux(ast, token_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_aux(elem, token_list)
    elif isinstance(ast, dict):
        token_list.append(ast["leading"])
        token_list.append(ast["token"])


def ast_to_code_collect_lines(ast, line_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_collect_lines(elem, line_list)
    elif isinstance(ast, dict):
        if "line" in ast:
            line_list.append(ast["line"])


def ast_to_code_print_lines(ast, line_list, token_list):
    if isinstance(ast, list):
        for elem in ast:
            ast_to_code_print_lines(elem, line_list, token_list)
    elif isinstance(ast, dict):
        if "line" in ast and ast["line"] in line_list:
            if len(token_list) > 0 and token_list[-1] == "//":
                token_list.append(" your code ...\n")
            token_list.append(ast["leading"])
            token_list.append(ast["token"])
        else:
            if len(token_list) > 0 and token_list[-1] != "//":
                token_list.append("\n")
                token_list.append("//")


def featurize_records_file(rpath, wpath):
    with open(rpath, "r") as inp:
        with open(wpath, "w") as outp:
            i = 0
            for line in inp:
                obj = json.loads(line)
                obj["features"] = collect_features_as_list(obj["ast"], True, False)[0]
                obj["index"] = i
                i += 1
                outp.write(json.dumps(obj))
                outp.write("\n")


def append_feature_index(is_init, is_counter, key, feature_list, c):
    if is_init:
        n = vocab.add_and_get_index(key)
    else:
        n = vocab.get_index(key)
    if is_counter:
        if n != str(config.NUM_FEATURE_MIN):
            c[n] += 1
    else:
        feature_list.append(n)


def append_feature_pair(
    is_init, is_counter, key, feature_list, leaf_features, sibling_idx, leaf_idx
):
    if is_init:
        n = vocab.add_and_get_index(key)
    else:
        n = vocab.get_index(key)
    if is_counter:
        if n != str(config.NUM_FEATURE_MIN):
            leaf_features[leaf_idx][n] += 1
            leaf_features[sibling_idx][n] += 1
    else:
        feature_list.append(n)
        feature_list.append(n)


def get_leftmost_leaf(ast):
    if isinstance(ast, list):
        for elem in ast:
            (success, token) = get_leftmost_leaf(elem)
            if success:
                return (success, token)
    elif isinstance(ast, dict):
        if "leaf" in ast and ast["leaf"]:
            return (True, ast["token"])
    return (False, None)


def get_var_context(p_idx, p_label, p_ast):
    if p_label == "#.#":
        return get_leftmost_leaf(p_ast[p_idx + 2])[1]
    else:
        return p_label + str(p_idx)


def collect_features_aux(
    ast,
    feature_list,
    parents,
    siblings,
    var_siblings,
    leaf_features,
    leaf_pair_features,
    is_init,
    is_counter,
):
    global leaf_idx
    if isinstance(ast, list):
        i = 0
        for elem in ast:
            parents.append((i, ast[0], ast))
            collect_features_aux(
                elem,
                feature_list,
                parents,
                siblings,
                var_siblings,
                leaf_features,
                leaf_pair_features,
                is_init,
                is_counter,
            )
            parents.pop()
            i += 1
    elif isinstance(ast, dict):
        if "leaf" in ast and ast["leaf"]:
            leaf_idx += 1
            is_var = False
            var_name = key = ast["token"]
            if config.IGNORE_VAR_NAMES and "var" in ast and not key[0].isupper():
                key = "#VAR"
                is_var = True
            c = None
            if is_counter:
                c = Counter()
                leaf_features.append(c)
            append_feature_index(is_init, is_counter, key, feature_list, c)
            count = 0
            for (i, p, p_ast) in reversed(parents):
                if p != "(#)" and re.match("^\{#*\}$", p) is None:
                    count += 1
                    key2 = p + str(i) + ">" + key
                    append_feature_index(is_init, is_counter, key2, feature_list, c)
                    if count >= config.N_PARENTS:
                        break
            count = 0
            if not config.IGNORE_VAR_SIBLING_FEATURES and is_var:
                (p_idx, p_label, p_ast) = parents[-1]
                var_context = get_var_context(p_idx, p_label, p_ast)
                if var_context is not None:
                    if var_name not in var_siblings:
                        var_siblings[var_name] = []
                    for (var_sibling_idx, var_context_sibling) in reversed(
                        var_siblings[var_name]
                    ):
                        count += 1
                        key2 = var_context_sibling + ">>>" + var_context
                        #                        logging.info(f"var sibling feature {key2}")
                        append_feature_pair(
                            is_init,
                            is_counter,
                            key2,
                            feature_list,
                            leaf_features,
                            var_sibling_idx,
                            leaf_idx - 1,
                        )
                        if count >= config.N_VAR_SIBLINGS:
                            break
                    var_siblings[var_name].append((leaf_idx - 1, var_context))
            count = 0
            if not config.IGNORE_SIBLING_FEATURES:  # and not is_var:
                for (sibling_idx, sibling) in reversed(siblings):
                    count += 1
                    key2 = sibling + ">>" + key
                    append_feature_pair(
                        is_init,
                        is_counter,
                        key2,
                        feature_list,
                        leaf_features,
                        sibling_idx,
                        leaf_idx - 1,
                    )
                    if count >= config.N_SIBLINGS:
                        break
                siblings.append((leaf_idx - 1, key))


def feature_list_to_doc(record):
    return " ".join([str(y) for y in record["features"]])


def counter_vectorize(rpath, wpath):
    with open(rpath, "r") as f:
        records = f.readlines()
    documents = [feature_list_to_doc(json.loads(x)) for x in records]
    vectorizer = CountVectorizer(min_df=1, binary=True)
    counter_matrix = vectorizer.fit_transform(documents)
    with open(wpath, "wb") as outf:
        pickle.dump((vectorizer, counter_matrix), outf)


def read_all_records(rpath):
    with open(rpath, "r") as f:
        records = f.readlines()
    ret = [json.loads(x) for x in records]
    return ret


def get_record_part(record):
    n_lines = record["endline"] - record["beginline"]
    if n_lines < config.SAMPLE_METHOD_MIN_LINES or "tests" in record["path"]:
        return None
    else:
        (_, ast) = get_sub_ast_aux(record["ast"], record["beginline"])
        if ast == None:
            return None
        else:
            ret = copy_record_with_ast(record, ast)
            ret["features"] = collect_features_as_list(ast, False, False)[0]
            return ret


def get_sub_ast_aux(ast, beginline, stop=False):
    if isinstance(ast, list):
        if stop:
            return (stop, None)
        else:
            ret = []
            for elem in ast:
                (stop, tmp) = get_sub_ast_aux(elem, beginline, stop)
                if tmp != None:
                    ret.append(tmp)
            if len(ret) >= 2:
                return (stop, ret)
            else:
                return (True, None)
    elif isinstance(ast, dict):
        if (
            "leaf" not in ast
            or not ast["leaf"]
            or (not stop and ast["line"] - beginline < config.SAMPLE_METHOD_MAX_LINES)
        ):
            return (stop, ast)
        else:
            return (True, None)
    else:
        return (stop, ast)


def print_features(fstr):
    print(" ".join([vocab.get_word(int(k)) for k in fstr]))


def sample_n_records(records, n):
    ret_indices = []
    ret_records = []
    for j in range(10000):
        if len(ret_indices) < n:
            i = random.randint(0, len(records) - 1)
            if not (i in ret_indices):
                record = get_record_part(records[i])
                if record != None:
                    ret_indices.append(i)
                    ret_records.append(record)
        else:
            logging.info("Sampled records")
            return (ret_indices, ret_records)
    logging.info("Sampled records")
    return (ret_indices, ret_records)


def my_similarity_score(M1, M2):
    return linear_kernel(M1, M2)


def find_similarity_score_features_un(record1, record2):
    features_as_counter1 = Counter(record1["features"])
    features_as_counter2 = Counter(record2["features"])
    return sum((features_as_counter1 & features_as_counter2).values())


def find_similarity_score_features_set(records):
    features_as_counters = []
    for record in records:
        features_as_counters.append(Counter(record["features"]))
    return distance_set(features_as_counters)


def find_similarity_score_features_set_un(records):
    features_as_counters = []
    for record in records:
        features_as_counters.append(Counter(record["features"]))
    intersection = None
    for counter in features_as_counters:
        if intersection is None:
            intersection = counter
        else:
            intersection = intersection & counter
    return sum(intersection.values())


def copy_record_with_ast(record, ast):
    ret = dict(record)
    ret["ast"] = ast
    return ret


def copy_record_with_features(record, features):
    ret = dict(record)
    ret["features"] = features
    return ret


def copy_leaf_dummy(ast):
    return {"token": " ... ", "leading": ast["leading"], "trailing": ast["trailing"]}


leaf_idx = 0


def prune_ast(ast, leaf_features):
    global leaf_idx
    if isinstance(ast, list):
        no_leaf = True
        ret = []
        for elem in ast:
            (flag, tmp) = prune_ast(elem, leaf_features)
            ret.append(tmp)
            no_leaf = no_leaf and flag
        if no_leaf:
            return (True, None)
        else:
            return (False, ret)
    elif isinstance(ast, dict):
        if "leaf" in ast and ast["leaf"]:
            leaf_idx += 1
            if leaf_features[leaf_idx - 1] is None:
                return (True, copy_leaf_dummy(ast))
            else:
                return (False, ast)
        else:
            return (True, ast)
    else:
        return (True, ast)


def prune_second_jd(record1, record2):
    return prune_last_jd([record1], record2)


def add_pair_features(new_features, leaf_pair_features, leaf_idx, current_leaf_indices):
    if not config.IGNORE_SIBLING_FEATURES:
        for sibling_idx in current_leaf_indices:
            if (sibling_idx, leaf_idx) in leaf_pair_features:
                new_features[leaf_pair_features[(sibling_idx, leaf_idx)]] += 1


def distance_set(counters):
    intersection = None
    union = None
    for counter in counters:
        if intersection is None:
            intersection = counter
        else:
            intersection = intersection & counter
    for counter in counters:
        if union is None:
            union = counter
        else:
            union = union | counter
    return sum(intersection.values()) / sum(union.values())


def distance(counter1, counter2):
    return sum((counter1 & counter2).values()) / sum((counter1 | counter2).values())


def copy_record(record2, ast, features):
    ret = dict(record2)
    ret["ast"] = ast
    ret["features"] = features
    ret["index"] = -1
    return ret


def get_completions_via_clustering(query_record, similar_records):
    features = [feature_list_to_doc(record) for record in similar_records]
    if len(features) > 1:
        vectorizer = CountVectorizer(min_df=1)
        X = vectorizer.fit_transform(features)
        if config.USE_DBSCAN:
            db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=2, metric="cosine")
            labels = db.fit_predict(X)
        else:
            db = AffinityPropagation()
            labels = db.fit_predict(X)
    else:
        labels = [0]

    print(f"Clustering labels: {labels}")
    logging.info(f"Clustering labels: {labels}")
    index_pairs = OrderedDict()
    ret = []
    n_clusters = 0
    n_uniques = 0
    for i in range(min(config.MIN_MERGED_CODE, len(similar_records))):
        if labels[i] < 0:
            ret.append((similar_records[i]["ast"], i, i))
    for i in range(len(labels)):
        if labels[i] >= 0:
            if labels[i] in index_pairs:
                if len(index_pairs[labels[i]]) == 1:
                    index_pairs[labels[i]].append(i)
            else:
                index_pairs[labels[i]] = [i]
                n_clusters += 1
        else:
            n_uniques += 1

    for p in index_pairs.values():
        if len(p) == 2:
            (i, j) = p
            pruned_record = prune_last_jd(
                [query_record, similar_records[j]], similar_records[i]
            )
            ret.append((pruned_record, i, j))
        else:
            ret.append((similar_records[p[0]]["ast"], p[0], p[0]))

    ret.sort(key=lambda t: t[1])
    logging.info(
        f"(# similars, #clusters, #singles, #completions) = ({len(similar_records)}, {n_clusters}, {n_uniques}, {len(ret)})"
    )
    print(
        f"(# similars, #clusters, #singles, #completions) = ({len(similar_records)}, {n_clusters}, {n_uniques}, {len(ret)})"
    )
    return ret


def get_completions2(query_record, candidate_records):
    l = len(candidate_records)
    ret = []

    n_clusters = 0
    n_uniques = 0
    print("2-way")
    for i in range(l):
        jmax = None
        maxscore = 0
        for j in range(i + 1, l):
            pscore = find_similarity_score_features(
                candidate_records[i][2], candidate_records[j][2]
            )
            if pscore > config.THRESHOLD1:
                query_score_un = find_similarity_score_features_un(
                    candidate_records[i][2], candidate_records[j][2]
                )
                tmp_score = find_similarity_score_features_un(
                    candidate_records[i][0], candidate_records[j][0]
                )
                if (
                    tmp_score > config.THRESHOLD2 * query_score_un
                    and tmp_score > maxscore
                ):
                    jmax = j
                    maxscore = tmp_score
        if jmax is not None:
            pruned_record = prune_last_jd(
                [query_record, candidate_records[jmax][0]], candidate_records[i][0]
            )
            ret.append((pruned_record, i, jmax))
            print(ast_to_code(pruned_record["ast"]))
            n_clusters += 1
        # else:
        #     ret.append((candidate_records[i][0]['ast'], i, i))
        #     n_uniques += 1

    ret2 = []
    print("3-way")
    for (record, i, j) in ret:
        if i != j:
            kmax = None
            maxscore = 0
            for k in range(l):
                if k != i and k != j:
                    pscore = find_similarity_score_features_set(
                        [
                            candidate_records[i][2],
                            candidate_records[j][2],
                            candidate_records[k][2],
                        ]
                    )
                    if pscore > config.THRESHOLD1:
                        query_score_un = find_similarity_score_features_set_un(
                            [
                                candidate_records[i][2],
                                candidate_records[j][2],
                                candidate_records[k][2],
                            ]
                        )
                        tmp_score = find_similarity_score_features_set_un(
                            [
                                candidate_records[i][0],
                                candidate_records[j][0],
                                candidate_records[k][0],
                            ]
                        )
                        if (
                            tmp_score > config.THRESHOLD2 * query_score_un
                            and tmp_score > maxscore
                        ):
                            kmax = k
                            maxscore = tmp_score
            if kmax is not None:
                pruned_record = prune_last_jd(
                    [query_record, candidate_records[kmax][0]], record
                )
                n_clusters += 1
                print(ast_to_code(pruned_record["ast"]))
                ret2.append((pruned_record, i, j, kmax))
    logging.info(
        f"(# similars, #clusters, #singles, #completions) = ({len(candidate_records)}, {n_clusters}, {n_uniques}, {len(ret)})"
    )
    print(
        f"(# similars, #clusters, #singles, #completions) = ({len(candidate_records)}, {n_clusters}, {n_uniques}, {len(ret)})"
    )
    return ret2 + ret


def get_completions3(query_record, candidate_records, top_n, threshold1, threshold2):
    l = len(candidate_records)
    ret = []
    acc = []

    for i in range(l):
        ret.append([i])
    changed = True
    while changed:
        ret2 = []
        changed = False
        for tuple in ret:
            kmax = None
            maxscore = 0
            for k in range(tuple[-1] + 1, l):
                record_list1 = []
                record_list2 = []
                for i in tuple:
                    record_list1.append(candidate_records[i][2])
                    record_list2.append(candidate_records[i][0])

                record_list1.append(candidate_records[k][2])
                record_list2.append(candidate_records[k][0])
                qlen = sum(Counter(record_list1[0]["features"]).values())
                iscore = find_similarity_score_features_set_un(record_list1)
                pscore = iscore / qlen
                #                pscore = find_similarity_score_features_set(record_list1)
                if pscore > threshold1:
                    query_score_un = find_similarity_score_features_set_un(record_list1)
                    tmp_score = find_similarity_score_features_set_un(record_list2)
                    if tmp_score > threshold2 * query_score_un and tmp_score > maxscore:
                        kmax = k
                        maxscore = tmp_score
            if kmax is not None:
                changed = True
                ret2.append(tuple + [kmax])
        acc = ret2 + acc
        ret = ret2
    ret = []
    acc = sorted(acc, key=lambda t: t[0] * 1000 - len(t))
    for i in range(len(acc)):
        tuple = acc[i]
        logging.info(f"Pruning {len(tuple)} {tuple}")
        is_subset = False
        s = set(tuple)
        for j in reversed(range(i)):
            if distance(Counter(tuple), Counter(acc[j])) > 0.5:
                is_subset = True
        if not is_subset:
            print(f"Pruning {len(tuple)} {tuple}")
            logging.info("recommending")
            pruned_record = candidate_records[tuple[0]][0]
            for j in range(1, len(tuple)):
                pruned_record = prune_last_jd(
                    [query_record, candidate_records[tuple[j]][0]], pruned_record
                )
            ret.append([pruned_record, candidate_records[tuple[0]][0]] + tuple)
            if len(ret) >= top_n:
                return ret
    return ret


def print_match_index(query_record, candidate_records):
    ret = -1
    i = 0
    for (candidate_record, score, pruned_record, pruned_score) in candidate_records:
        if query_record["index"] == candidate_record["index"]:
            ret = i
        i += 1
    if ret < 0:
        print("Failed to match original method.")
    elif ret > 0:
        print(f"Matched original method. Rank = {ret}")
    else:
        print(f"Matched original method perfectly.")


#### Interface methods ####


def find_indices_similar_to_features(
    vectorizer, counter_matrix, feature_lists, num_similars, min_similarity_score
):
    doc_counter_vector = vectorizer.transform(feature_lists)
    len = my_similarity_score(doc_counter_vector, doc_counter_vector).flatten()[0]
    cosine_similarities = my_similarity_score(
        doc_counter_vector, counter_matrix
    ).flatten()
    related_docs_indices = [
        i
        for i in cosine_similarities.argsort()[::-1]
        if cosine_similarities[i] > min_similarity_score * len
    ][0:num_similars]
    return [(j, cosine_similarities[j]) for j in related_docs_indices]


def find_similarity_score_features(record1, record2):
    features_as_counter1 = Counter(record1["features"])
    features_as_counter2 = Counter(record2["features"])
    return distance(features_as_counter1, features_as_counter2)


def prune_last_jd(records, record2):
    other_features = [Counter(record["features"]) for record in records]
    ast = record2["ast"]
    leaf_features, leaf_pair_features = collect_features_as_list(ast, False, True)
    out_features = [None] * len(leaf_features)
    current_features = Counter()
    current_leaf_indices = []
    for features1 in other_features:
        score = distance(features1, current_features)
        done = False
        while not done:
            max = score
            max_idx = None
            i = 0
            for leaf_feature in leaf_features:
                if leaf_feature is not None:
                    new_features = current_features + leaf_feature
                    #                    add_pair_features(new_features, leaf_pair_features, i, current_leaf_indices)
                    tmp = distance(features1, new_features)
                    if tmp > max:
                        max = tmp
                        max_idx = i
                i += 1
            if max_idx is not None:
                score = max
                out_features[max_idx] = leaf_features[max_idx]
                current_features = current_features + leaf_features[max_idx]
                #                add_pair_features(current_features, leaf_pair_features, max_idx, current_leaf_indices)
                current_leaf_indices.append(max_idx)
                leaf_features[max_idx] = None
            else:
                done = True
    global leaf_idx
    leaf_idx = 0
    pruned_ast = prune_ast(ast, out_features)[1]
    pruned_features = collect_features_as_list(pruned_ast, False, False)[0]
    return copy_record(record2, pruned_ast, pruned_features)


def ast_to_code(tree):
    token_list = []
    ast_to_code_aux(tree, token_list)
    token_list.append("\n")
    return "".join(token_list)


def ast_to_code_with_full_lines(tree, fulltree):
    line_list = []
    ast_to_code_collect_lines(tree, line_list)
    token_list = []
    ast_to_code_print_lines(fulltree, line_list, token_list)
    token_list.append("\n")
    return "".join(token_list)


def find_similar(
    query_record,
    records,
    vectorizer,
    counter_matrix,
    num_similars,
    min_similarity_score,
    min_pruned_score,
):
    print("Query features: ")
    print_features(query_record["features"])
    similars = find_indices_similar_to_features(
        vectorizer,
        counter_matrix,
        [feature_list_to_doc(query_record)],
        num_similars,
        min_similarity_score,
    )
    candidate_records = []
    for (idx, score) in similars:
        pruned_record = prune_second_jd(query_record, records[idx])
        pruned_score = find_similarity_score_features(query_record, pruned_record)
        if pruned_score > min_pruned_score:
            candidate_records.append((records[idx], score, pruned_record, pruned_score))
    candidate_records = sorted(candidate_records, key=lambda v: v[3], reverse=True)
    logging.info(f"# of similar snippets = {len(candidate_records)}")
    return candidate_records


def cluster_and_intersect(
    query_record, candidate_records, top_n, threshold1, threshold2
):
    clustered_records = []
    if len(candidate_records) > 0:
        if config.CLUSTER:
            clustered_records = get_completions3(
                query_record, candidate_records, top_n, threshold1, threshold2
            )
    return clustered_records


def print_similar_and_completions(query_record, records, vectorizer, counter_matrix):
    candidate_records = find_similar(
        query_record,
        records,
        vectorizer,
        counter_matrix,
        config.NUM_SIMILARS,
        config.MIN_SIMILARITY_SCORE,
        config.MIN_PRUNED_SCORE,
    )
    print_match_index(query_record, candidate_records)
    clustered_records = cluster_and_intersect(
        query_record,
        candidate_records,
        config.TOP_N,
        config.THRESHOLD1,
        config.THRESHOLD2,
    )

    print(
        f"################ query code ################ index = {query_record['index']}"
    )
    print(ast_to_code(query_record["ast"]))
    if query_record["index"] >= 0:
        print("---------------- extracted from ---------------")
        print(ast_to_code(records[query_record["index"]]["ast"]))

    for clustered_record in clustered_records:
        print(
            f"------------------- suggested code completion ------------------"
        )  # idxs = ({clustered_record[1:]}), score = {candidate_records[clustered_record[1]][3]}")
        print(
            ast_to_code_with_full_lines(
                clustered_record[0]["ast"], clustered_record[1]["ast"]
            )
        )

    if config.PRINT_SIMILAR:
        j = 0
        for (candidate_record, score, pruned_record, pruned_score) in candidate_records:
            print(
                f"idx = {j}:------------------- similar code ------------------ index = {candidate_record['index']}, score = {score}"
            )
            print(ast_to_code(candidate_record["ast"]))
            print(
                f"------------------- similar code (pruned) ------------------ score = {pruned_score}"
            )
            print(ast_to_code(pruned_record["ast"]))
            j += 1
    print("", flush=True)


def collect_features_as_list(ast, is_init, is_counter):
    feature_list = []
    leaf_features = []
    leaf_pair_features = dict()
    global leaf_idx
    leaf_idx = 0
    collect_features_aux(
        ast,
        feature_list,
        [],
        [],
        dict(),
        leaf_features,
        leaf_pair_features,
        is_init,
        is_counter,
    )
    if is_counter:
        return (leaf_features, leaf_pair_features)
    else:
        return (feature_list, None)


def read_and_featurize_record_file(rpath):
    with open(rpath, "r") as inp:
        for line in inp:
            obj = json.loads(line)
            obj["features"] = collect_features_as_list(obj["ast"], False, False)[0]
            obj["index"] = -1
            return obj


def test_record_at_index(idx):
    record = get_record_part(records[idx])
    if record != None:
        print_similar_and_completions(record, records, vectorizer, counter_matrix)


def featurize_and_test_record(record_files, keywords):
    set_tmp = None
    record_final = None
    for record_file in record_files:
        record = read_and_featurize_record_file(record_file)
        if record is not None:
            record_final = record
            if set_tmp is not None:
                set_tmp = set_tmp & Counter(record["features"])
            else:
                set_tmp = Counter(record["features"])
            # need to figure out how to merge asts as well
    if set_tmp is None:
        set_tmp = Counter()
    for keyword in keywords:
        set_tmp[vocab.get_index(keyword)] += 1
    if record_final is None:
        record_final = {"ast": None, "index": -1, "features": list(set_tmp.elements())}
    else:
        record_final["features"] = list(set_tmp.elements())
    if len(record_final["features"]) > 0:
        print_similar_and_completions(record_final, records, vectorizer, counter_matrix)


def test_all():
    N = config.N_SAMPLES
    (sampled_indices, sampled_records) = sample_n_records(records, N)
    for k, record in enumerate(sampled_records):
        print(f"{k}: ", end="")
        print_similar_and_completions(record, records, vectorizer, counter_matrix)


def load_all(counter_path, asts_path):
    with open(counter_path, "rb") as outf:
        (vectorizer, counter_matrix) = pickle.load(outf)
    records = read_all_records(asts_path)
    logging.info("Read all records.")
    return (vectorizer, counter_matrix, records)


def setup(records_file):
    global config
    global vocab

    config = Config()
    logging.basicConfig(level=logging.DEBUG)
    random.seed(config.SEED)
    os.makedirs(options.working_dir, exist_ok=True)

    if records_file is None:
        vocab = Vocab.load()
    else:
        vocab = Vocab.load(True)
        featurize_records_file(
            records_file, os.path.join(options.working_dir, config.FEATURES_FILE)
        )
        vocab.dump()
        logging.info("Done featurizing.")
        counter_vectorize(
            os.path.join(options.working_dir, config.FEATURES_FILE),
            os.path.join(options.working_dir, config.TFIDF_FILE),
        )
        logging.info("Done computing counter matrix.")


logging.basicConfig(level=logging.DEBUG)
options = parse_args()
setup(options.corpus)

(vectorizer, counter_matrix, records) = load_all(
    os.path.join(options.working_dir, config.TFIDF_FILE),
    os.path.join(options.working_dir, config.FEATURES_FILE),
)

if options.index_query is not None:
    test_record_at_index(options.index_query)
elif len(options.file_query) > 0 or len(options.keywords) > 0:
    featurize_and_test_record(options.file_query, options.keywords)
elif options.testall:
    test_all()
