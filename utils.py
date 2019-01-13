from os.path import join
import os
import codecs
import json
import pickle
import numpy as np
from nltk.corpus import stopwords


def get_words(content, window=None, remove_stopwords=True):
    import re
    content = content.lower()
    r = re.compile(r'[a-z]+')
    words = re.findall(r, content)
    if remove_stopwords:
        stpwds = stopwords.words('english')
        words = [w for w in words if w not in stpwds]
    if window is not None:
        words = words[:window]
    return words


def subname_equal(n1, n2):
    return 1 if n1 == n2 else -1


def load_json_lines(fpath, fname):
    items = []
    with codecs.open(join(fpath, fname), 'r', encoding='utf-8') as rf:
        for line in rf:
            paper = json.loads(line)
            paper['title'] = paper['title'].lower()
            items.append(paper)
    return items


def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfdir, wfname):
    os.makedirs(wfdir, exist_ok=True)
    with codecs.open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)


def dump_data(obj, wfpath, wfname):
    os.makedirs(wfpath, exist_ok=True)
    with open(join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def scale_matrix(matrix):
    matrix -= np.mean(matrix, axis=0)
    size = matrix.shape[0]
    return np.array(matrix).reshape(size, size, 1)


def prec_at_top_withid(preds, labels, k):
    n = len(preds)
    hits = np.zeros(n, dtype=int)
    for i in range(n):
        # print(labels[i])
        # print(preds[i][:k])
        hits[i] = int(labels[i] in preds[i][:k])
    return hits.mean()
