import os
import jsonpickle
import numpy as np

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory += os.sep if not directory[-1] == os.sep else ''
    return directory

# TODO (fabawi) replace with configparser and move to config.py
def store_settings(store_object, json_file):
    # convert args to dict
    with open(json_file, 'w') as fobj:
        json_obj = jsonpickle.encode(store_object)
        fobj.write(json_obj)

# TODO (fabawi) replace with configparser and move to config.py
def load_settings(json_file):
    # convert args to dict
    with open(json_file, 'r') as fobj:
        json_obj = fobj.read()
        obj = jsonpickle.decode(json_obj)
    return obj


def write_line(fout, lst):
    fout.write('\t'.join([str(x) for x in lst]) + '\n')


def load_csv(fn):
    ret = []
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            ret.append(lln)
    return ret


def load_csv_g(fn):
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            yield lln


def store_csv(csv, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            write_line(fout, x)


def load_list(fn):
    with open(fn, encoding="utf-8") as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st


def store_list(st, ofn):
    with open(ofn, "w", encoding="utf-8") as fout:
        for k in st:
            fout.write(str(k) + "\n")

def pad_to_fixed(xs, tokens, fixed_len=999):
    longest =  fixed_len
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:, 0] = tokens.startid()
    for i, x in enumerate(xs):
        x = x[:fixed_len - 2]
        for j, z in enumerate(x):
            X[i, 1 + j] = tokens.id(z)
        X[i, 1 + len(x)] = tokens.endid()
    return X


def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len)) + 2, max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:, 0] = tokens.startid()
    for i, x in enumerate(xs):
        x = x[:max_len - 2]
        for j, z in enumerate(x):
            X[i, 1 + j] = tokens.id(z)
        X[i, 1 + len(x)] = tokens.endid()
    return X


def input_2_bool(input_element):
    if isinstance(input_element, str):
        if input_element in ["True", "true", "1", "t", "T"]:
            return True
        else:
            return False
    elif isinstance(input_element, bool):
        return input_element
    elif isinstance(input_element, int) or isinstance(input_element, float):
        if input_element >= 1:
            return True
        else:
            return False
    else:
        return False


def parenthesis_split(sentence, delimiter=" ", lparen="[", rparen="]"):
    nb_brackets=0
    sentence = sentence.strip(delimiter) # get rid of leading/trailing seps

    l=[0]
    for i,c in enumerate(sentence):
        if c==lparen:
            nb_brackets+=1
        elif c==rparen:
            nb_brackets-=1
        elif c==delimiter and nb_brackets==0:
            l.append(i)
        # handle malformed string
        if nb_brackets<0:
            raise Exception("Syntax error")

    l.append(len(sentence))
    # handle missing closing parentheses
    if nb_brackets>0:
        raise Exception("Syntax error")
    return([sentence[i:j].strip(delimiter) for i,j in zip(l,l[1:])])


def freq_dict_2_list(dt):
    return sorted(dt.items(), key=lambda d: d[-1], reverse=True)