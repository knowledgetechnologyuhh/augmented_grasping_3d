import os
import jsonpickle

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory += os.sep if not directory[-1] == os.sep else ''
    return directory

def write_line(fout, lst):
    fout.write('\t'.join([str(x) for x in lst]) + '\n')


def load_csv(fn):
    ret = []
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            ret.append(lln)
    return ret

def store_csv(csv, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for x in csv:
            write_line(fout, x)

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

