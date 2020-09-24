import os
import numpy as np

from ..utils import helper

# TODO (fabawi): This class is to be removed soon
class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):    return self.t2id.get(x, 1)

    def token(self, x):    return self.id2t[x]

    def num(self):        return len(self.id2t)

    def startid(self):  return 2

    def endid(self):    return 3



def make_s2s_dict(fn=None, min_freq=5, delimiter=' ',lparen="[", rparen="]", dict_file=None, store_target_only=False):
    if dict_file is not None and os.path.exists(dict_file):
        print('loading', dict_file)
        lst = helper.load_list(dict_file)
        midpos = lst.index('<@@@>')
        itokens = TokenList(lst[:midpos])
        otokens = TokenList(lst[midpos + 1:])
        return itokens, otokens
    data = helper.load_csv(fn)
    wdicts = [{}, {}]
    for ss in data:
        for seq, wd in zip(ss, wdicts):
            for w in helper.parenthesis_split(seq, delimiter, lparen, rparen):
                wd[w] = wd.get(w, 0) + 1
    wlists = []
    for wd in wdicts:
        wd = helper.freq_dict_2_list(wd)
        wlist = [x for x, y in wd if y >= min_freq]
        wlists.append(wlist)
    print('seq 1 words:', len(wlists[0]))
    print('seq 2 words:', len(wlists[1]))
    itokens = TokenList(wlists[0])
    otokens = TokenList(wlists[1])
    if dict_file is not None:
        if store_target_only:
            helper.store_list(['<@@@>'] + wlists[1], dict_file)
        else:
            helper.store_list(wlists[0] + ['<@@@>'] + wlists[1], dict_file)
    return itokens, otokens


def make_s2s_data(fn=None, itokens=None, otokens=None, delimiter=' ', lparen="[", rparen="]", h5_file=None, max_len=200):
    import h5py

    if h5_file is not None and os.path.exists(h5_file):
        print('loading', h5_file)
        with h5py.File(h5_file) as dfile:
            X, Y = dfile['X'][:], dfile['Y'][:]
        return X, Y
    data = helper.load_csv_g(fn)
    Xs = [[], []]
    for ss in data:
        for seq, xs in zip(ss, Xs):
            xs.append(list(helper.parenthesis_split(seq, delimiter, lparen, rparen)))
    X, Y = helper.pad_to_longest(Xs[0], itokens, max_len), helper.pad_to_longest(Xs[1], otokens, max_len)
    if h5_file is not None:
        with h5py.File(h5_file, 'w') as dfile:
            dfile.create_dataset('X', data=X)
            dfile.create_dataset('Y', data=Y)
    return X, Y


def data_generator(fn, itokens, otokens, batch_size=64, delimiter=' ', lparen="[", rparen="]", max_len=999):
    Xs = [[], []]
    while True:
        for ss in helper.load_csv_g(fn):
            for seq, xs in zip(ss, Xs):
                xs.append(list(helper.parenthesis_split(seq, delimiter, lparen, rparen)))
            if len(Xs[0]) >= batch_size:
                X, Y = helper.pad_to_longest(Xs[0], itokens, max_len), helper.pad_to_longest(Xs[1], otokens, max_len)
                yield [X, Y], None
                Xs = [[], []]

