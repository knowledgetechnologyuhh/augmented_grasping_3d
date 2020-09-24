"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@contribution: Fares Abawi (abawi@informatik.uni-hamburg.de)
"""

import numpy as np
import random
import threading
from six import raise_from
import csv
import sys
import os.path

from keras_transformer.utils.helper import (parenthesis_split, pad_to_fixed, freq_dict_2_list, load_list, store_list)



class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.t2id = {v: k for k, v in enumerate(self.id2t)}

    def id(self, x):    return self.t2id.get(x, 1)

    def token(self, x):    return self.id2t[x]

    def num(self):        return len(self.id2t)

    def startid(self):  return 2

    def endid(self):    return 3


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_annotations(csv_reader):
    """ Read annotations from the csv_reader.
    """
    # skip the first line in csv file since it's the header
    next(csv_reader)

    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            source, target, command_id = row[:3]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'source_sequence\ttarget_sequence\tcommand_id\''.format(line)),
                       None)

        if command_id not in result:
            result[command_id] = [command_id]

        source = _parse(source, str, 'line {}: malformed source sequence: {{}}'.format(line))
        target = _parse(target, str, 'line {}: malformed target sequence: {{}}'.format(line))

        result[command_id].append({'source': source, 'target': target})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class TransformerCSVGenerator(object):
    """ Abstract generator class.
    """

    def __init__(self,
                 csv_data_file,
                 golden_data_file=None,
                 base_dir=None,
                 batch_size=2,
                 group_method='ratio',  # one of 'none', 'random', 'ratio'
                 shuffle_groups=True,
                 sequence_max_length=100,
                 delimiter=" ",
                 lparen='[',
                 rparen=']',
                 min_word_count=1,
                 i_tokens=None,
                 o_tokens=None,
                 tokens_file=None,
                 i_embedding_matrix_file=None,
                 o_embedding_matrix_file=None
                 ):

        """ Initialize a CSV data generator.
         Args for csv
                    csv_data_file: Path to the CSV annotations file.
                    csv_class_file: Path to the CSV classes file.
                    base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).


        Args for generator
        """
        self.golden_data_file = golden_data_file
        self.base_dir = base_dir
        self.command_2_group_table = {}

        # take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.command_data = _read_annotations(csv.reader(file, delimiter='\t'))
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.command_names = list(self.command_data.keys())


        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.sequence_max_length = sequence_max_length

        self.group_index = 0
        self.lock = threading.Lock()

        self.lparen = lparen
        self.rparen = rparen
        self.delimiter = delimiter

        if i_tokens is not None and o_tokens is not None:
            self.i_tokens, self.o_tokens = i_tokens, o_tokens
        else:
            self.i_tokens, self.o_tokens = self.create_lookup_table(min_freq=min_word_count, tokens_file=tokens_file)
        self.group_sources()
        self.command_lookup()

        self.o_embedding_matrix = None
        self.i_embedding_matrix = None
        if i_embedding_matrix_file is not None:
            self.i_embedding_matrix = self.load_embedding(i_embedding_matrix_file)
        if i_embedding_matrix_file is not None:
            self.o_embedding_matrix = self.load_embedding(o_embedding_matrix_file)

    def load_embedding(self, embedding_file):
        return np.load(embedding_file)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.command_names)

    def get_source_sequence(self, command_index=None, command=None):
        if command_index is not None:
            sequence = self.command_data[self.command_names[command_index]][1]['source']
        elif command is not None:
            sequence = command
        else:
            raise_from(ValueError('provide a command_index or a command_filename. command_index takes precedence'), None)
        xs = parenthesis_split(sequence, self.delimiter, self.lparen, self.rparen)
        return xs

    def load_source(self, command_index):
        xs = []
        sequence = self.command_data[self.command_names[command_index]][1]['source']
        xs.append(list(parenthesis_split(sequence, self.delimiter, self.lparen, self.rparen)))
        source = pad_to_fixed(xs, self.i_tokens, self.sequence_max_length).squeeze()
        return source

    def load_target(self, command_index):
        xt = []
        sequence = self.command_data[self.command_names[command_index]][1]['target']
        xt.append(list(parenthesis_split(sequence, self.delimiter, self.lparen, self.rparen)))
        target = pad_to_fixed(xt, self.o_tokens, self.sequence_max_length).squeeze()
        return target

    def load_source_group(self, group):
        """ Load sources for all commands in a group.
        """
        sources_group = [self.load_source(source_index) for source_index in group]

        return sources_group


    def load_target_group(self, group):
        """ Load targets for all commands in a group.
        """
        return [self.load_target(target_index) for target_index in group]


    def preprocess_group_entry(self, source, target):
        """ TODO (fabawi): mutate the text here
        """
        # preprocess the text

        return source, target

    def preprocess_group(self, source_group, target_group):
        """ Preprocess each source and target sequence.
        """
        for index, (source, target) in enumerate(zip(source_group, target_group)):
            # preprocess a single group entry
            source, target = self.preprocess_group_entry(source, target)

            # copy processed data back to group
            source_group[index] = source
            target_group[index] = target

        return source_group, target_group

    def group_sources(self):
        """ Order the sequences according to the source length.
        """
        # determine the order of the sequences
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.source_length(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def source_length(self, source_index):
        command_id = self.command_names[source_index]
        seq = self.command_data[command_id][1]['source']
        seq_length = parenthesis_split(seq, self.delimiter, self.lparen, self.rparen)
        return seq_length

    def create_lookup_table(self, min_freq=1, tokens_file=None):
        if tokens_file is not None and os.path.exists(tokens_file):
            print('loading', tokens_file)
            lst = load_list(tokens_file)
            midpos = lst.index('<@@@>')
            itokens = TokenList(lst[:midpos])
            otokens = TokenList(lst[midpos + 1:])
            return itokens, otokens

        data = self.command_data
        wdicts = [{}, {}]
        for key, ss in data.items():
            for seq, wd in zip([ss[1]['source'], ss[1]['target']], wdicts):
                for w in parenthesis_split(seq, self.delimiter, self.lparen, self.rparen):
                    wd[w] = wd.get(w, 0) + 1
        wlists = []
        for wd in wdicts:
            wd = freq_dict_2_list(wd)
            wlist = [x for x, y in wd if y >= min_freq]
            wlists.append(wlist)
        i_tokens = TokenList(wlists[0])
        o_tokens = TokenList(wlists[1])

        if tokens_file is not None:
            store_list(wlists[0] + ['<@@@>'] + wlists[1], tokens_file)
        return i_tokens, o_tokens

    def command_lookup(self):
        for idx, (_, data) in enumerate(self.command_data.items()):
            if data[0] in self.command_2_group_table:
                self.command_2_group_table[data[0]].extend([idx])
            else:
                self.command_2_group_table[data[0]] = [idx]

    def command_2_group(self, layout_id):
        """ Randomly gets a command from the group if more than one exists
        """
        random.shuffle(self.command_2_group_table[layout_id])
        return self.command_2_group_table[layout_id][0]

    def compute_inputs(self, source_group, target_group):
        """ Compute inputs for the network using the source_group and target_group.
        """
        input_batch = []
        input_batch.append(np.array(source_group))
        input_batch.append(np.array(target_group))
        return input_batch

    def compute_targets(self, target_group):
        """ Compute target outputs for the network.
        """
        output_batch = np.array(target_group)

        return output_batch

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load source and target sequences.
        source_group = self.load_source_group(group)
        target_group = self.load_target_group(group)

        # perform preprocessing steps
        source_group, target_group = self.preprocess_group(source_group, target_group)

        # compute network inputs
        inputs = self.compute_inputs(source_group, target_group)

        # compute network targets
        targets = self.compute_targets(target_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)
