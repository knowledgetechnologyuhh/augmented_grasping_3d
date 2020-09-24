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

def _read_annotations(csv_reader, coordinate_number):
    """ Read annotations from the csv_reader.
    """
    # skip the first line in csv file since it's the header
    next(csv_reader)
    grasping_coordinates = []
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            command_id, initial_layout_id, final_layout_id, *grasping_coordinates[:coordinate_number], scene_id = row[:8 + coordinate_number]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be command_id,initial_layout_id,final_layout_id,'
                'initial_grasping_coordinates[:{}],final_grasping_coordinates[:{}],scene_id'.format(line, coordinate_number//2, coordinate_number//2)),
                       None)

        grasping_coordinates = list(map(float, grasping_coordinates))

        initial_grasping_coordinates = grasping_coordinates[:len(grasping_coordinates)//2]
        final_grasping_coordinates = grasping_coordinates[len(grasping_coordinates)//2:]

        random_id = random.randint(10000, 99999)
        result[str(scene_id) + '_' + str(random_id)] = [scene_id]

        # command_id = _parse(command_id, int, 'line {}: malformed command_id: {{}}'.format(line))
        # initial_layout_id = _parse(initial_layout_id, int, 'line {}: malformed initial_layout_id: {{}}'.format(line))
        # final_layout_id = _parse(final_layout_id, int, 'line {}: malformed final_layout_id: {{}}'.format(line))

        result[str(scene_id) + '_' + str(random_id)].append({'command_id': command_id,
                                 'initial_layout_id': initial_layout_id, 'final_layout_id': final_layout_id,
                                 'initial_grasping_coordinates': initial_grasping_coordinates,
                                 'final_grasping_coordinates': final_grasping_coordinates})
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


class FusionnetCSVGenerator(object):
    """ Abstract generator class.
    """

    def __init__(self,
                 csv_data_file,
                 base_dir=None,
                 min_max_joints_file=None,
                 batch_size=2,
                 group_method='ratio',  # one of 'none', 'random', 'ratio'
                 shuffle_groups=True,
                 submodule_generators=None,
                 output_filter=None,
                 coordinate_number=8,
                 coordinate_limits_file=None,
                 coordinate_scale_divisor=10  # this is the number by which the coordinates are divided, ignored if limits file provided
                 ):

        """ Initialize a CSV data generator.
         Args for csv
                    csv_data_file: Path to the CSV annotations file.
                    csv_class_file: Path to the CSV classes file.
                    base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).


        Args for generator
        """

        self.base_dir = base_dir
        self.scene_2_group = {}

        # take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # read csv
        try:
            with _open_for_csv(csv_data_file) as file:
                self.command_data = _read_annotations(csv.reader(file, delimiter=','), coordinate_number=coordinate_number)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.command_names = list(self.command_data.keys())

        self.submodule_generators = submodule_generators

        self.coordinate_scale_divisor = coordinate_scale_divisor
        self.coordinate_limits_file = coordinate_limits_file

        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.output_filter = output_filter
        self.group_index = 0
        self.lock = threading.Lock()

        self.group_inputs()

    def size(self):
        """ Size of the dataset.
        """
        return len(self.command_names)

    # TODO (fabawi): make load_target call preprocess_joints to resize the joints
    def preprocess_joints(self, joints):
        loaded_joint_extremes = np.load(self.coordinate_limits_file)
        min_joints = np.squeeze(loaded_joint_extremes['min'])
        max_joints = np.squeeze(loaded_joint_extremes['max'])
        temp_joints = joints
        # convert nan to 1
        joints_normalized = np.divide(
            np.subtract(temp_joints, min_joints), np.subtract(max_joints, min_joints))
        joints_normalized[np.isnan(joints_normalized)] = 1
        return joints_normalized

    def reverse_preprocess_joints(self, joints):
        loaded_joint_extremes = np.load(self.coordinate_limits_file)
        min_joints = np.squeeze(loaded_joint_extremes['min'])
        max_joints = np.squeeze(loaded_joint_extremes['max'])
        temp_joints = joints
        # convert nan to 1
        joints_unnormalized = np.multiply(temp_joints, np.subtract(max_joints, min_joints)) + min_joints
        joints_unnormalized[np.isnan(joints_unnormalized)] = 1
        return joints_unnormalized

    def load_target(self, command_index):
        target = []
        # scale joints by limits stored in file
        if self.coordinate_limits_file is not None:
            initial_grasping_coordinates = self.preprocess_joints(
                np.array(self.command_data[self.command_names[command_index]][1]['initial_grasping_coordinates']))
            final_grasping_coordinates = self.preprocess_joints(
                np.array(self.command_data[self.command_names[command_index]][1]['final_grasping_coordinates']))
        # or scale joints by a constant
        else:
            initial_grasping_coordinates = np.divide(
                np.array(self.command_data[self.command_names[command_index]][1]['initial_grasping_coordinates']),
                self.coordinate_scale_divisor)
            final_grasping_coordinates = np.divide(
                np.array(self.command_data[self.command_names[command_index]][1]['final_grasping_coordinates']),
                self.coordinate_scale_divisor)

        concatenated_target = np.concatenate([initial_grasping_coordinates, final_grasping_coordinates], axis=0)
        target.extend(concatenated_target)

        # target.append(np.array(self.command_data[self.command_names[command_index]][1]['initial_grasping_coordinates']))
        # target.append(np.array(self.command_data[self.command_names[command_index]][1]['final_grasping_coordinates']))
        return target

    def load_target_group(self, group):
        """ Load targets for all commands in a group.
        """
        return [self.load_target(target_index) for target_index in group]



    def group_inputs(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.input_length(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def input_length(self, input_index):
        command_id = self.command_names[input_index]
        # TODO (fabawi): just a dummy for now. Find a more meaningful criteria for ordering the input
        sequence_length = self.command_data[command_id][1]['initial_grasping_coordinates'][0]
        return sequence_length


    def compute_targets(self, target_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        output_batch = np.array(target_group)

        return [output_batch]

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        target_group = self.load_target_group(group)

        # get other inputs and outputs from submodules
        targets, submodule_inputs, submodule_targets = list(), list(), list()

        # compute network targets
        # TODO (fabawi): make this cleaner. The generator at some point will be unaware of the network names
        if any('fusionnet_regression' in filt for filt in self.output_filter):
            targets = self.compute_targets(target_group)

        if self.submodule_generators is not None:

            if 'vision_generator' in self.submodule_generators:
                vision_group = []
                for group_index in group:
                    layout_id = self.command_data[self.command_names[group_index]][1]['initial_layout_id']
                    vision_group.append(self.submodule_generators['vision_generator'].layout_2_group(layout_id))
                vision_inputs, vision_targets = \
                    self.submodule_generators['vision_generator'].compute_input_output(vision_group)
                # append single arrays, extend list for multiple arrays
                submodule_inputs.append(vision_inputs)
                # TODO (fabawi): make this cleaner. The generator at some point will be unaware of the network names
                if any('retinanet_regression' in filt for filt in self.output_filter):
                    submodule_targets.append(vision_targets[0])
                if any('retinanet_classification' in filt for filt in self.output_filter):
                    submodule_targets.append(vision_targets[1])

            if 'language_translation_generator' in self.submodule_generators:
                language_translation_group = []
                for group_index in group:
                    command_id = self.command_data[self.command_names[group_index]][1]['command_id']
                    language_translation_group.append(self.submodule_generators['language_translation_generator'].command_2_group(command_id))
                language_translation_inputs, language_translation_targets = \
                    self.submodule_generators['language_translation_generator'].compute_input_output(language_translation_group)
                submodule_inputs.extend(language_translation_inputs)
                # TODO (fabawi): make this cleaner. The generator at some point will be unaware of the network names
                if any('transformer_classification' in filt for filt in self.output_filter):
                    submodule_targets.append(language_translation_targets)

        inputs = submodule_inputs
        targets = submodule_targets + targets
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

        inputs_outputs = self.compute_input_output(group)
        return inputs_outputs
