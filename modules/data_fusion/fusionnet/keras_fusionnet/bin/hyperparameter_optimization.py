#!/usr/bin/python3
# performing hyperparameter optimization on Fusion net, to minimize validation perplexity
# Note that each run takes about 4 hours on a GTX 1080. setting max_eval to 100 would need 400 hours
import os
import pickle
import re
import sys
from copy import deepcopy

import argparse
import hyperopt
import numpy as np

from keras_fusionnet.utils.helper import make_dir
import keras_fusionnet.bin.train as fusionnet_train

HYPEROPT_POST_PROC = 'HYPEROPT_POST_PROC'

class Optimizer(object):
    def __init__(self, args):
        self.hyperopt_identifier = 101

        # the default experiment of choice
        self.experiment_name = args.hyperopt.experiment_name

        # optimizer algorithm
        self.optimizer_algo = args.hyperopt.optimizer

        # the output directory
        self.output_trial_directory = args.hyperopt.hyperopt_log_dir

        # maximum number of experiments to evaluate
        self.num_max_evals = args.hyperopt.max_evals_number

        # number of iterations before saving a new hyperopt log dump
        self.save_interval = args.hyperopt.save_interval

        # number of trials per training hyper-param combination
        self.num_trials = args.hyperopt.trials_number

        # parse arguments
        self.parsed_fusionnet_args = args

    def objective_fusionnet(self, optimization_params):
        # post process a hyper-parameter
        optimization_params_modified = deepcopy(optimization_params)
        postprocessed_params = dict()
        for optimization_param_key, optimization_param_val in list(optimization_params_modified.items()):
            if optimization_param_key[0] == HYPEROPT_POST_PROC:

                if 'output_filter' in optimization_param_key:
                    postprocessed_params[(*optimization_param_key[1:],)] = [
                        'retinanet_regression:' + str(optimization_param_val[0]),
                        'retinanet_classification:' + str(optimization_param_val[1]),
                        'transformer_classification:' + str(optimization_param_val[2]),
                        'fusionnet_regression:' + str(optimization_param_val[3])]
                del optimization_params_modified[optimization_param_key]

        optimization_params_modified.update(postprocessed_params)

        # add experiment name to the arguments
        # optimization_params_modified.update({('experiment_key',): self.experiment_name})
        # optimization_params_modified.update({('comet_project_name',): self.experiment_name})

        # add experiment tags and include the experiment name in the arguments
        if self.experiment_name == 'fusionnet_weights_exp':
            optimization_params_modified.update({('experiment_tag',):
                                                     self.parsed_fusionnet_args.experiment_tag + '_' +
                                                     '_'.join(str(float("{0:.2f}".format(weight)))
                                                              for weight in optimization_params.get(
                                                         (HYPEROPT_POST_PROC, 'fusionnet','output_filter')))})

        cumulative_training_report = list()
        average_training_report = dict()

        # begin training for assigned number of trials
        for trial in range(self.num_trials):
            training_report = fusionnet_train.main(deepcopy(self.parsed_fusionnet_args), optimization_params_modified)
            cumulative_training_report.append(deepcopy(training_report))

        required_keys = ['best_val_loss', 'best_train_loss', 'best_epoch']

        for report in cumulative_training_report:
            report_item = report
            for required_key in required_keys:
                if required_key in average_training_report:
                    average_training_report[required_key] += report_item[required_key]
                else:
                    average_training_report[required_key] = report_item[required_key]

        for required_key in required_keys:
            average_training_report[required_key] = average_training_report[required_key] / self.num_trials

        if 'best_val_loss' in average_training_report and average_training_report['best_val_loss'] is not None:
            return {'loss': average_training_report['best_val_loss'], 'status': hyperopt.STATUS_OK,
                    # other data
                    'training_loss': average_training_report['best_train_loss'],
                    'epoch': average_training_report['best_epoch']}

        else:
            return {'status': hyperopt.STATUS_FAIL}

    def optimize_fusionnet(self):

        # architectures

        no_layers = None
        dense_original_architecture = [('dense_small', None, None), ('dense_small', None, None)]

        dense_small_architecture = [('dense', hyperopt.hp.choice('dense_s_l1_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_s_l1_activation', ['tanh', 'sigmoid', 'relu'])),
                                    ('dense', hyperopt.hp.choice('dense_s_l2_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_s_l2_activation', ['tanh', 'sigmoid', 'relu']))]

        dense_large_architecture = [('dense', hyperopt.hp.choice('dense_l_l1_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_l_l1_activation', ['tanh', 'sigmoid', 'relu'])),
                                    ('dense', hyperopt.hp.choice('dense_l_l2_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_l_l2_activation', ['tanh', 'sigmoid', 'relu'])),
                                    ('dense', hyperopt.hp.choice('dense_l_l3_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_l_l3_activation', ['tanh', 'sigmoid', 'relu'])),
                                    ('dense', hyperopt.hp.choice('dense_l_l4_size', np.arange(32, 180, dtype=int)),
                                     hyperopt.hp.choice('dense_l_l4_activation', ['tanh', 'sigmoid', 'relu']))]

        fusionnet_architectures = [no_layers] + [dense_original_architecture] + [dense_small_architecture] + \
                                  [dense_large_architecture]

        if self.experiment_name == 'fusionnet_weights_exp':
            space = {
                (HYPEROPT_POST_PROC, 'fusionnet', 'output_filter'): [ hyperopt.hp.uniform('opt_fusionnet_retinanet_regression', 0.01, 10),
                                                                      hyperopt.hp.uniform('opt_fusionnet_retinanet_classification', 0.01, 10),
                                                                      hyperopt.hp.uniform('opt_fusionnet_transformer_cassification', 0.01, 10),
                                                                      hyperopt.hp.uniform('opt_fusionnet_fusionnet_regression', 0.01, 10)]
            }
        elif self.experiment_name == 'fusionnet_architecture_exp':
            space = {
                # 'opt_lr_val': hyperopt.hp.uniform('opt_lr_val', 0.0001, 0.9),
                # 'opt_reducelr_plateau': hyperopt.hp.choice('opt_reducelr_plateau', [False, True]),
                ('fusionnet', 'node_number'): hyperopt.hp.choice('opt_fusionnet_nodes', np.arange(1, 6, dtype=int)),
                ('fusionnet', 'fusionnet_layers'): hyperopt.hp.choice('opt_fusionnet_layers', fusionnet_architectures)
            }

        trials = hyperopt.Trials()
        count_current_eval = 0

        if self.optimizer_algo == 'rand':
            algo = hyperopt.rand.suggest
        elif self.optimizer_algo == 'tpe':
            algo = hyperopt.tpe.suggest
        else:
            algo = hyperopt.rand.suggest
        print("hyperopt algorithm : ", self.optimizer_algo)

        while count_current_eval < self.num_max_evals:

            # find the latest trial for given hyperopt_identifier and set count_current_eval if any(not implemented yet)
            hyper_opt_files = [f for f in os.listdir(self.output_trial_directory)
                               if os.path.isfile(os.path.join(self.output_trial_directory, f))]
            rx = r'{0}_\d+_hyperopt_trials_dump.p'.format(str(self.hyperopt_identifier))
            regex = re.compile(rx)
            selected_files = list(filter(regex.search, hyper_opt_files))
            found = re.findall('_(\d+?)_hyperopt_trials_dump.p', ''.join(selected_files))
            # print(found)
            if found:
                found = list(map(int, found))
                count_current_eval = max(found)
                # print(count_current_eval)
                trials = pickle.load(
                    open(os.path.join(self.output_trial_directory, str(self.hyperopt_identifier) + "_" +
                                      str(count_current_eval) + "_hyperopt_trials_dump.p"), "rb"))

                # count_current_eval = len(trials.trials)
            count_current_eval += self.save_interval  # number of iterations before you write a new log file

            # Issue with rstate
            # best_model = hyperopt.fmin(self.objective_fusionnet, space, algo=algo, max_evals=count_current_eval + 1,
            #                            rstate=np.random.RandomState(self.hyperopt_identifier), trials=trials) # set max_evals=100
            #
            best_model = hyperopt.fmin(self.objective_fusionnet, space,
                                       algo=algo, max_evals=count_current_eval, trials=trials)  # set max_evals=100
            # save the trials
            pickle.dump(trials, open(os.path.join(self.output_trial_directory, str(self.hyperopt_identifier) + "_" +
                                                  str(count_current_eval) + "_hyperopt_trials_dump.p"),
                                     "wb"))  # str(len(trials.trials))
            count_current_eval += 1
            print(trials.trials)
            print(hyperopt.space_eval(space, best_model))
            trials.delete_all()

def additional_args():
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(add_help=False)
    # sub_parsers = parser.add_subparsers(help='Hyperoptimization of the fusion network')

    # fusionnet model arguments
    # hyperopt_parser = sub_parsers.add_parser('hyperopt')

    parser.add_argument('experiment_name', help='Name of the experiment to be conducted.')
    parser.add_argument('hyperopt_log_dir',
                        type=str, help='The trial dump directory.',
                        default="../../logs/hyperopt_final_results")

    parser.add_argument('--optimizer', help='The optimizer to be used. options:["tpe","rand"]. default is tpe.',
                                  type=str,
                                  default="tpe")
    parser.add_argument('--coordinate-number', help='The number of output coordinates.',
                                  type=int,
                                  default=14)
    parser.add_argument('--trials-number', help='Number of trials per single experiment.',
                        type=int,
                        default=3)
    parser.add_argument('--max-evals-number', help='The total number of experiments to conduct.',
                        type=int,
                        default=400)
    parser.add_argument('--save-interval', help='Number of experiments before storing the trial dump.',
                        type=int,
                        default=1)

    return {'hyperopt': [parser]}

if __name__ == '__main__':
    args = sys.argv[1:]
    fusionnet_args = fusionnet_train.parse_args(args, parents=additional_args())
    make_dir(fusionnet_args.hyperopt.hyperopt_log_dir)
    Optimizer(fusionnet_args).optimize_fusionnet()
