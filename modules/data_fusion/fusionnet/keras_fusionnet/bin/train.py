import argparse
import os
import sys
import gc
from copy import copy
import warnings
from functools import reduce  # forward compatibility for Python 3
import operator

from comet_ml import Experiment
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import keras.preprocessing.image
import tensorflow as tf

from keras_fusionnet.utils.keras_version import check_keras_version
from keras_fusionnet.utils.helper import make_dir
from keras_fusionnet.utils.config import write_config_file

from keras_fusionnet.models.fusionnet import FusionNet
from keras_fusionnet.models.model_setup import create_vision_objects, create_translation_objects
from keras_fusionnet.models.model_setup import create_extra_fusionnet_args, create_extra_vision_args, create_extra_translation_args
from keras_fusionnet.models.model_setup import create_fusionnet_model_config, create_vision_model_config, create_translation_model_config
from keras_fusionnet.preprocessing.generator_setup import create_fusionnet_generators, create_vision_generators, create_translation_generators
from keras_fusionnet.losses_setup import fusionnet_model_losses, vision_model_losses, translation_model_losses
from keras_fusionnet.losses_setup import fusionnet_model_metrics, vision_model_metrics, translation_model_metrics
from keras_fusionnet.callbacks import RedirectModel
from keras_fusionnet.callbacks.eval_setup import fusionnet_model_callback, vision_model_callback, translation_model_callback


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.train_losses = []
        self.epochs = 0
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")

    def on_epoch_end(self, batch, logs={}):
        if 'fusionnet_regression_loss' in logs.keys() and 'val_fusionnet_regression_loss' in logs.keys():
            self.train_losses.append(logs.get('fusionnet_regression_loss'))
            self.val_losses.append(logs.get('val_fusionnet_regression_loss'))
        else:
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

        self.epochs += 1
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_train_loss = self.train_losses[-1]
            self.best_epoch = self.epochs

def get_session():
    """ Construct a modified tf session.
    """
    # sess = K.get_session()
    # K.clear_session()
    # sess.close()
    tf.reset_default_graph()
    config = tf.ConfigProto()  # inter_op_parallelism_threads=1
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(snapshot, multi_gpu, batch_size, fusionnet_model_config, vision_model_config, translation_model_config, log_path=None):
    """ Creates three models (model, training_model, prediction_model).    """

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model

    fusionnet = FusionNet(batch_size, fusionnet_model_config, vision_model_config, translation_model_config, log_path)
    fusion_model = fusionnet.create_model()

    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = fusion_model
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = fusion_model
        training_model = model

    # load snapshot for the entire model
    if snapshot is not None:
        print('Loading model, this may take a second...')
        model.load_weights(snapshot, by_name=True, skip_mismatch=True)

    # load pretrained snapshots for all submodels individually
    fusionnet.load_model_specific_weights()

    # make prediction models
    prediction_models = fusionnet.create_prediction_models()

    losses = {}
    loss_weights = {}
    metrics = {}
    for output in fusionnet_model_config['output_filter']:

        loss, weight = vision_model_losses(output, vision_model_config)
        losses.update(loss)
        loss_weights.update(weight)
        loss, weight = translation_model_losses(output, translation_model_config)
        losses.update(loss)
        loss_weights.update(weight)
        loss, weight = fusionnet_model_losses(output, fusionnet_model_config)
        losses.update(loss)
        loss_weights.update(weight)
        metrics.update(vision_model_metrics(output, vision_model_config))
        metrics.update(translation_model_metrics(output, translation_model_config))
        metrics.update(fusionnet_model_metrics(output, fusionnet_model_config))

    # compile model
    training_model.compile(
        loss= losses,
        metrics=metrics,
        loss_weights=loss_weights,
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)  # fabawi: change the learning rate to 1e-3 from 1e-5 to accomodate the change in the anchors
    )

    return model, training_model, prediction_models


def create_callbacks(history, callback_objects, model, prediction_models, validation_generators, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    # extracting tensorboard and comet experiment
    callbacks = []

    tensorboard_callback = None
    comet_experiment = None
    if 'tensorboard' in callback_objects:
        tensorboard_callback = callback_objects['tensorboard']
        callbacks.append(tensorboard_callback)
    if 'comet' in callback_objects:
        comet_experiment = callback_objects['comet']

    csv_logger = CSVLogger(os.path.join(args.log_path,'results.csv'), append=True)
    callbacks.append(csv_logger)

    if args.fusionnet.evaluation and validation_generators['fusionnet']:
        fusionnet_callback = fusionnet_model_callback(generator=validation_generators['fusionnet'],
                                               model=prediction_models['fusionnet'],
                                               tensorboard_callback=tensorboard_callback,
                                               comet_ml_experiment=comet_experiment,
                                               args=args.fusionnet)
        if fusionnet_callback is not None: callbacks.append(fusionnet_callback)

    if args.language_translation.evaluation and validation_generators['language_translation']:
        translation_callback = translation_model_callback(generator=validation_generators['language_translation'],
                                               model=prediction_models['language_translation'],
                                               tensorboard_callback=tensorboard_callback,
                                               comet_ml_experiment=comet_experiment,
                                               args=args.language_translation)
        if translation_callback is not None: callbacks.append(translation_callback)

    if args.vision.evaluation and validation_generators['vision']:
        vision_callback = vision_model_callback(generator=validation_generators['vision'],
                                               model=prediction_models['vision'],
                                               tensorboard_callback=tensorboard_callback,
                                               comet_ml_experiment=comet_experiment,
                                               args=args.vision)
        if vision_callback is not None: callbacks.append(vision_callback)

    # save the model
    if args.snapshots:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type='fusionnet')
            ),
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            # save_best_only=True,
            # monitor="loss",
            # mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)


    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    loss_history = history
    callbacks.append(loss_history)

    return callbacks


def create_generators(args, extra_args):
    """ Create generators for training and validation.

    Args
        args : parseargs object containing configuration for generators.
        extra_args : dictionary containing extra arguments for generators.
    """
    if args.module_type == 'vision':
        train_generator, validation_generator = create_vision_generators(args, extra_args)

    elif args.module_type == 'language_translation':
        train_generator, validation_generator = create_translation_generators(args, extra_args)

    elif args.module_type == 'fusionnet':
        train_generator, validation_generator = create_fusionnet_generators(args, extra_args)

    else:
        train_generator, validation_generator = None, None

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    return parsed_args

def multispaces_parse_args(parser, commands, args):
    # Divide argv by commands
    split_argv = [[]]
    for c in args:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args

def parse_args(args, parents=None):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Simple training script for training a hierarchical multitask network.')
    sub_parsers = parser.add_subparsers(help='Arguments for specific models.', dest='module_type')

    vision_parsers = sub_parsers.add_parser('vision').add_subparsers(help='Choose a specific vision networks.', dest='vision_module_type')
    language_translation_parsers = sub_parsers.add_parser('language_translation').add_subparsers(help='Choose a specific language translation networks.', dest='translation_module_type')

    def csv_list(string):
        return string.split(',')

    # add subparsers from external resources
    if parents is not None:
        for key, val in parents.items():
            sub_parsers.add_parser(key, parents=val)

    # fusionnet model arguments
    fusionnet_parser = sub_parsers.add_parser('fusionnet')

    fusionnet_parser.add_argument('--output-filter', help='A list of output layers to be included in the learning procedure'
                                                          '(Only certain layers are allowed: by default all are allowed). '
                                                          'Optionally each output can have a weight associated with it by '
                                                          'following the weight with a colon and a weight factor ranging between'
                                                          '0 and 1',
                            type=csv_list, default='retinanet_regression:1,retinanet_classification:1,'
                                                   'transformer_classification:1,'
                                                   'fusionnet_regression:1')

    # fusionnet_parser.add_argument('--output-filter',
    #                               help='A list of output layers to be included in the learning procedure'
    #                                    '(Only certain layers are allowed: by default all are allowed).',
    #                               type=csv_list, default='fusionnet_regression')

    fusionnet_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    fusionnet_parser.add_argument('--val-annotations',
                                    help='Path to CSV file containing annotations for validation (optional).')
    fusionnet_parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                                  action='store_false')
    fusionnet_parser.add_argument('--coordinate-limits', help='Path to NPZ file containing joint limits (optional)',
                                  default=None)
    fusionnet_parser.add_argument('--coordinate-number', help='The number of output coordinates.',
                                    type=int,
                                    default=14)
    fusionnet_parser.add_argument('--node-number', help='The number of dimensions for the nodes connecting the input modalities',
                                  type=int,
                                  default=4)
    fusionnet_parser.add_argument('--fusionnet-layers',
                                  help='The fusionnet layers represented as a list. The list should be wrapped inside a string',
                                  default=None)

    fusionnet_parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    # retinanet model arguments
    retinanet_parser = vision_parsers.add_parser('retinanet')

    retinanet_parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    retinanet_parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')

    retinanet_parser.add_argument('--imagenet-weights',
                       help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
                       action='store_const', const=True, default=True)
    retinanet_parser.add_argument('--weights', help='Initialize the model with weights from a file.')
    retinanet_parser.add_argument('--no-weights', help='Don\'t initialize the model with any weights.', dest='imagenet_weights',
                       action='store_const', const=False)
    retinanet_parser.add_argument('--pretrained-snapshot',
                                  help='Initialize the model with weights from a pretrained model file. '
                                       'This overrides the snapshot for the overall model')
    retinanet_parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')

    retinanet_parser.add_argument('--random-transform', help='Randomly transform image and annotations.',
                                  action='store_true')
    retinanet_parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',
                                  type=int,
                                  default=480)
    retinanet_parser.add_argument('--image-max-side',
                                  help='Rescale the image if the largest side is larger than max_side.',
                                  type=int, default=640)
    retinanet_parser.add_argument('--weighted-average',
                        help='Compute the mAP using the weighted average of precisions among classes.',
                        action='store_true')

    retinanet_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    retinanet_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    retinanet_parser.add_argument('--val-annotations',
                            help='Path to CSV file containing annotations for validation (optional).')
    retinanet_parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    # simple cnn vision model arguments
    simple_cnn_parser = vision_parsers.add_parser('simple_cnn')
    simple_cnn_parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                                  action='store_false')

    simple_cnn_parser.add_argument('--random-transform', help='Randomly transform image and annotations.',
                                   action='store_true')
    simple_cnn_parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',
                                   type=int,
                                   default=480)
    simple_cnn_parser.add_argument('--image-max-side',
                                   help='Rescale the image if the largest side is larger than max_side.',
                                   type=int, default=640)

    simple_cnn_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    simple_cnn_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    simple_cnn_parser.add_argument('--val-annotations',
                                   help='Path to CSV file containing annotations for validation (optional).')

    # transformer model arguments
    transformer_parser = language_translation_parsers.add_parser('transformer')

    transformer_parser.add_argument('--pretrained-snapshot',
                                  help='Initialize the model with weights from a pretrained model file. '
                                       'This overrides the snapshot for the overall model')
    transformer_parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                                  action='store_false')

    transformer_parser.add_argument('--sequence-max-length', help='The maximum length for all sequences.',
                                  type=int,
                                  default=200)

    transformer_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    transformer_parser.add_argument('--vocab', help='Path to an already existing vocabulary file (optional).')
    transformer_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    transformer_parser.add_argument('--val-golden-set', help='Path to the human annotated golden set for validation.')
    transformer_parser.add_argument('--i-embedding-matrix',
                        help='Path to an already existing pretrained source embedding (optional).',default=None)
    transformer_parser.add_argument('--o-embedding-matrix',
                        help='Path to an already existing pretrained target embedding (optional).', default=None)
    transformer_parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    parser.add_argument('--batch-size',       help='Size of the batches.', default=2, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=5000)
    parser.add_argument('--val-steps',        help='Number of steps for validation.', type=int, default=700)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'../../snapshots\')', default='../../snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output  (defaults to \'../../logs/tensorboard\')') # default='../../logs/tensorboard'
    parser.add_argument('--log-path',         help='Path to store the logs (results, configurations and evaluation temporary files) (defaults to \'../../logs\')', default='../../logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')

    parser.add_argument('--experiment-tag', help='A tag to identify the experiment by.', type=str, default='DEFAULT_EXPERIMENT_TAG')
    parser.add_argument('--experiment-key', help='A tag to identify the experiment by.', type=str, default='DEFAULT_EXPERIMENT_KEY')

    parser.add_argument('--comet-api-key', help='The comet-ml api key.', type=str)
    parser.add_argument('--comet-project-name', help='The comet-ml project name.', type=str)
    parser.add_argument('--comet-workspace', help='The comet-ml workspace.', type=str)

    return check_args(multispaces_parse_args(parser, sub_parsers, args))


def update_args(args_dict, targeted_key, updated_value):
    # update the general arguments first
    args_dict[targeted_key] = updated_value
    # update module specific arguments
    args_dict_module_specific = {key: val for key, val in args_dict.items()
                           if isinstance(val, argparse.Namespace)}
    for key, val in args_dict_module_specific.items():
        if targeted_key in args_dict_module_specific[key]:
            vars(args_dict_module_specific[key])[targeted_key] = updated_value

def merge_hyperparam_args(args, hyperparam_args):
    args_dict = vars(args)
    args_dict_module_specific = {key: val for key, val in args_dict.items()
                                 if isinstance(val, argparse.Namespace)}
    for hyperparam_arg_name, hyperparam_val in hyperparam_args.items():
        if len(hyperparam_arg_name) > 1:
            if hyperparam_arg_name[0] in args_dict_module_specific and \
                    hyperparam_arg_name[1] in args_dict_module_specific[hyperparam_arg_name[0]]:
                vars(args_dict_module_specific[hyperparam_arg_name[0]])[hyperparam_arg_name[1]] = hyperparam_val
        else:
            args_dict[hyperparam_arg_name[0]] = hyperparam_val
    return args

def main(args=None, hyperparam_args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    if hyperparam_args is None:
        args = parse_args(args)
    else:
        args = merge_hyperparam_args(args, hyperparam_args)

    # create callback objects for the evaluation
    callback_objects = dict()

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callback_objects['tensorboard'] = tensorboard_callback

    if args.comet_api_key:
        comet_experiment = Experiment(api_key=args.comet_api_key,
                                      project_name=args.comet_project_name, workspace=args.comet_workspace, disabled=False)
        comet_experiment.add_tag(args.experiment_tag)
        comet_experiment.set_name(args.experiment_tag)
        callback_objects['comet'] = comet_experiment
        # get the experiment key from comet and replace the one passed throught the arguments
        args.experiment_key = comet_experiment.get_key()

    # get the dictionary of the arguments
    args_dict = vars(args)

    # create all the directories and update args as well
    update_args(args_dict, 'snapshot_path', make_dir(os.path.join(args.snapshot_path, args.experiment_key)))
    update_args(args_dict, 'log_path', make_dir(os.path.join(args.log_path, args.experiment_key)))
    make_dir(os.path.join(args.log_path, 'evaluation'))
    make_dir(os.path.join(args.log_path, 'prediction'))
    # careful! if a global argument is actually passed it will not be updated for all modules. They will take the default value instead. Hence:
    update_args(args_dict, 'batch_size', args.batch_size)
    if args.tensorboard_dir:
        update_args(args_dict, 'tensorboard_dir', make_dir(os.path.join(args.tensorboard_dir, args.experiment_key)))


    # save the arguments
    write_config_file(args, os.path.join(args.log_path,'arguments.json'))

    # save the arguments to comet if available
    if args.comet_api_key:
        for arg_key, arg_val in args_dict.items():
            if isinstance(arg_val, argparse.Namespace):
                comet_experiment.log_parameters(vars(arg_val),arg_key)
            else:
                comet_experiment.log_parameter(arg_key, arg_val)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    session = get_session()
    keras.backend.tensorflow_backend.set_session(session)

    # create the generators
    validation_generators = {}
    training_generators = {}

    # create the vision model objects
    vision_objects = create_vision_objects(args.vision)
    # add extra arguments
    extra_vision_args = create_extra_vision_args(args.vision, vision_objects)
    # create the vision generators
    vision_train_generator, vision_validation_generator = create_generators(args.vision, extra_vision_args)
    training_generators['vision'], validation_generators['vision'] = vision_train_generator, vision_validation_generator
    # set the vision model configuration
    vision_model_config = create_vision_model_config(args.vision, vision_objects, vision_train_generator)

    # create the translation model objects
    translation_objects = create_translation_objects(args.language_translation)
    # add extra arguments
    extra_translation_args = create_extra_translation_args(args.language_translation, translation_objects)
    # create the translation generators
    translation_train_generator, translation_validation_generator = create_generators(args.language_translation,
                                                                                 extra_translation_args)
    training_generators['language_translation'], validation_generators['language_translation'] = \
        translation_train_generator, translation_validation_generator
    # set the translation model configuration
    translation_model_config = create_translation_model_config(args.language_translation, translation_objects,
                                                               translation_train_generator)

    # create the fusionnet model objects
    fusionnet_objects = {'vision_train_generator': vision_train_generator,
                         'translation_train_generator': translation_train_generator,
                         'vision_val_generator': vision_validation_generator,
                         'translation_val_generator': translation_validation_generator}
    # add extra arguments
    extra_fusionnet_args = create_extra_fusionnet_args(args.fusionnet, fusionnet_objects)
    # create the fusionnet generators
    fusionnet_train_generator, fusionnet_validation_generator = create_generators(args.fusionnet, extra_fusionnet_args)
    training_generators['fusionnet'], validation_generators['fusionnet'] = fusionnet_train_generator, fusionnet_validation_generator
    # set the fusionnet model configuration
    fusionnet_model_config = create_fusionnet_model_config(args.fusionnet, fusionnet_objects, fusionnet_train_generator)

    # create the model
    print('Creating model, this may take a second...')
    general_model, training_model, prediction_models = create_models(
        snapshot=args.snapshot,
        multi_gpu=args.multi_gpu,
        batch_size=args.batch_size,
        fusionnet_model_config=fusionnet_model_config,
        vision_model_config=vision_model_config,
        translation_model_config=translation_model_config,
        log_path=args.log_path
    )

    # print model summary
    print(general_model.summary())

    # TODO (fabawi): move backbone shape computation to somewhere else
    # # this lets the generator compute backbone layer shapes using the actual backbone model
    # if 'vgg' in args.backbone or 'densenet' in args.backbone:
    #     vision_train_generator.compute_shapes = make_shapes_callback(model)
    #     if vision_validation_generator:
    #         vision_validation_generator.compute_shapes = vision_train_generator.compute_shapes
    #

    # create the callbacks
    loss_history = LossHistory()
    callbacks = create_callbacks(
        loss_history,
        callback_objects,
        general_model,
        prediction_models,
        validation_generators,
        args,
    )

    # start training
    training_model.fit_generator(
        generator=fusionnet_train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        validation_data=fusionnet_validation_generator,
        validation_steps=args.val_steps,
        verbose=1,
        callbacks=callbacks,
        use_multiprocessing=False
    )

    loss_history = copy(loss_history)

    # delete model to free up memory and cleanup
    if 'comet' in callback_objects:
        callback_objects['comet'].clean()
        callback_objects['comet'].end()
    K.clear_session()
    del training_model
    del general_model
    for key in list(prediction_models.keys()):
        del prediction_models[key]
    del prediction_models
    for key in list(training_generators.keys()):
        del training_generators[key]
    del training_generators
    for key in list(validation_generators.keys()):
        del validation_generators[key]
    del validation_generators
    del callbacks
    tf.reset_default_graph()
    session.close()
    gc.collect()

    # end cleanup

    return {'best_epoch': loss_history.best_epoch,
            'best_val_loss': loss_history.best_val_loss,
            'best_train_loss': loss_history.best_train_loss}


if __name__ == '__main__':
    main()
