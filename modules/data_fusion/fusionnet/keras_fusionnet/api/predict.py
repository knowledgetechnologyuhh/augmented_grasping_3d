
import os

import keras
import keras.preprocessing.image
import tensorflow as tf

from keras_fusionnet.utils.keras_version import check_keras_version
from keras_fusionnet.utils.config import read_config_file

from keras_fusionnet.models.fusionnet import FusionNet
from keras_fusionnet.models.model_setup import create_vision_objects, create_translation_objects
from keras_fusionnet.models.model_setup import create_extra_fusionnet_args, create_extra_vision_args, create_extra_translation_args
from keras_fusionnet.models.model_setup import create_fusionnet_model_config, create_vision_model_config, create_translation_model_config
from keras_fusionnet.preprocessing.generator_setup import create_fusionnet_generators, create_vision_generators, create_translation_generators
from keras_fusionnet.callbacks.predict_setup import fusionnet_model_prediction, vision_model_prediction, translation_model_prediction


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class Prediction(object):

    def __init__(self, log_path=None):

        # read the arguments
        args = read_config_file(os.path.join(log_path,'arguments.json'))

        # make sure keras is the minimum required version
        check_keras_version()

        # optionally choose specific GPU
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())

        # create the generators
        self.validation_generators = {}

        # create the vision model objects
        vision_objects = create_vision_objects(args.vision)
        # add extra arguments
        extra_vision_args = create_extra_vision_args(args.vision, vision_objects)
        # create the vision generators
        vision_validation_generator = self._create_generators(args.vision, extra_vision_args)
        self.validation_generators['vision'] = vision_validation_generator
        # set the vision model configuration
        vision_model_config = create_vision_model_config(args.vision, vision_objects, vision_validation_generator)

        # create the translation model objects
        translation_objects = create_translation_objects(args.language_translation)
        # add extra arguments
        extra_translation_args = create_extra_translation_args(args.language_translation, translation_objects)
        # create the translation generators
        translation_validation_generator = self._create_generators(args.language_translation, extra_translation_args)
        self.validation_generators['language_translation'] = translation_validation_generator
        # set the translation model configuration
        translation_model_config = create_translation_model_config(args.language_translation, translation_objects,
                                                                   translation_validation_generator)

        # create the fusionnet model objects
        fusionnet_objects = {'vision_train_generator': vision_validation_generator,
                             'translation_train_generator': translation_validation_generator,
                             'vision_val_generator': vision_validation_generator,
                             'translation_val_generator': translation_validation_generator}
        # add extra arguments
        extra_fusionnet_args = create_extra_fusionnet_args(args.fusionnet, fusionnet_objects)
        # create the fusionnet generators
        fusionnet_validation_generator = self._create_generators(args.fusionnet, extra_fusionnet_args)
        self.validation_generators['fusionnet'] = fusionnet_validation_generator
        # set the fusionnet model configuration
        fusionnet_model_config = create_fusionnet_model_config(args.fusionnet, fusionnet_objects, fusionnet_validation_generator)

        # create the model
        print('Creating model, this may take a second...')
        self.prediction_models = self._create_models(
            snapshot=args.snapshot,
            multi_gpu=args.multi_gpu,
            batch_size=1,
            fusionnet_model_config=fusionnet_model_config,
            vision_model_config=vision_model_config,
            translation_model_config=translation_model_config,
        )
        self.args = args

    def _create_models(self, snapshot, multi_gpu, batch_size, fusionnet_model_config, vision_model_config, translation_model_config):
        """ Creates two models (model, prediction_model).    """

        # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
        # optionally wrap in a parallel model

        fusionnet = FusionNet(batch_size, fusionnet_model_config, vision_model_config, translation_model_config, log_path=None)
        fusion_model = fusionnet.create_model()

        if multi_gpu > 1:
            from keras.utils import multi_gpu_model
            with tf.device('/cpu:0'):
                model = fusion_model
        else:
            model = fusion_model

        # load snapshot for the entire model
        if snapshot is not None:
            print('Loading model, this may take a second...')
            model.load_weights(snapshot,skip_mismatch=True, by_name=True)

        # load pretrained snapshots for all submodels individually
        fusionnet.load_model_specific_weights()

        # make prediction models
        prediction_models = fusionnet.create_prediction_models()



        return prediction_models

    def _run_prediction_models(self, prediction_models, validation_generators, command_sequence, image_path, args):
        """ Creates the callbacks to use during training.

        Args
            prediction_model: The model that should be used for validation.
            validation_generator: The generator for creating validation data.
            args: parseargs args object.

        Returns:
            The predictions.
        """
        predictions = list()
        processed_data_objects = list()
        # the order of the prediction matters since their data is appended to a list which will be used for fusion
        if 'vision' in validation_generators and validation_generators['vision']:
            vision_prediction, processed_data = vision_model_prediction(generator=validation_generators['vision'],
                                                   model=prediction_models['vision'], image=image_path,
                                                   args=args.vision)
            predictions.append(vision_prediction)
            processed_data_objects.append(processed_data)

        if 'language_translation' in validation_generators and validation_generators['language_translation']:
            translation_prediction, processed_data = translation_model_prediction(generator=validation_generators['language_translation'],
                                                   model=prediction_models['language_translation'], command=command_sequence,
                                                   args=args.language_translation)
            predictions.append(translation_prediction)
            processed_data_objects.extend(processed_data)

        # if args.fusionnet.evaluation and validation_generators['fusionnet']:
        #     fusionnet_prediction = fusionnet_model_callback(generator=validation_generators['fusionnet'],
        #                                            model=prediction_models['fusionnet'],
        #                                            args=args.fusionnet)

        return predictions, processed_data_objects

    def _create_generators(self, args, extra_args):
        """ Create generators for training and validation.

        Args
            args : parseargs object containing configuration for generators.
            extra_args : dictionary containing extra arguments for generators.
        """
        if args.module_type == 'vision':
            _, validation_generator = create_vision_generators(args, extra_args)

        elif args.module_type == 'language_translation':
            _, validation_generator = create_translation_generators(args, extra_args)

        elif args.module_type == 'fusionnet':
            _, validation_generator = create_fusionnet_generators(args, extra_args)

        else:
            _, validation_generator = None, None

        return validation_generator

    def predict(self, image, command):
        # create the callbacks
        predictions, processed_data_objects = self._run_prediction_models(
            self.prediction_models,
            self.validation_generators,
            command_sequence=command,
            image_path=image,
            args=self.args)

        # TODO (fabawi): rescale the joints and move this to an independent script like retinanet and the transformer
        final_prediction = self.prediction_models['fusionnet'].predict_on_batch(processed_data_objects)
        predictions.append({'joint_angles_normalized':final_prediction[3]})
        return predictions
