import os
import ast
from collections import OrderedDict
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, GRU, Dense, Reshape, Flatten, \
    Conv1D, MaxPooling1D, AveragePooling1D, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Dropout, TimeDistributed

from keras_fusionnet.utils.config import read_config_file, write_config_file

from keras_retinanet.utils.anchors import AnchorParameters
from keras_retinanet.utils.config import parse_anchor_parameters
from keras_retinanet.models.retinanet import default_classification_model, default_regression_model, retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model

from keras_transformer.models.transformer import Transformer, transformer, default_classification_layer, transformer_inference


def kronecker_product(mat1, mat2):
    """
    Compute the Kronecker product of two matrices
    :param mat1: The first matrix
    :param mat2: The second matrix
    :return: The product of the two matrices
    """
    batch, m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = K.reshape(mat1, [-1, m1, 1, n1, 1])
    batch, m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = K.reshape(mat2, [-1, 1, m2, 1, n2])
    return K.reshape(mat1_rsh * mat2_rsh, [-1, m1 * m2, n1 * n2])


def kronecker_product2D(tensors):
    tensor1 = tensors[0]
    tensor2 = tensors[1]
    #Separete slices of tensor and computes appropriate matrice kronecker product
    batch, m1, n1= tensor1.get_shape().as_list()
    batch, m2, n2 = tensor2.get_shape().as_list()
    x_list = []
    x_list.append(kronecker_product(tensor1[:,:,:], tensor2[:,:,:]))
    return K.reshape(x_list, [-1, m1 * m2, n1 * n2])

class FusionNet():

    def __init__(self, batch_size, fusionnet_model_config, vision_model_config, translation_model_config, log_path):
        self.num_nodes = fusionnet_model_config['node_number']
        if 'fusionnet_layers' in fusionnet_model_config:
            self.fusionnet_layers = fusionnet_model_config['fusionnet_layers']
        else:
            self.fusionnet_layers = None
        if 'config' in fusionnet_model_config:
            fusionnet_model_config['fusionnet_params'] = read_config_file(fusionnet_model_config['config'])
            if 'layer_parameters' in fusionnet_model_config['fusionnet_params']:
                self.num_nodes = int(fusionnet_model_config['fusionnet_params']['layer_parameters']['num_nodes'])
                if self.fusionnet_layers is not None:
                    fusionnet_model_config['fusionnet_params']['layer_parameters']['layers'] = str(self.fusionnet_layers)
            if log_path is not None:
                write_config_file(fusionnet_model_config['fusionnet_params'],
                                  os.path.join(log_path, 'fusion_config.ini'))
        if self.fusionnet_layers is not None and isinstance(self.fusionnet_layers, str):
            self.fusionnet_layers = ast.literal_eval(self.fusionnet_layers)

        if vision_model_config is not None:
            if vision_model_config['name'] == 'retinanet':
                # load anchor parameters, or pass None (so that defaults will be used)
                anchor_params = None
                num_anchors = None
                if 'config' in vision_model_config:
                    vision_model_config['retinanet_params'] = read_config_file(vision_model_config['config'])
                    if 'anchor_parameters' in vision_model_config['retinanet_params']:
                        anchor_params = parse_anchor_parameters(vision_model_config['retinanet_params'])
                        num_anchors = anchor_params.num_anchors()
                    # write config to log directory
                    if log_path is not None:
                        write_config_file(vision_model_config['retinanet_params'],
                                          os.path.join(log_path, 'vision_config.ini'))
                vision_model_config['anchor_params'] = anchor_params
                vision_model_config['num_anchors'] = num_anchors
                # TODO (fabawi): will need to flip 'image-max-side' 'image-min-side'], if the images were horizontal. make it dynamic
                vision_model_input =  keras.layers.Input(batch_shape=(batch_size, vision_model_config['image_min_side'], vision_model_config['image_max_side'], 3))
                self.vision_model = self.create_vision_retinanet(input=vision_model_input, **vision_model_config)
                # default to imagenet if nothing else is specified
                if vision_model_config['weights'] is None and vision_model_config['imagenet_weights']:
                    weights = vision_model_config['backbone'].download_imagenet()
                else:
                    weights = vision_model_config['weights']
                self.vision_model = self.model_with_weights(self.vision_model, weights, skip_mismatch=False)
                self.vision_model_config = vision_model_config

            if vision_model_config['name'] == 'simple_cnn':
                # TODO (fabawi): will need to flip 'image-max-side' 'image-min-side'], if the images were horizontal. make it dynamic
                vision_model_input = keras.layers.Input(batch_shape=(batch_size, vision_model_config['image_min_side'], vision_model_config['image_max_side'], 3))
                self.vision_model = self.create_vision_cnn(vision_model_input)
                self.vision_model_config = vision_model_config

        if translation_model_config is not None:
            if translation_model_config['name'] == 'transformer':
                if 'config' in translation_model_config:
                    translation_model_config['transformer_params'] = read_config_file(translation_model_config['config'])
                    translation_model_config['transformer_params']['init']['len_limit'] = str(translation_model_config['sequence_max_length'])
                else:
                    translation_model_config['transformer_params'] = {'init':{'len_limit': translation_model_config['sequence_max_length']}}
                    # write config to log directory
                    if log_path is not None:
                        write_config_file(translation_model_config['transformer_params'],
                                          os.path.join(log_path, 'language_translation_config.ini'))

                translation_model_inputs =  [keras.layers.Input(batch_shape=(batch_size, translation_model_config['sequence_max_length'])),
                                             keras.layers.Input(batch_shape=(batch_size, translation_model_config['sequence_max_length']))]
                self.language_translation_model = self.create_language_translation_transformer(inputs=translation_model_inputs, **translation_model_config)
                self.language_translation_model_config = translation_model_config

            self.fusionnet_model = None
            self.fusionnet_model_config = fusionnet_model_config

    def create_language_translation_transformer(self, inputs, i_tokens, o_tokens, **kwargs):
        """
        Create the Transformer language translation network
        :param inputs: The input
        :param i_tokens: The input tokens
        :param o_tokens: The output tokens
        :param kwargs: The Transformer parameters
        :return: The Transformer model
        """
        if 'transformer_params' in kwargs:
            s2s = Transformer(i_tokens, o_tokens, **kwargs['transformer_params']['init'])
            encoder_only = kwargs['transformer_params']['init'].get('encoder_only', False)
        else:
            s2s = Transformer(i_tokens, o_tokens)
            encoder_only = False

        def _transformer_sublayers_with_fusion_node():
            return [
                ('transformer_classification', default_classification_layer(s2s.o_tokens.num())),
                ('transformer_fusion_node', Dense(self.num_nodes, name='transformer_fusion_node'))
            ]

        return transformer(transformer_structure=s2s, inputs=inputs,
                           sublayers=_transformer_sublayers_with_fusion_node(), encoder_only=encoder_only)

    def create_vision_retinanet(self, input, backbone_retinanet, num_classes, num_anchors=None, freeze_backbone=None, **kwargs):
        """
        Create the RetinaNet visual network variant
        :param input: The input
        :param backbone_retinanet: The RetinaNet backbone e.g.: ResNet or MobileNet
        :param num_classes: The number of block classes to identify
        :param num_anchors: The number of anchors
        :param freeze_backbone: If true, the backbone weights are frozen
        :param kwargs: None
        :return: The RetinaNet model with the fusion nude
        """
        def _retinanet_submodels_with_fusion_node(num_classes, num_anchors):
            if num_anchors is None:
                num_anchors = AnchorParameters.default.num_anchors()
            return [
                ('retinanet_regression', default_regression_model(4, num_anchors)),
                ('retinanet_classification', default_classification_model(num_classes, num_anchors)),
                ('retinanet_fusion_node', default_regression_model(self.num_nodes, num_anchors=num_anchors,
                                                                   regression_feature_size=2, name='retinanet_fusion_submodel'))
            ]

        modifier = freeze_model if freeze_backbone else None

        return backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, inputs=input,
                           submodels=_retinanet_submodels_with_fusion_node(num_classes, num_anchors))

    def create_vision_cnn(self, l_in, name='vision_cnn'):
        """
        Create the CNN visual network variant
         Neural End-to-End Self-learning of Visuomotor Skills by Environment Interaction, M.Kerzel and S. Wermter, 2016
        :param l_in: The input layers
        :param name: The name of the subnetwork
        :return: The CNN model
        """
        # first conv layer
        l_conv_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', name='first_conv')(l_in)
        # max pooling layer
        l_pool_1 = MaxPooling2D((1, 1), name='first_pool')(l_conv_1)
        # another convolution layer
        l_conv_2 = Conv2D(16, (4, 4), activation='relu', kernel_initializer='glorot_uniform', name='second_conv')(l_pool_1)
        # another max pool
        l_pool_2 = MaxPooling2D((1, 1), name='second_pool')(l_conv_2)
        # l_pool_2 = Flatten()(l_pool_2)
        l_pool_2 = Reshape((-1, self.num_nodes), name='first_reshape')(l_pool_2)

        return Model(l_in, [l_pool_2], name=name)

    def create_language_translation_rnn(self, l_in, name='language_translation_rnn'):
        # TODO (fabawi): Language translation using rnns2s. To be implemented
        pass

    def create_model(self, mode='kronecker'):
        """
        Create the FusionNet model
        :param mode: The fusion mode. Options: concatenate|Kronecker
        :return: The model
        """
        vision_model = self.vision_model
        if len(vision_model._output_layers) > 2:  # retinanet
            vision_model_fusion_branch = vision_model.output[2]
        else: # cnn
            vision_model_fusion_branch = vision_model.output


        transformer = self.language_translation_model
        transformer_fusion_branch = transformer.output[1]

        if mode == 'concatenate':
            fusion_bottleneck = keras.layers.concatenate([vision_model_fusion_branch, transformer_fusion_branch], axis=1)
        elif mode == 'kronecker':
            fusion_bottleneck = Lambda(kronecker_product2D)([vision_model_fusion_branch, transformer_fusion_branch])

        fusion_bottleneck = Flatten()(fusion_bottleneck)
        fusion_bottleneck = self.create_fusionnet_layers(fusion_bottleneck, layers=self.fusionnet_layers)
        fusionnet_output = Dense(self.fusionnet_model_config['coordinate_number'], kernel_initializer=keras.initializers.glorot_uniform(),
                                 activation='sigmoid', name='fusionnet_regression')(fusion_bottleneck)

        inputs = [self.vision_model.input] + self.language_translation_model.input

        vision_outputs,language_translation_outputs,fusionnet_outputs = list(), list(), list() # [retinanet.output[0], retinanet.output[1]] + [transformer.output[0]] + [fusionnet_output]
        if any('retinanet_regression' in filt for filt in self.fusionnet_model_config['output_filter']):
            vision_outputs.append(vision_model.output[0])
        if any('retinanet_classification' in filt for filt in self.fusionnet_model_config['output_filter']):
            vision_outputs.append(vision_model.output[1])

        if any('transformer_classification' in filt for filt in self.fusionnet_model_config['output_filter']):
            language_translation_outputs.append(transformer.output[0])

        if any('fusionnet_regression' in filt for filt in self.fusionnet_model_config['output_filter']):
            fusionnet_outputs.append(fusionnet_output)
        # get the output of the model

        model = Model(inputs=inputs, outputs= vision_outputs + language_translation_outputs + fusionnet_outputs)

        self.fusionnet_model = model
        return model

    def model_with_weights(self, model, weights, skip_mismatch):
        """
        Load weights for a given model
        :param model: The model to load weights for
        :param weights: The weights to load
        :param skip_mismatch: If True, skips layers whose shape of weights doesn't match with the model
        :return: The model with loaded weights
        """
        if weights is not None:
            model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        return model

    def load_model_specific_weights(self):
        """
        Load module specific weights
        :return: The model with loaded weights
        """
        if 'pretrained_snapshot' in self.vision_model_config:
            if self.vision_model_config['pretrained_snapshot'] is not None:
                self.vision_model.load_weights(self.vision_model_config['pretrained_snapshot'],
                                               skip_mismatch=True, by_name=True)
        if 'pretrained_snapshot' in self.language_translation_model_config:
            if self.language_translation_model_config['pretrained_snapshot'] is not None:
                self.language_translation_model.load_weights(self.language_translation_model_config['pretrained_snapshot'],
                                                             skip_mismatch=True, by_name=True)

    def create_prediction_models(self):
        """
        Create inference models
        :return: The inference model
        """
        return_models = {}
        if self.vision_model_config['name'] == 'retinanet':
            return_models['vision'] =  retinanet_bbox(model=self.vision_model, anchor_params=self.vision_model_config['anchor_params'])
        if self.language_translation_model_config['name'] == 'transformer':
            return_models['language_translation'] = transformer_inference(model=self.language_translation_model)
        return_models['fusionnet'] = self.fusionnet_model
        return return_models


    def create_fusionnet_layers(self, fusionnet, layers=None):
        """
        Extend the FusionNet with more layers
        :param fusionnet: The FusionNet model
        :param layers: The layer structure for the added layers
        :return: The FusionNet model with the added layers
        """
        # stack a deep densely-connected network on top
        if layers is None:
            return fusionnet

            # cnn
            # layers = [('dense_large', None, None),
            #           ('reshape_expand_dim_4', None, None),
            #           ('conv2d', 32, 'sigmoid'),
            #           ('conv_max_pool_large', None, None),
            #           ('dropout', 0.25, None),
            #           ('flatten', None, None)]

            # lstm/gru
            # layers = [('dense_large', None, None),
            #           ('reshape_expand_dim_3', None, None),
            #           ('rnn', 32, 'sigmoid'),
            #           ('dropout', 0.25, None)]

        for layer, size, activation in layers:
            # this needs to be preceeded by a dense layer being a multiple of 300 to work
            if layer == 'reshape_expand_dim_4':
                # fusionnet = Reshape((-1,keras.backend.get_variable_shape(fusionnet)[1]/700, 700))(fusionnet)
                fusionnet = Reshape((30, 10, -1))(fusionnet)
            if layer == 'reshape_expand_dim_3':
                # fusionnet = Reshape((-1,keras.backend.get_variable_shape(fusionnet)[1]/700, 700))(fusionnet)
                fusionnet = Reshape((30, -1))(fusionnet)

            elif layer == 'dense_large':
                fusionnet = Dense(900, activation='sigmoid',
                                  kernel_initializer=keras.initializers.glorot_uniform())(fusionnet)
            elif layer == 'dense_small':
                fusionnet = Dense(64, activation='sigmoid',
                                  kernel_initializer=keras.initializers.glorot_uniform())(fusionnet)
            elif layer == 'dense':
                fusionnet = Dense(size, activation=activation,
                                  kernel_initializer=keras.initializers.glorot_uniform())(fusionnet)

            elif layer == 'conv2d_large':
                fusionnet = Conv2D(64, (6, 6), activation='relu', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'conv2d_small':
                fusionnet = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'conv2d':
                fusionnet = Conv2D(size, (3, 3), activation=activation, kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'conv1d':
                fusionnet = Conv1D(size, 3, activation=activation, kernel_initializer='glorot_uniform')(fusionnet)

            elif layer == 'conv2d_max_pool_large':
                fusionnet = MaxPooling2D((2, 2))(fusionnet)
            elif layer == 'conv2d_max_pool_small':
                fusionnet = MaxPooling2D((1, 1))(fusionnet)
            elif layer == 'conv1d_max_pool_large':
                fusionnet = MaxPooling1D(2)(fusionnet)
            elif layer == 'conv1d_max_pool_small':
                fusionnet = MaxPooling1D(1)(fusionnet)

            elif layer == 'conv2d_avg_pool_large':
                fusionnet = AveragePooling2D((2, 2))(fusionnet)
            elif layer == 'conv2d_avg_pool_small':
                fusionnet = AveragePooling2D((1, 1))(fusionnet)
            elif layer == 'conv1d_avg_pool_large':
                fusionnet = AveragePooling1D(2)(fusionnet)
            elif layer == 'conv1d_avg_pool_small':
                fusionnet = AveragePooling1D(1)(fusionnet)


            elif layer == 'lstm_large':
                fusionnet = LSTM(128, activation='tanh', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'lstm_small':
                fusionnet = LSTM(32, activation='tanh', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'lstm':
                fusionnet = LSTM(size, activation=activation, kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'lstm_sequence':
                fusionnet = LSTM(size, activation=activation, kernel_initializer='glorot_uniform', return_sequences=True)(fusionnet)

            elif layer == 'gru_large':
                fusionnet = GRU(128, activation='tanh', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'gru_small':
                fusionnet = GRU(32, activation='tanh', kernel_initializer='glorot_uniform')(fusionnet)
            elif layer == 'gru':
                fusionnet = GRU(size, activation=activation, kernel_initializer='glorot_uniform')(fusionnet)

            elif layer == 'dropout':
                fusionnet = Dropout(size)(fusionnet)

            # always flatten after conv
            elif layer == 'flatten':
                fusionnet = Flatten()(fusionnet)

        return fusionnet
