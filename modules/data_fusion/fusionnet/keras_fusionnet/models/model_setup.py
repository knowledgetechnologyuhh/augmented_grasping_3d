import argparse

from keras_retinanet import layers  # noqa: F401
from keras_retinanet import models
from keras_retinanet.utils.image import normalize_image

def create_extra_fusionnet_args(args, objects):
    return argparse.Namespace(**{
        "train_generators": {
            'vision_generator': objects['vision_train_generator'],
            'language_translation_generator': objects['translation_train_generator']
        },
        "val_generators": {
            'vision_generator': objects['vision_val_generator'],
            'language_translation_generator': objects['translation_val_generator']
        }
    })


def create_fusionnet_model_config(args, objects, generator):
    return_config = {
        'name': 'fusionnet',
        'output_filter': args.output_filter,
        'coordinate_number': args.coordinate_number,
        'node_number': args.node_number}
    if args.fusionnet_layers:
        return_config['fusionnet_layers'] = args.fusionnet_layers
    if args.config:
        return_config['config'] = args.config
    return return_config


def create_vision_objects(args):
    if args.vision_module_type == 'retinanet':
        return {'backbone': models.backbone(args.backbone)}

def create_extra_vision_args(args, objects):
    if args.vision_module_type == 'retinanet':
        return argparse.Namespace(**{"preprocess_image": objects['backbone'].preprocess_image})
    elif args.vision_module_type == 'simple_cnn':
        return argparse.Namespace(**{"preprocess_image": normalize_image})

def create_vision_model_config(args, objects, generator):
    if args.vision_module_type == 'retinanet':
        return_config = {
            'name': 'retinanet',
            'backbone_retinanet': objects['backbone'].retinanet,
            'backbone': objects['backbone'],
            'num_classes': generator.num_classes(),
            'freeze_backbone': args.freeze_backbone,
            'weights': args.weights,
            'imagenet_weights': args.imagenet_weights,
            'pretrained_snapshot': args.pretrained_snapshot,
            'image_max_side': args.image_max_side,
            'image_min_side': args.image_min_side
        }
        if args.config:
            return_config['config'] = args.config
        return return_config

    elif args.vision_module_type == 'simple_cnn':
        return_config = {
            'name': 'simple_cnn',
            'image_max_side': args.image_max_side,
            'image_min_side': args.image_min_side
        }
        return return_config

def create_translation_objects(args):
    if args.translation_module_type == 'transformer':
        return None


def create_extra_translation_args(args, objects):
    if args.translation_module_type == 'transformer':
        return None


def create_translation_model_config(args, objects, generator):
    if args.translation_module_type == 'transformer':
        return_config = {
            'name': 'transformer',
            'sequence_max_length': generator.sequence_max_length,
            'i_tokens': generator.i_tokens,
            'o_tokens': generator.o_tokens,
            'i_embedding_matrix': generator.i_embedding_matrix,
            'o_embedding_matrix': generator.o_embedding_matrix,
            'pretrained_snapshot': args.pretrained_snapshot,

        }
        if args.config:
            return_config['config'] = args.config
        return return_config
