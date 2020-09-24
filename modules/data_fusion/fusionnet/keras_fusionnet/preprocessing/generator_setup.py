from keras_retinanet.utils.transform import random_transform_generator
from keras_fusionnet.preprocessing.vision.retinanet_generator import RetinanetCSVGenerator
from keras_fusionnet.preprocessing.fusionnet_generator import FusionnetCSVGenerator
from keras_fusionnet.preprocessing.language_translation.transformer_generator import TransformerCSVGenerator


def create_fusionnet_generators(args, extra_args):
    common_args = {
        'batch_size': args.batch_size,
        'output_filter': args.output_filter,
        'coordinate_number': args.coordinate_number,
        'coordinate_limits_file': args.coordinate_limits,
    }

    train_generator = FusionnetCSVGenerator(
        csv_data_file=args.annotations,
        submodule_generators=extra_args.train_generators,
        **common_args
    )

    if args.val_annotations:
        validation_generator = FusionnetCSVGenerator(
            csv_data_file=args.val_annotations,
            submodule_generators=extra_args.val_generators,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def create_vision_generators(args, extra_args):
    if args.vision_module_type == 'retinanet':
        common_args = {
            'batch_size': args.batch_size,
            'config': args.config,
            'image_min_side': args.image_min_side,
            'image_max_side': args.image_max_side,
            'preprocess_image': extra_args.preprocess_image,
        }

        # create random transform generator for augmenting training data
        if args.random_transform:
            transform_generator = random_transform_generator(
                min_rotation=-0.1,
                max_rotation=0.1,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.0, # disabled for equivarience purposes - was 0.5
                flip_y_chance=0.0, # disabled for equivarience purposes - was 0.5
            )
        else:
            transform_generator = random_transform_generator(flip_x_chance=0.5)

        train_generator = RetinanetCSVGenerator(
            csv_data_file=args.annotations,
            csv_class_file=args.classes,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = RetinanetCSVGenerator(
                csv_data_file=args.val_annotations,
                csv_class_file=args.classes,
                **common_args
            )
        else:
            validation_generator = None

        return train_generator, validation_generator

    if args.vision_module_type == 'simple_cnn':
        common_args = {
            'batch_size': args.batch_size,
            'image_min_side': args.image_min_side,
            'image_max_side': args.image_max_side,
            'preprocess_image': extra_args.preprocess_image,
        }

        # create random transform generator for augmenting training data
        if args.random_transform:
            transform_generator = random_transform_generator(
                min_rotation=-0.1,
                max_rotation=0.1,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        else:
            transform_generator = random_transform_generator(flip_x_chance=0.5)

        # uses the same generator as the retinanet
        train_generator = RetinanetCSVGenerator(
            csv_data_file=args.annotations,
            csv_class_file=args.classes,
            transform_generator=transform_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = RetinanetCSVGenerator(
                csv_data_file=args.val_annotations,
                csv_class_file=args.classes,
                **common_args
            )
        else:
            validation_generator = None

        return train_generator, validation_generator

def create_translation_generators(args, extra_args):
    if args.translation_module_type == 'transformer':
        common_args = {
            'batch_size': args.batch_size,
            'sequence_max_length': args.sequence_max_length,
            'i_embedding_matrix_file': args.i_embedding_matrix,
            'o_embedding_matrix_file': args.o_embedding_matrix
        }

        train_generator = TransformerCSVGenerator(
            csv_data_file=args.annotations,
            tokens_file=args.vocab,

            **common_args
        )

        if args.val_annotations:
            validation_generator = TransformerCSVGenerator(
                csv_data_file=args.val_annotations,
                golden_data_file=args.val_golden_set,
                shuffle_groups=False,
                # i_tokens=train_generator.i_tokens,
                # o_tokens=train_generator.o_tokens,
                tokens_file=args.vocab,
                **common_args
            )
        else:
            validation_generator = None

        return train_generator, validation_generator