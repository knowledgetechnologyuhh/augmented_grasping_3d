import os
from keras_fusionnet.callbacks import RedirectModel
from keras_transformer.callbacks.eval import Evaluate as TransformerEval
from keras_retinanet.callbacks.eval import Evaluate as RetinanetEval

def translation_model_callback(generator, model, tensorboard_callback, comet_ml_experiment, args):
    if args.translation_module_type == 'transformer':
        transformer_eval = TransformerEval(generator, tensorboard=tensorboard_callback, comet=comet_ml_experiment,
                                           evaluate_metrics=True, save_path=os.path.join(args.log_path,'evaluation'))
        transformer_evaluation = RedirectModel(transformer_eval, model)
        return transformer_evaluation

def vision_model_callback(generator, model, tensorboard_callback, comet_ml_experiment, args):
    if args.vision_module_type == 'retinanet':
        retinanet_eval = RetinanetEval(generator, tensorboard=tensorboard_callback, comet=comet_ml_experiment, location_bias=True,
                                       weighted_average=args.weighted_average, save_path=os.path.join(args.log_path,'evaluation'))
        retinanet_evaluation = RedirectModel(retinanet_eval, model)
        return retinanet_evaluation


def fusionnet_model_callback(generator, model, tensorboard_callback, comet_ml_experiment, args):
    return None