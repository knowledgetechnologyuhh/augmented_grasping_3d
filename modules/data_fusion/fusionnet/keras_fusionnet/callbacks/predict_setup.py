import os
from keras_transformer.utils.predict import predict as transformer_predict
from keras_retinanet.utils.csv_predict import predict as retinanet_predict


def translation_model_prediction(generator, model, command, args):
    if args.translation_module_type == 'transformer':
        prediction, processed_data = transformer_predict(generator, model, command, beam_search=False, beam_width=5,
                                               save_path=os.path.join(args.log_path,'prediction'))
        return prediction, processed_data


def vision_model_prediction(generator, model, image, args):
    if args.vision_module_type == 'retinanet':
        prediction, processed_data = retinanet_predict(generator, model, image, score_threshold = 0.05, max_detections=1024,
                                           save_path=os.path.join(args.log_path,'prediction'))
        return prediction, processed_data


def fusionnet_model_prediction(generator, model, args):
    return None