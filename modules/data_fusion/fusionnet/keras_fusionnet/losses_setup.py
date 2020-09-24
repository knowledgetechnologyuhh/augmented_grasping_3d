from keras_transformer import losses as transformer_losses
from keras_transformer import metrics as transformer_metrics
from keras_retinanet import losses as retinanet_losses


def vision_model_losses(output, config):
    if 'retinanet' in output:
        if 'retinanet_regression' in output:
            try:
                weight = {'retinanet_regression': float(output.split(':')[1])}
            except:
                weight = {'retinanet_regression': 1.0}
            return {'retinanet_regression': retinanet_losses.smooth_l1()}, weight
        if 'retinanet_classification' in output:
            try:
                weight = {'retinanet_classification': float(output.split(':')[1])}
            except:
                weight = {'retinanet_classification': 1.0}
            return {'retinanet_classification': retinanet_losses.focal()}, weight
    return {},{}


def translation_model_losses(output, config):
    if 'transformer' in output:
        if 'transformer_classification' in output:
            try:
                weight = {'transformer_classification': float(output.split(':')[1])}
            except:
                weight = {'transformer_classification': 1.0}
            return {'transformer_classification': transformer_losses.masked_ce(layer_size=config['sequence_max_length'])}, weight
    return {},{}


def fusionnet_model_losses(output, config):
    if 'fusionnet_regression' in output:
        try:
            weight = {'fusionnet_regression': float(output.split(':')[1])}
        except:
            weight = {'fusionnet_regression': 1.0}
        return {'fusionnet_regression': 'mean_squared_error'}, weight
    return {},{}


def vision_model_metrics(output, config):
    return {}


def translation_model_metrics(output, config):
    if 'transformer' in output:
        if 'transformer_classification' in output:
            return {
                'transformer_classification':
                    [transformer_metrics.masked_accuracy(layer_size=config['sequence_max_length']),
                     transformer_metrics.masked_perplexity(layer_size=config['sequence_max_length'])]
            }
    return {}


def fusionnet_model_metrics(output, config):
    return {}