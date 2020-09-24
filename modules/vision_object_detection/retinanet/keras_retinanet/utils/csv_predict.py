from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import cv2

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, image_filename, score_threshold=0.05, max_detections=1024, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        image_filename  : The filename of the image to be predicted
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    raw_image    = generator.load_image(image_filename=image_filename)
    image        = generator.preprocess_image(raw_image.copy())
    image, scale = generator.resize_image(image)

    if keras.backend.image_data_format() == 'channels_first':
        image = image.transpose((2, 0, 1))

    # run network
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    if save_path is not None:
        draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)
        cv2.imwrite(os.path.join(save_path, 'retinanet_prediction.png'), raw_image)

    detections = {"raw_image": raw_image, "image_boxes": image_boxes, "image_scores": image_scores, "image_labels":image_labels}
    return detections, np.expand_dims(image, axis=0)


def predict(
    generator,
    model,
    image_filename,
    score_threshold=0.05,
    max_detections=1024,
    save_path=None
):
    """ Predict a given image filename using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        image_filename  : The filename of the image to be predicted
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections
    detections, processed_data = _get_detections(generator, model, image_filename,
                                         score_threshold=score_threshold,
                                         max_detections=max_detections,
                                         save_path=save_path)

    return detections, processed_data
