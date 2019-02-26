from __future__ import print_function
# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util as lmu
import xml.etree.ElementTree as ET
import glob
import numpy as np
import scipy
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def main():
    # Adding flags for script
    tf.app.flags.DEFINE_string('server', 'localhost:9000',
                            'PredictionService host:port')
    tf.app.flags.DEFINE_string('input_image', '', 'path to image in JPEG format')
    tf.app.flags.DEFINE_string('path_to_labels', '', 'path to labels')
    FLAGS = tf.app.flags.FLAGS

    # Minimum treshold of certainty for boxes to be included, in percentage.
    min_score_tresh = 0.5

    # Create stub
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create prediction request object
    request = predict_pb2.PredictRequest()

    # Specify model name (must be the same as when the TensorFlow serving serving was started)
    request.model_spec.name = 'model_test'

    # Initalize prediction 
    # Specify signature name (should be the same as specified when exporting model)
    request.model_spec.signature_name = ""


    images = glob.glob(FLAGS.input_image + "/*.jpg")
    for image in images:
        # Reading image from given path
        img = scipy.misc.imread(image)
        # Reading the resolution of said image.
        height, width, channels = img.shape

        request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape)))

        # Call the prediction server
        result = stub.Predict(request, 180.0)  # 10 secs timeout


        # Plot boxes on the input image
        pred_category_index = lmu.create_category_index_from_labelmap(FLAGS.path_to_labels, False)
        pred_boxes = result.outputs['detection_boxes'].float_val
        pred_classes = result.outputs['detection_classes'].float_val
        pred_scores = result.outputs['detection_scores'].float_val

        # Format output properly before converting to Pascal VOC
        pred_boxes = np.reshape(pred_boxes,[100,4])
        pred_classes = np.squeeze(pred_classes).astype(np.int32)
        pred_scores = np.squeeze(pred_scores)

        # Iterate through each box predicted by the served model
        for i in range(pred_boxes.shape[0]):
            # Discard any boxas under 50% certainty
            if pred_scores[i] > min_score_tresh:

                # Format box output
                pred_box = tuple(pred_boxes[i].tolist())

                # Fetch "class_name", also known as label.
                if pred_classes[i] in pred_category_index.keys():
                    pred_class_name = pred_category_index[pred_classes[i]]['name']
                else:
                    pred_class_name = 'N/A'

                # Fetch x and y positions of box-corners in percentage.
                pred_ymin_percent, pred_xmin_percent, pred_ymax_percent, pred_xmax_percent = pred_box

                # Calculate the pixel the box corners are positioned at.
                pred_ymin = int(pred_ymin_percent * height)
                pred_ymax = int(pred_ymax_percent * height)
                pred_xmin = int(pred_xmin_percent * width)
                pred_xmax = int(pred_xmax_percent * width)

                print('mainrun')


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


if __name__ == '__main__':
    main()

    # Define the XML path of the new Pascal VOC file.
    # xml_path = image.rstrip('.jpg') + '.xml'