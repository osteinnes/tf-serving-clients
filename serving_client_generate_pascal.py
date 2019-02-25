from __future__ import print_function
# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util as lmu
import glob
import numpy as np
import scipy
import tensorflow as tf
from pascal_voc_writer import Writer
#import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# Adding flags for script
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('input_image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('path_to_labels', '', 'path to labels')
FLAGS = tf.app.flags.FLAGS

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

    # Create Pascal VOC writer.
    writer = Writer(image, width, height)

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape)))

    # Call the prediction server
    result = stub.Predict(request, 180.0)  # 10 secs timeout


    # Plot boxes on the input image
    category_index = lmu.create_category_index_from_labelmap(FLAGS.path_to_labels, False)
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val

    boxes = np.reshape(boxes,[100,4])
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)


    for i in range(boxes.shape[0]):
        if scores[i] > min_score_tresh:
            box = tuple(boxes[i].tolist())

            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'

            ymin_percent, xmin_percent, ymax_percent, xmax_percent = box

            ymin = ymin_percent * height
            ymax = ymax_percent * height

            xmin = xmin_percent * width
            xmax = xmax_percent * width

            writer.addObject(class_name, xmin, ymin, xmax, ymax)

    xml_path = image.rstrip('.jpg') + '.xml'
    writer.save(xml_path)

            

    #image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    #    img,
    #    np.reshape(boxes,[100,4]),
    #    np.squeeze(classes).astype(np.int32),
    #    np.squeeze(scores),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    line_thickness=4)
    
