from __future__ import print_function
# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
from utils import visualization_utils as vis_util
from utils import label_map_util as lmu
import numpy as np
import scipy
import tensorflow as tf
import cv2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('input_image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('path_to_labels', '', 'path to labels')
FLAGS = tf.app.flags.FLAGS

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
img = scipy.misc.imread(FLAGS.input_image)
request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape)))

# Call the prediction server
result = stub.Predict(request, 180.0)  # 10 secs timeout

print(result)

# Plot boxes on the input image
#category_index = lmu.load_labelmap(FLAGS.path_to_labels)
category_index2 = lmu.create_category_index_from_labelmap(FLAGS.path_to_labels, False)
#print(category_index2.Keys())
boxes = result.outputs['detection_boxes'].float_val
classes = result.outputs['detection_classes'].float_val
scores = result.outputs['detection_scores'].float_val
image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    np.reshape(boxes,[100,4]),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index2,
    use_normalized_coordinates=True,
    line_thickness=5)

# Save inference to disk

scipy.misc.imsave('%s.jpg'%(FLAGS.input_image), image_vis)

