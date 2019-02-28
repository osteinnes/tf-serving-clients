from __future__ import print_function
# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util as lmu
import xml.etree.ElementTree as ET
import glob
import numpy as np
import scipy
import cv2
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tools.iou_xml import Writer


def main():
    # Adding flags for script
    tf.app.flags.DEFINE_string('server', 'localhost:9000',
                               'PredictionService host:port')
    tf.app.flags.DEFINE_string(
        'input_image', '', 'path to image in JPEG format')
    tf.app.flags.DEFINE_string('path_to_labels', '', 'path to labels')
    tf.app.flags.DEFINE_string('debug', 'no', 'Debug status')
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.debug == 'yes':
        DEBUG = True
    else:
        DEBUG = False

    # Minimum treshold of certainty for boxes to be included, in percentage.
    min_score_tresh = 0.5

    writer = Writer(FLAGS.input_image)

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

        print()
        print()
        print("Results for image: " + image)
        print()

        ious = []
        iou_tester = 0

        # Reading image from given path
        img = scipy.misc.imread(image)
        # Reading the resolution of said image.
        height, width, channels = img.shape

        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(img, shape=[1] + list(img.shape)))

        # Call the prediction server
        result = stub.Predict(request, 180.0)  # 10 secs timeout

        # Plot boxes on the input image
        pred_category_index = lmu.create_category_index_from_labelmap(
            FLAGS.path_to_labels, False)
        pred_boxes = result.outputs['detection_boxes'].float_val
        pred_classes = result.outputs['detection_classes'].float_val
        pred_scores = result.outputs['detection_scores'].float_val

        # Format output properly before converting to Pascal VOC
        pred_boxes = np.reshape(pred_boxes, [100, 4])
        pred_classes = np.squeeze(pred_classes).astype(np.int32)
        pred_scores = np.squeeze(pred_scores)

        xml_path = image.rstrip('.jpg') + '.xml'

        annotation_names, annotation_boxes = read_pascal_voc(xml_path)

        cert_scores = []

        num_label_boxes = 0
        num_pred_boxes = 0

        cracks = []

        actu_class = 'N/A'

        # Iterate through each box predicted by the served model
        for i in range(pred_boxes.shape[0]):
            # Discard any boxas under 50% certainty
            if pred_scores[i] > min_score_tresh:

                if DEBUG:
                    print_img = cv2.imread(image)
                    gt = []
                    print_img_array = []

                

                cert_scores = []
                cert_scores.append(pred_scores[i])

                iou_prelim = 0
                iou_result = 0

                certainty = 0
                certainty = pred_scores[i]

                num_pred_boxes = i+1

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

                pred = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]

                # load the image
                print("Predicted box number ", i,
                      " Predicted class: ", pred_class_name)

                num_label_boxes = len(annotation_boxes)

                k = 0
                for k in range(len(annotation_boxes)):

                    # draw the ground-truth bounding box along with the predicted
                    # bounding box
                    if DEBUG:

                        # Add current label.
                        gt.append(annotation_boxes[k])
                        # Add current image.
                        print_img_array.append(print_img)

                        # Draw rectangle on image.
                        cv2.rectangle(print_img_array[k], tuple(gt[k][:2]),
                                      tuple(gt[k][2:]), (0, 255, 0), 2)
                        cv2.rectangle(print_img_array[k], tuple(pred[:2]),
                                      tuple(pred[2:]), (0, 0, 255), 2)

                        # Show image.
                        cv2.imshow("Image " + str(k), print_img_array[k])

                    # Calculate IOU result.
                    iou_prelim = bb_intersection_over_union(
                        pred, annotation_boxes[k])

                    # Checks if IOU is above 40%
                    if iou_prelim > 0.1:
                        print(iou_prelim)

                        actu_class = annotation_names[k]

                        if pred_class_name == annotation_names[k]:
                            print("Class is correct: ", annotation_names[k])
                        else:
                            print("Class is incorrect: ", annotation_names[k])

                        if iou_prelim > iou_result:
                            iou_result = iou_prelim
                        
                        print()
                        print(iou_result)


                    if DEBUG:
                        # Wait, and destroy.
                        cv2.waitKey(10000)
                        cv2.destroyAllWindows()

                cracks.append([i, iou_result,
                               certainty,
                               pred_class_name,
                               actu_class])

                iou_result = 0
                iou_tester = 0
                iou_prelim = 0

        writer.addObject(image,
                         num_label_boxes,
                         num_pred_boxes,
                         cracks)

    writer.save("/home/osteinnes/prog/tfserving-client/output/validation_acc_output_01iou_05pred.xml")


def read_pascal_voc(xml_file: str):

    # Retrieve XML-file with element tree
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initiate empty vectors
    list_with_all_boxes = []
    list_with_all_names = []

    # Iterate through each object in the Pascal VOC xml.
    for boxes in root.iter('object'):

        # Retrieve filename
        filename = root.find('filename').text

        # Define None values.
        ymin, xmin, ymax, xmax = None, None, None, None
        name = None

        # Class name of the label in labels.
        name = str(boxes.find("name").text)
        # List of label names.
        list_with_all_names.append(name)

        # Iterate through bndbox to find limits of box.
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

            # Represent a single box's boundaries
            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            # List of boxes.
            list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_names, list_with_all_boxes


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


if __name__ == '__main__':
    main()
