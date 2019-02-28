import xml.etree.ElementTree as ET
import numpy as np
from scipy import stats


class Accuracy_Calc:

    def __init__(self, xml_path, output_filename):

        # Retrieving data from IOU xml.
        list_with_all_label_cracks, list_with_all_pred_cracks, list_with_iou, list_with_certainty, list_with_pred_class, list_with_actu_class = self.read_iou_xml(
            xml_path)

        # Do some necessary calculations for data representation
        total_pred_cracks = sum(list_with_all_pred_cracks)
        total_actu_cracks = sum(list_with_all_label_cracks)
        total_missed_cracks = total_actu_cracks - total_pred_cracks
        if total_missed_cracks > 0:
            for i in range(total_missed_cracks):
                list_with_iou.append(0)

        # Find statistics
        describe = stats.describe(list_with_iou)
        nobs, minmax, mean, variance, skewness, kurtosis = describe

        # Write output
        self.write_output_file(output_filename, xml_path, total_pred_cracks, total_actu_cracks,
                          total_missed_cracks, nobs, minmax, mean, variance, skewness, kurtosis)

    def read_iou_xml(self, xml_file: str):

        # Retrieve XML-file with element tree
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Initiate empty vectors
        list_with_all_label_cracks = []
        list_with_all_pred_cracks = []
        list_with_iou = []
        list_with_certainty = []
        list_with_pred_class = []
        list_with_actu_class = []

        # Iterate through each object in the Pascal VOC xml.
        for images in root.iter('object'):

            # Retrieve filename
            filename = images.find('filename').text

            # Define None values.
            iou = None
            certainty = None
            pred_class = None
            actu_class = None
            num_label_cracks = None
            num_pred_cracks = None

            # Class name of the label in labels.
            num_label_cracks = int(images.find(
                "data").find("num_label_cracks").text)
            num_pred_cracks = int(images.find(
                "data").find("num_pred_cracks").text)

            # List of label names.
            list_with_all_label_cracks.append(num_label_cracks)
            list_with_all_pred_cracks.append(num_pred_cracks)

            # Iterate through crack to find values for each image.
            for crack in images.find("data").findall("crack"):
                iou = float(crack.find("iou").text)
                certainty = float(crack.find("certainty").text)
                pred_class = str(crack.find("pred_class").text)
                actu_class = str(crack.find("actu_class").text)

                list_with_iou.append(iou)
                list_with_certainty.append(certainty)
                list_with_pred_class.append(pred_class)
                list_with_actu_class.append(actu_class)

        return list_with_all_label_cracks, list_with_all_pred_cracks, list_with_iou, list_with_certainty, list_with_pred_class, list_with_actu_class

        def write_output_file(self, output_filename, xml_path, total_pred_cracks, total_actu_cracks, total_missed_cracks, nobs, minmax, mean, variance, skewness, kurtosis):
            
            # Create file for output.
            output_file = open(output_filename, "w+")
            
            output_file.write("ACCURACY OF OBJECT DETECTION MODEL \n")
            output_file.write(
                "----------------------------------------------------------\n")
            output_file.write("Path of XML: " + xml_path + "\n")
            output_file.write(
                "----------------------------------------------------------\n")
            output_file.write(
                "Total number of predicted crakcs: " + str(total_pred_cracks) + "\n")
            output_file.write(
                "Total number of labeled cracks: " + str(total_actu_cracks) + "\n")
            output_file.write("Total number of missed cracks: " +
                              str(total_missed_cracks) + "\n")
            output_file.write(
                "----------------------------------------------------------\n")

            output_file.write("See scipy.describe for documentation\n")
            output_file.write(
                "----------------------------------------------------------\n")
            output_file.write("Nobs: " + str(nobs) + "\n")
            output_file.write("MinMax: " + str(minmax) + "\n")
            output_file.write("Mean: " + str(mean) + "\n")
            output_file.write("Variance: " + str(variance) + "\n")
            output_file.write("Skewness: " + str(skewness) + "\n")
            output_file.write("Kurtosis: " + str(kurtosis) + "\n")


if __name__ == '__main__':
    Accuracy_Calc("/home/osteinnes/prog/tfserving-client/output/example_output.xml",
                  "output/train_output_acc.txt")
