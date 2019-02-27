
import xml.etree.ElementTree as ET
import numpy as np
from scipy import stats


def main():

    # Path of xml.
    xml_path = "/home/osteinnes/prog/tfserving-client/output/example_output.xml"

    list_with_all_label_cracks, list_with_all_pred_cracks, list_with_iou, list_with_certainty, list_with_pred_class, list_with_actu_class = read_iou_xml(
        xml_path)

    # Print output as test.
    #print(list_with_all_label_cracks)
    #print(list_with_all_pred_cracks)
    #print(list_with_iou)
    #print(list_with_certainty)
    #print(list_with_actu_class)
    #print(list_with_pred_class)

    total_pred_cracks = sum(list_with_all_pred_cracks)
    total_actu_cracks = sum(list_with_all_label_cracks)

    print()
    print("Total number of predicted cracks: ", total_pred_cracks)
    print("Total number of labeled cracks: ", total_actu_cracks)

    averages = sum(list_with_iou, 0.0) / len(list_with_iou)

    mean = np.mean(list_with_iou)

    describe = stats.describe(list_with_iou)

    print()
    print("Average of IOU: ", averages)
    print("Mean of IOU: ", mean)
    print(describe)


    


def read_iou_xml(xml_file: str):

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
        num_label_cracks = int(images.find("data").find("num_label_cracks").text)
        num_pred_cracks = int(images.find("data").find("num_pred_cracks").text)

        # List of label names.
        list_with_all_label_cracks.append(num_label_cracks)
        list_with_all_pred_cracks.append(num_pred_cracks)

        # Iterate through bndbox to find limits of box.
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


if __name__ == '__main__':
    main()
