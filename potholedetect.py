######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys


CLASS_MAP = {
    1.0: 'Pothole',
    2.0: 'Car',
    3.0: 'Potholegroup',
    4.0: 'Vehicle'
}

# Draw the predicted bounding box


def drawPred(frame, labels, classId, conf, top, left, bottom, right):
    # Draw a bounding box.
    # y1 x1 y2 x2
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    className = CLASS_MAP[classId]  # self.yolo.class_names[classId]
    label = '%s %.2f' % (className, conf)
    score = label
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                  (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 1)
    labels.append({
        'label':
        className,
        'score':
        score,
        'sector':
        sector(left, top, frame.shape[0], frame.shape[1])
    })


def sector(left, top, frameHeight, frameWidth):
    verticalGrid = ['top', 'middle', 'bottom']
    horizontalGrid = ['left', 'center', 'right']
    normalizedX = left / (frameWidth / 3.0)
    normalizedY = top / (frameHeight / 3.0)
    x_idx = min(int(normalizedX), 2)  # 0: left, 1: center, 2: right
    y_idx = min(int(normalizedY), 2)  # 0: top, 1: middle, 2: bottom
    return [verticalGrid[y_idx], horizontalGrid[x_idx]]


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file (not in repo!)
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

gpu_frac = 0.5
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.force_gpu_compatible = True
    if gpu_frac is not None:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config, graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def detectPotholes(imagePath):
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(imagePath)
    if image is None or len(image) == 0:
        return
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    out_boxes, out_scores, out_classes, num = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    labels = []
    height = len(image)
    width = len(image[0])
    # For example, if an image is 100 x 200 pixels (height x width)
    # and the bounding box is [0.1, 0.2, 0.5, 0.9],
    # the upper-left and bottom-right coordinates of the bounding box will be
    # (40, 10) to (180, 50) (in (x,y) coordinates).
    print("out_boxes has %d items, and 0th has %d subitems" % (len(out_boxes), len(out_boxes[0])))
    print("out_classes has %d items, and 0th has %d subitems" % (len(out_classes), len(out_classes[0])))
    min_score = 0.4
    for i in range(len(out_boxes[0])):
        print(out_boxes[0][i])
        roi = out_boxes[0][i]
        classid = out_classes[0][i]
        score = out_scores[0][i]
        if score > min_score:
            top, left, bottom, right = roi
            drawPred(image, labels, classid, score, int(top*height), int(left*width), int(bottom*height), int(right*width))

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)


IMAGE_DIR = 'pothole_image_data'
#IMAGE_NAME = '630.jpg'
items = os.listdir(IMAGE_DIR)

for imageName in items:
    # Path to image
    imagePath = os.path.join(CWD_PATH, IMAGE_DIR, imageName)
    detectPotholes(imagePath)


# Clean up
cv2.destroyAllWindows()
