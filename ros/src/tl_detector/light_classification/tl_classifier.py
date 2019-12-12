from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys

import tensorflow as tf

from io import StringIO

from PIL import Image

import cv2

import label_map_util
import visualization_utils as vis_util

import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        cwd = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_LABELS = cwd + '/graph/label_map.pbtxt'
        GRAPH_FILE = cwd + '/graph/frozen_inference_graph.pb'

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=4, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph) 
                tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

        self.boxed_image = None

        rospy.logwarn("Classifier initialized")

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image, boxes, classes, scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=6)
        self.boxed_image = image
        if (scores[0]>0.2):
            rospy.logwarn("Classes: {}, Scores: {}".format(classes[0], scores[0]))
            if (classes[0] == 1):
                return TrafficLight.RED
            elif (classes[0] == 2):
                return TrafficLight.YELLOW
            elif (classes[0] == 3):
                return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
        
