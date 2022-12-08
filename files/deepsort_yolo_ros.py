#!/usr/bin/env python3
# ROS MODULES
import sys
import os
import random
import time
import cv2
import numpy as np
import rospy
import roslib
roslib.load_manifest('deepsort_yolo_ros')
from sensor_msgs.msg import Image
from darknet_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
from pathlib import Path
#sys.path.remove(os.path.dirname(__file__))

# YOLOv4 & TENSORFLOW MODULES
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Comment out to enable tensorflow logging outputs
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes, YOLO, decode
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# DEEP SORT MODULES
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
################################################################################
def main():
    config = Config()
    m = ModelConfig(config)
    m.buildModel()
    deepdarkros = DeepSortDarknetROS(config)

    rospy.init_node('deepsort_yolo_ros', anonymous=True)

    # Loop here to send/receive images
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("NODE ERROR")
    rospy.loginfo("SHUTTING DOWN")
    if deepdarkros.videoFlag and deepdarkros.initFlag:
        deepdarkros.video.release()
    cv2.destroyAllWindows()

# Configuration Class
class Config:
    def __init__(self):
        self.framework = "tf" #tensorflow framework (tf, tflite, trt)
        self.weights = "" #path to weights file
        self.tiny = rospy.get_param('/deepsort_yolo_ros/yolo_tiny_flag')  # YOLO or YOLO-tiny
        self.size = 416 #resize images to
        self.model = "yolov4" #yolov3 or yolov4
        self.iou = 0.45 #iou threshold
        self.score = 0.5 #score threshold

class ModelConfig:
    def __init__(self, config):
        self.tiny = config.tiny #'is yolo-tiny or not')
        if self.tiny is True:
            self.weights = "/darknet/yolov4-tiny.weights"
            self.output = str(Path(__file__).parent)+"/checkpoints/yolov4-tiny-416"
        else:
            self.weights = "/darknet/yolov4.weights"
            self.output = str(Path(__file__).parent)+"/checkpoints/yolov4-416"
        self.input_size = 416 # define input size of export model
        self.score_thres = 0.2 # define score threshold
        self.framework = config.framework # define what framework do you want to convert (tf, trt, tflite)
        self.model = config.model # yolov3 or yolov4

    def buildModel(self):
        if self.checkBuilt(self.model, self.tiny) is True:
            return 0
        else:
            STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self)

            input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
            feature_maps = YOLO(input_layer, NUM_CLASS, self.model, self.tiny)
            bbox_tensors = []
            prob_tensors = []
            if self.tiny:
                for i, fm in enumerate(feature_maps):
                  if i == 0:
                    output_tensors = decode(fm, self.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                  else:
                    output_tensors = decode(fm, self.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                  bbox_tensors.append(output_tensors[0])
                  prob_tensors.append(output_tensors[1])
            else:
                for i, fm in enumerate(feature_maps):
                  if i == 0:
                    output_tensors = decode(fm, self.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                  elif i == 1:
                    output_tensors = decode(fm, self.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                  else:
                    output_tensors = decode(fm, self.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.framework)
                  bbox_tensors.append(output_tensors[0])
                  prob_tensors.append(output_tensors[1])
            pred_bbox = tf.concat(bbox_tensors, axis=1)
            pred_prob = tf.concat(prob_tensors, axis=1)
            if self.framework == 'tflite':
                pred = (pred_bbox, pred_prob)
            else:
                boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=self.score_thres, input_shape=tf.constant([self.input_size, self.input_size]))
                pred = tf.concat([boxes, pred_conf], axis=-1)
            model = tf.keras.Model(input_layer, pred)
            utils.load_weights(model, self.weights, self.model, self.tiny)
            model.summary()
            model.save(self.output)
            with open("/catkin/src/pedestrian_tracker/deepsort_yolo_ros/src/buildLog.txt", 'w') as file:
                lines = [self.model, str(self.tiny)]
                file.write('\n'.join(lines))
                file.close()

    def checkBuilt(self, model, tiny):
        try:
            f = open("/catkin/src/pedestrian_tracker/deepsort_yolo_ros/src/buildLog.txt", "r")
            temp = f.read().splitlines()
            f.close()
            if len(temp) == 2 and temp[0] == model and temp[1] == str(tiny):
                print("MODEL ALREADY EXISTS:", model, "TINY:", tiny)
                return True
            else:
                print("INCORRECT EXISTING MODEL:", model, "TINY:",tiny)
                print("BUILDING MODEL")
                return False
        except IOError:
            print("BUILDING MODEL:", model, "TINY:", tiny)
            f = open("/catkin/src/pedestrian_tracker/deepsort_yolo_ros/src/buildLog.txt", "w+")
            f.close()
            return False

class DeepSortDarknetROS:
    def __init__(self, config):
        self.pub = rospy.Publisher("bounding_boxes",BoundingBoxes, queue_size=1)
        self.sub = rospy.Subscriber("pipeline_image",Image,self.callback)
        #self.sub = rospy.Subscriber("/gmsl/A0/image_color",Image,self.callback)
        #self.sub = rospy.Subscriber("/gmsl/A1/image_color",Image,self.callback)
        #self.sub = rospy.Subscriber("/gmsl/A2/image_color",Image,self.callback)
        self.config = config
        self.bridge = CvBridge()

        self.initFlag = False

        self.bboxFlag = rospy.get_param('/deepsort_yolo_ros/bbox_flag') # Display detected bounding boxes
        self.trackFlag = rospy.get_param('/deepsort_yolo_ros/track_flag') # Display tracked bounding boxes
        self.displayFlag = rospy.get_param('/deepsort_yolo_ros/display_flag') # Display results
        self.videoFlag = rospy.get_param('/deepsort_yolo_ros/video_flag')  # Save output video
        self.output_file = rospy.get_param('/deepsort_yolo_ros/output_file')

        self.frame_id = 1
        self.prevTime = 0

        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        self.nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = str(Path(__file__).parent)+'/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric)

        if self.config.tiny is True:
            self.config.weights = str(Path(__file__).parent)+"/checkpoints/yolov4-tiny-416"
        else:
            self.config.weights = str(Path(__file__).parent)+"/checkpoints/yolov4-416"

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.config)

        # load tflite model if flag is set
        if self.config.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.config.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            self.saved_model_loaded = tf.saved_model.load(self.config.weights, tags=[tag_constants.SERVING])
            self.infer = self.saved_model_loaded.signatures['serving_default']

        # Read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # Set Allowed Classes
        #self.allowed_classes = list(class_names.values()) # Allow all classes in .names file
        #self.allowed_classes = ['person']
        self.allowed_classes = ['traffic light']
        self.class_colours = {"traffic light": (255, 255, 0)}

    def callback(self, image):
        #rospy.loginfo(rospy.get_caller_id())
        rospy.loginfo("RECEIVED IMAGE")

        try:
           frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
           print(e)

        if self.initFlag is False:
            self.height, self.width, self.channels = frame.shape
            print('INITIALISED')
            print('VIDEO PROPERTIES:')
            print('     WIDTH:   ', self.width)
            print('     HEIGHT:  ', self.height)
            print('     CHANNELS:', self.channels)
            if self.videoFlag:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video = cv2.VideoWriter(self.output_file, fourcc, 30, (self.width, self.height))
            self.initFlag = True

        print("STAMP: "+str(image.header.stamp.secs)+"."+str(image.header.stamp.nsecs))
        print("SEQ:",image.header.seq)
        print("FRAME:",self.frame_id)
        self.frame_id += 1
        #print("TIME:",rospy.Time.now())

          # Process Image
        prev_time = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_size = frame.shape[:2]
        input_size = self.config.size
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # run detections on tflite if flag is set
        if self.config.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if self.config.model == 'yolov3' and self.tinyFlag == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = self.infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.config.iou,
            score_threshold=self.config.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            if class_name not in self.allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        #cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # Create Bounding Box Messages
        boundingboxes = BoundingBoxes()
        boundingboxes.header = image.header
        boundingboxes.image_header = image.header

        # CREATE YOLO BOUNDING BOXES
        print("DETECTION COUNT:", len(detections))
        for detection in detections:
            bbox = detection.to_tlbr()

            if self.bboxFlag:
                #colour = (255, 255, 0) # Colour all boxes yellow
                colour = self.class_colours[detection.class_name]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colour, 2)
                cv2.putText(frame, str(round(detection.confidence*100,2))+"%",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,0),2)

            bboxmsg = BoundingBox()
            bboxmsg.xmin = int(bbox[0])
            bboxmsg.ymax = int(bbox[3])
            bboxmsg.xmax = int(bbox[2])
            bboxmsg.ymin = int(bbox[1])
            bboxmsg.confidence = detection.confidence
            bboxmsg.class_name = detection.class_name
            bboxmsg.id = -1

            ##print(detection.confidence)

            boundingboxes.bounding_boxes.append(bboxmsg)

        # Create Colour Map
        cmap = plt.get_cmap('tab20b')
        colours = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # CREATE DEEP SORT BOUNDING BOXES
        print("TRACK COUNT:", len(self.tracker.tracks))
        if len(self.tracker.tracks) > 0:
            print("TRACKED BOUNDING BOX COORDINATES:")
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if self.trackFlag:
                colour = colours[int(track.track_id) % len(colours)]
                colour = [i * 255 for i in colour]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colour, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), colour, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            print("   "+class_name+"-"+str(track.track_id)+":", "xmin="+str(int(bbox[0])),"ymin="+str(int(bbox[1])),"xmax="+str(int(bbox[2])),"ymax="+str(int(bbox[3])))

            bboxmsg = BoundingBox()
            bboxmsg.xmin = int(bbox[0])
            bboxmsg.ymax = int(bbox[3])
            bboxmsg.xmax = int(bbox[2])
            bboxmsg.ymin = int(bbox[1])
            #bboxmsg.confidence = track.confidence
            bboxmsg.confidence = -1
            bboxmsg.class_name = class_name
            bboxmsg.id = track.track_id

            boundingboxes.bounding_boxes.append(bboxmsg)

        #if args.save_labels:
            #save_annotations(image_name, image, detections, class_names)
        #darknet.print_detections(detections, args.ext_output)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = round((1/(time.time() - prev_time)),1)
        #fps = int(1/(time.time() - self.prevTime))
        #self.prevTime = time.time()
        print("FPS:",fps)
        if self.displayFlag:
            window_name = "DeepSort + YOLO4"
            cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
            #cv2.moveWindow(window_name, 0,0)
            #cv2.resizeWindow(window_name, 640, 480)
            #cv2.resizeWindow(window_name, 1280,1024)
            cv2.resizeWindow(window_name, 1280,800)
            cv2.imshow(window_name, result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        if self.videoFlag:
            self.video.write(result);

        # Publish Bounding Boxes
        try:
            self.pub.publish(boundingboxes)
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
