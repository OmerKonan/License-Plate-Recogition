import numpy as np
import tensorflow as tf
import cv2
import time
import os


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":  
    model_path = '/home/konan/License-Plate-Recognition/inference_graph/frozen_inference_graph.pb'
    saved_path = '/home/konan/License-Plate-Recognition/img_plate/'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    input_path = '/home/konan/License-Plate-Recognition/test_img'
    image_names = os.listdir(input_path)
    print("Program started....")
    for image_name in image_names:
        if image_name[-4:] == ".png" or image_name[-4:] == ".jpg":
            img = cv2.imread(input_path + "/" + image_name)
            img = cv2.resize(img, (1280, 720))
            boxes, scores, classes, num = odapi.processFrame(img)

            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    roi = img[box[0]:box[2],box[1]:box[3]]
                    #cv2.imwrite((saved_path + image_name), roi)
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                cv2.imshow("preview", img)
                print("#---------------------------#")
                print("Press q for next images")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break
                    
#############################
#------For video input------#
#############################
    """
    cap = cv2.VideoCapture('/path/to/input/video')

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                elif key & 0xFF == ord('q'):
                    break
    """