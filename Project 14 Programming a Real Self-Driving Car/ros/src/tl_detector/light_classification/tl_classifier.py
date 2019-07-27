from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        red_lb = np.array([0, 100, 100],np.uint8)
        red_ub = np.array([10, 255, 255],np.uint8)
        thresh = cv2.inRange(hsv, red_lb, red_ub)
        if cv2.countNonZero(thresh) > 50:
            return TrafficLight.RED
        return TrafficLight.UNKNOWN