import numpy as np
import torch

# function below is from https://github.com/Nevaera/SocialDistance-ZED-SDK-YOLOv3-DeepSORT/tree/master
def load_image_into_numpy_array(image):
    '''
        Loads an image from the ZED into a uint numpy array for use with image processing
    '''
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)
    

class Detector():
    """
    Class for YOLOv5 Detector

    ...

    Attributes
    ----------
    model : torch hub model
        YOLOv5 model 
    model_name : str
        name of the YOLOv5 model
    wanted_class : str
        The class of object we are detecting for
    confidence level : float
        minimum confidence level to accept detections

    """
    def __init__(self, wanted_class, model_name=None, confidence_level=0.5) -> None:
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        self.confidence_level = confidence_level
        self.wanted_class = wanted_class
        self.class_names = self.model.names
        self.class_name_index = list(self.class_names.keys())[list(self.class_names.values()).index(self.wanted_class)]
        

    def load_model(self):
        if self.model_name:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="actual 1-3.pt", force_reload=True)

        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.to(self.device)
        print(self.device)

    def get_features(self, image_patch):
        return self.model(image_patch)['model'][0][-1]
    
    def get_model(self):
        return self.model

    def detect(self, image_frame):
        """
        Runs detections on the given image and outputs a list of detections in the format for deepsort tracker

        Parameters
        ----------
        image_frame : np array
            image that we are performing detections on

        Returns
        -------
        bbs : list
            A list of tuples, (left, top, confidence, class), where each tuple is a detection in the image
        """
        results = self.model(image_frame)
        labels = results.xyxy[0][:, -1]
        data = results.xyxy[0]

        # removing rows in the data which aren't classes that we are looking for
        # this applies to models which have more than 1 class trained
        data = data[data[:, 5] == self.class_name_index]
        bbs = []

        if self.class_name_index in labels:
            # program is designed to only have 1 object detected in each frame, so we take the one with the highest confidence
            confidence_list = data[:, 4]
            highest_confidence = torch.argmax(confidence_list)

            row = data[highest_confidence] #take the row with the highest confidence
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = row[4].item()
            if confidence > self.confidence_level:
                bbs.append(([x1,y1, int(x2-x1), int(y2-y1)], confidence, self.wanted_class))

        return bbs
