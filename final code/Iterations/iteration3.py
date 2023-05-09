import pyzed.sl as sl
import cv2
import torch
import numpy as np

import time
import math


#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'custom', path="banana.pt", force_reload=True)
class_names = model.names

# function below is from https://github.com/Nevaera/SocialDistance-ZED-SDK-YOLOv3-DeepSORT/tree/master
def load_image_into_numpy_array(image):
    '''
        Loads an image from the ZED into a uint numpy array for use with image processing
    '''
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)
    
# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30 
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_maximum_distance = 3
init_params.depth_minimum_distance = 1



err = zed.open(init_params)
print(err)
if err != sl.ERROR_CODE.SUCCESS:
        exit(1)


runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
runtime.confidence_threshold = 100
runtime.textureness_confidence_threshold = 100
image = sl.Mat()
point_cloud = sl.Mat()


from deep_sort_realtime.deepsort_tracker import DeepSort

object_tracker = DeepSort(max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None)

wanted_class = 'banana'
class_name_index = list(class_names.keys())[list(class_names.values()).index(wanted_class)]


key = ''
while key != 113:
    start = time.perf_counter()

    err = zed.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        image_np = load_image_into_numpy_array(image)

        bbs = [] #used for object tracking later

        results = model(image_np)
        labels = results.xyxy[0][:, -1]
        data = results.xyxy[0]
 

        # removing rows in the data which aren't classes that we are looking for
        # this applies to models which have more than 1 class trained
        data = data[data[:, 5] == class_name_index]

        if class_name_index in labels:
            confidence_list = data[:, 4]
            highest_confidence = torch.argmax(confidence_list)

            row = data[highest_confidence] #take the row with the highest confidence
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = row[4].item()
            bbs.append(([x1,y1, int(x2-x1), int(y2-y1)], confidence, wanted_class))

        # performing object tracking
        # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

        tracks = object_tracker.update_tracks(bbs, frame=image_np)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            bbox = ltrb

            xc = int((bbox[0] + bbox[2])/2)
            yc = int((bbox[1] + bbox[3])/2)

            pc_err, centre3D = point_cloud.get_value(xc,yc)
            if not np.isnan(centre3D).all():
                # xyz world coordinates in meters
                xm = centre3D[0]
                ym = centre3D[1]
                zm = centre3D[2]
                

                #prints the depth of the bbox object
                zc_dist = math.sqrt(centre3D[0]*centre3D[0] + centre3D[1]*centre3D[1] + centre3D[2]*centre3D[2])
                cv2.putText(image_np, str(round(zc_dist,2)), (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
            cv2.rectangle(image_np,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(image_np, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time


        cv2.putText(image_np, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow("ZED", image_np)
        key = cv2.waitKey(5)    
    else:
            key = cv2.waitKey(5)
cv2.destroyAllWindows()

zed.close()
