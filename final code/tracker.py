import pyzed.sl as sl
import cv2
import numpy as np

import time

import socket

from library import *

from deep_sort_realtime.deepsort_tracker import DeepSort


# global variables
image = sl.Mat()
image_np = None
point_cloud = sl.Mat()

# using udp
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sending to this port, ensure port is free
server_address_port = ("127.0.0.1", 8051)

def zed_initialise():
     # Create a ZED camera object
    zed = sl.Camera()
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30 
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP # Unity coordinate system
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 3
    init_params.depth_minimum_distance = 1


    #checks if the camera is opened successfully, if fails means camera is not connected correctly
    err = zed.open(init_params)
    print(err)
    if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime.confidence_threshold = 100
    runtime.textureness_confidence_threshold = 100

    return zed, runtime

def send_3D_coords(bbox):
    xc = int((bbox[0] + bbox[2])/2)
    yc = int((bbox[1] + bbox[3])/2)

    # depth calculations from zed camera
    pc_err, centre3D = point_cloud.get_value(xc,yc)
    if not np.isnan(centre3D).all():

        # xyz world coordinates in meters
        xm = centre3D[0]
        ym = centre3D[1]
        zm = centre3D[2]
        
        # data to be sent to port
        coord_data = str(round(xm, 2)) + "," + str(round(ym, 2)) + "," + str(round(zm, 2))
        coord_data = str.encode(coord_data)
        sock.sendto(coord_data, server_address_port)
    
    

def main():
    # initialise detector
    model = Detector(confidence_level=0.3, wanted_class='person')
    

    # initialising deepsort tracker
    object_tracker = DeepSort(max_age=5)
    
    # initialise zed camera
    zed, runtime = zed_initialise()
    
    key = ''
    while key != 113:
        start = time.perf_counter()

        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            image_np = load_image_into_numpy_array(image)

            

            # object detection performed and outputs the detections in the format used for tracking
            # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            bbs = model.detect(image_np)
        
            # object tracking
            tracks = object_tracker.update_tracks(bbs, frame=image_np)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = ltrb
                
                send_3D_coords(bbox)
                
                cv2.rectangle(image_np,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
                cv2.putText(image_np, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
            #displays fps
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

if __name__ == '__main__':
    main()












