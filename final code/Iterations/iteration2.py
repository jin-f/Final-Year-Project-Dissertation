import numpy as np
import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def load_model(model_name=None):
    model = None
    if model_name:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

#the object we are trying to detect
wanted_class = "banana"
confidence_level = 0.3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

model = load_model("banana.pt")
model.to(device)
classes = model.names

object_tracker = DeepSort(max_age=5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    
    bbs = [] #for tracking
    results = model(img)

    labels = results.xyxy[0][:, -1]
    data = results.xyxy[0]
    wanted_class_index = list(classes.keys())[list(classes.values()).index(wanted_class)]
    if wanted_class_index in labels:
            # program is designed to only have 1 object detected in each frame, so we take the one with the highest confidence
            confidence_list = data[:, 4]
            highest_confidence = torch.argmax(confidence_list)

            row = data[highest_confidence] #take the row with the highest confidence
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = row[4].item()
            if confidence > confidence_level:
                bbs.append(([x1,y1, int(x2-x1), int(y2-y1)], confidence, wanted_class))

    tracks = object_tracker.update_tracks(bbs, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
    
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()
