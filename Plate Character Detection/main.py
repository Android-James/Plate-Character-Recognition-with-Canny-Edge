from ultralytics import YOLO
import cv2

results = {}


# load models
#model1 (pre-trained on COCO2017 for cars detection)
coco_model = YOLO('yolov8n.pt') 

#model2 (pre-trained on license plates dataset for license plates detection)

# load video
cap = cv2.VideoCapture('/')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])



# load video



# read frames

# detect vehicles
       
# track vehicles

# detect license plates
      

# assign license plate to car


# crop license plate

# process license plate

# canny edge

# read license plate number

# write results
