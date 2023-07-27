from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('C:\YOLOv8_and_Canny_Edge\Plate-Character-Recognition-with-Canny-Edge\\runs\\detect\\train3\\weights\\last.pt')

# load video
cap = cv2.VideoCapture('C:\YOLOv8_and_Canny_Edge\Plate-Character-Recognition-with-Canny-Edge\\license_dataset\\videos\\testVideo1080-60.MOV')

vehicles = [3]

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

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]


                license_plate_crop_blurred = cv2.GaussianBlur(license_plate_crop, (5, 5), 0)
                license_plate_crop_canny = cv2.Canny(license_plate_crop_blurred, 30, 100, 1)

                # plt.imshow(license_plate_crop, cmap='gray')
                # plt.title('Canny Edge Detection')
                # plt.axis('off')
                # plt.show()

                # plt.imshow(license_plate_crop_canny, cmap='gray')
                # plt.title('Canny Edge Detection')
                # plt.axis('off')
                # plt.show()

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_canny)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

#  write results
write_csv(results, './test-2.csv')