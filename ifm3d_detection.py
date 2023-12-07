import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import os
from ifm3dpy.device import O3R
from ifm3dpy.framegrabber import FrameGrabber, buffer_id

class ObjectDetection:
    
    def __init__(self):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')
        self.counter = 0
        # font 
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # fontScale 
        self.fontScale = 1
        self.safety_area_xy1 = (0, 200)
        self.safety_area_xy2 = (640, 350)
        self.center_safety_x = int((self.safety_area_xy1[0] + self.safety_area_xy2[0])/2)
        self.center_safety_y = int((self.safety_area_xy1[1] + self.safety_area_xy2[1])/2)
    
    def load_model(self):
        
        model = YOLO('cable_detector.pt') # load model
        model.fuse() # fuse for speed
        
        return model
    
    def predict(self, image):
        
        results = self.model(image)
        
        return results
    
    def plot_bboxes(self, results, image):
        
        xyxys = []
        confidences = []
        #extract detections
        for r in results:
            boxes = r.boxes.cpu().numpy()
        
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            
            if len(boxes.xyxy) > 0:
                for xyxy in xyxys:
                    cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), (5, 94, 255), 2)
                    # center_x = int((xyxy[0][0] + xyxy[0][2])/2)
                    center_y = int((xyxy[0][1] + xyxy[0][3])/2)
                    # cv2.circle(image, (center_x, center_y), radius=0, color=(5, 94, 255), thickness=5)

                    # cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])-25), (int(xyxy[0][0])+100, int(xyxy[0][1])), (5, 94, 255), -1)
                    # cv2.putText(image, f"{center_x}, {center_y}", (int(xyxy[0][0]), int(xyxy[0][1])-6), self.font, self.fontScale, (255, 255, 255), 2, cv2.LINE_AA) 
                    
                    distance_y = center_y - self.center_safety_y
                    # cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][3])), (int(xyxy[0][0])+300, int(xyxy[0][3])+25), (5, 94, 255), -1)
                    # cv2.putText(image, f"distance in y = {distance_y} px", (int(xyxy[0][0]+10), int(xyxy[0][3])+20), self.font, self.fontScale, (255, 255, 255), 2, cv2.LINE_AA)
                    print(f"distance in y = {distance_y} px")
                    

        return image, xyxys, confidences



detection = ObjectDetection()
detection.model = detection.load_model()

# Initialize the objects
o3r = O3R()
fg = FrameGrabber(o3r, pcic_port=50010)

# Change port to RUN state
config=o3r.get()
config["ports"]["port0"]["state"]= "RUN"
o3r.set(config)

# Register a callback and start streaming frames
fg.start([buffer_id.JPEG_IMAGE])

while(1):
    [ok, frame] = fg.wait_for_frame().wait_for(1500)  # wait with 1500ms timeout

    # Check that a frame was received
    if not ok:
        raise RuntimeError("Timeout while waiting for a frame.")

    # Read the distance image and display a pixel in the center
    rgb = cv2.imdecode(frame.get_buffer(buffer_id.JPEG_IMAGE), cv2.IMREAD_UNCHANGED)


    results = detection.predict(rgb)
    image, xyxys, confidences = detection.plot_bboxes(results, rgb)
    cv2.rectangle(image, (0, 200), (rgb.shape[1], 350), (0, 255, 0), 1)
    cv2.imshow("2D image", image)
    cv2.waitKey(1)

# Stop the streaming
fg.stop()

    