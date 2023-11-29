import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import os

class ObjectDetection:
    
    def __init__(self):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')
        self.counter = 0
        # font 
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        # fontScale 
        self.fontScale = 1
    
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
                    center_x = int((xyxy[0][0] + xyxy[0][2])/2)
                    center_y = int((xyxy[0][1] + xyxy[0][3])/2)
                    cv2.circle(image, (center_x, center_y), radius=0, color=(5, 94, 255), thickness=5)
                    cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])-25), (int(xyxy[0][0])+100, int(xyxy[0][1])), (5, 94, 255), -1)
                    cv2.putText(image, f"{center_x}, {center_y}", (int(xyxy[0][0]), int(xyxy[0][1])-6), self.font, self.fontScale, (255, 255, 255), 2, cv2.LINE_AA) 
                    
                
            path = f'centroids/{self.counter}.jpg'
            print(path)
            self.counter += 1
            cv2.imwrite(path, image) 
            
        
        return image, xyxys, confidences
    
detection = ObjectDetection()
detection.model = detection.load_model()
images = os.listdir('test/images')
for i in images:
    image= cv2.imread(f'test/images/{i}')
    results = detection.predict(image)
    detection.plot_bboxes(results, image)
    
    