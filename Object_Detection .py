import cv2
from ultralytics import YOLO
import torch
import numpy as np
import os
from datetime import datetime
import time

class EnhancedObjectDetector:
    def __init__(self):
        print("Initializing Enhanced Object Detector...")
        
        # Define object categories with their properties
        self.object_categories = {
            # Dangerous objects (danger_level >= 2)
            'gun': {'danger_level': 3, 'color': (0, 0, 255), 'track': True},
            'knife': {'danger_level': 3, 'color': (0, 0, 255), 'track': True},
            'scissors': {'danger_level': 2, 'color': (0, 165, 255), 'track': True},
            
            # Common COCO objects (danger_level = 0)
            'person': {'danger_level': 0, 'color': (0, 255, 0), 'track': True},
            'bottle': {'danger_level': 0, 'color': (255, 255, 0), 'track': True},
            'cup': {'danger_level': 0, 'color': (255, 255, 0), 'track': True},
            'watch': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'cell phone': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'laptop': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'book': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'backpack': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'handbag': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'chair': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'dining table': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'keyboard': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'mouse': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
            'remote': {'danger_level': 0, 'color': (255, 191, 0), 'track': True},
        }
        
        # Initialize YOLO model
        try:
            self.model = YOLO('yolov8n.pt')  # Using YOLOv8n as base for COCO objects
            print("YOLO model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        # Detection settings
        self.conf_threshold = 0.35  # Lower threshold for better detection
        self.iou_threshold = 0.45
        
        # Initialize directories
        self.base_dir = "detections"
        self.dangerous_dir = os.path.join(self.base_dir, "dangerous")
        self.common_dir = os.path.join(self.base_dir, "common")
        os.makedirs(self.dangerous_dir, exist_ok=True)
        os.makedirs(self.common_dir, exist_ok=True)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()




    def init_camera(self):
        """Initialize camera with optimal settings"""
        print("Initializing camera...")
        camera_methods = [
            lambda: cv2.VideoCapture(1),
            lambda: cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION),
            lambda: cv2.VideoCapture(1),
            lambda: cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        ]
        
        for method in camera_methods:
            cap = method()
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    print("Camera initialized successfully!")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                cap.release()
        
        raise RuntimeError("Could not initialize camera!")

    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        detected_objects = {
            'dangerous': [],
            'common': []
        }
        
        if detections is None or len(detections) == 0:
            return frame, detected_objects
        
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = self.model.names[int(det.cls[0])]
            
            # Check if detected object is in our categories
            if cls in self.object_categories:
                obj_info = self.object_categories[cls]
                color = obj_info['color']
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f'{cls} ({conf:.2f})'
                labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x1, y1 - labelSize[1] - 10), 
                            (x1 + labelSize[0], y1),
                            color, 
                            -1)
                
                # Draw label text
                cv2.putText(frame, 
                        label, 
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2)
                
                # Add warning for dangerous objects
                if obj_info['danger_level'] >= 2:
                    warning = "DANGER!"
                    cv2.putText(frame, 
                            warning,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2)
                    detected_objects['dangerous'].append(cls)
                else:
                    detected_objects['common'].append(cls)
        
        return frame, detected_objects

    def save_detection(self, frame, objects_detected):
        """Save frame based on detection type"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save dangerous object detections
        if objects_detected['dangerous']:
            filename = f"dangerous_{timestamp}.jpg"
            path = os.path.join(self.dangerous_dir, filename)
            cv2.imwrite(path, frame)
        
        # Save common object detections if specified
        if objects_detected['common']:
            filename = f"common_{timestamp}.jpg"
            path = os.path.join(self.common_dir, filename)
            cv2.imwrite(path, frame)

    def run_detection(self):
        """Run real-time detection"""
        try:
            cap = self.init_camera()
            frame_time = time.time()
            frames_processed = 0
            
            print("\nStarting detection...")
            print("Press 'q' to quit")
            print("Press 's' to save current frame")
            print("Press 'h' to toggle help overlay")
            
            show_help = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame!")
                    continue
                
                # Process frame
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    agnostic_nms=True
                )
                
                # Draw detections
                frame, detected_objects = self.draw_detections(frame, results[0])
                
                # Calculate FPS
                frames_processed += 1
                if time.time() - frame_time >= 1.0:
                    self.fps = frames_processed
                    frames_processed = 0
                    frame_time = time.time()
                
                # Draw overlay information
                self.draw_overlay(frame, detected_objects, show_help)
                
                # Display frame
                cv2.imshow('Enhanced Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_detection(frame, detected_objects)
                    print("Frame saved!")
                elif key == ord('h'):
                    show_help = not show_help
                
                # Auto-save dangerous detections
                if detected_objects['dangerous']:
                    self.save_detection(frame, detected_objects)
            
        except Exception as e:
            print(f"Error occurred: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nDetection stopped.")
            print(f"Detections saved in: {self.base_dir}")

    def draw_overlay(self, frame, detected_objects, show_help):
        """Draw information overlay on frame"""
        # Draw FPS
        cv2.putText(frame, f'FPS: {self.fps}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detection counts
        y_offset = 60
        if detected_objects['dangerous']:
            text = "DANGEROUS: " + ", ".join(detected_objects['dangerous'])
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        if detected_objects['common']:
            text = "Objects: " + ", ".join(detected_objects['common'])
            cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 191, 0), 2)
        
        # Show help overlay 
        if show_help:
            help_text = [
                "Controls:",
                "Q - Quit",
                "S - Save frame",
                "H - Toggle help"
            ]
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (frame.shape[1] - 200, 30 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    detector = EnhancedObjectDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()
