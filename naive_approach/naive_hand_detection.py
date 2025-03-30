"""
NaiveHandDetector Class:
---------------------------
Uses HSV color space for skin detection
Defines a range of skin colors (can be adjusted)
Uses morphological operations to clean up the detection
Detects hand contours and finds fingers using convexity defects
Draws the detected hand outline and finger count


Main Features:
----------------------------
Real-time webcam processing
Skin color segmentation
Contour detection for hand shape
Finger counting using convexity defects
Visual feedback with contour drawing and finger count
"""

import cv2
import numpy as np

class NaiveHandDetector:
    def __init__(self):
        # Define the range of skin color in HSV
        # These values can be adjusted based on testing
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Morphological operation kernels
        self.kernel = np.ones((3,3), np.uint8)
        
    def detect_hand(self, frame):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.dilate(mask, self.kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5,5), 100)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assuming it's the hand)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # Only process if the contour is large enough (to avoid noise)
            if area > 5000:
                # Get the convex hull and defects
                hull = cv2.convexHull(max_contour)
                hull = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexityDefects(max_contour, hull)
                
                # Draw the contour and convex hull
                cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                
                # Count fingers using convexity defects
                finger_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])
                        
                        # Calculate angle between fingers
                        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
                        
                        # If angle is less than 90 degrees, treat as finger
                        if angle <= np.pi/2:
                            finger_count += 1
                            cv2.circle(frame, far, 5, [0, 0, 255], -1)
                
                # Draw finger count
                cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                return frame, mask, True
                
        return frame, mask, False

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = NaiveHandDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        processed_frame, mask, hand_detected = detector.detect_hand(frame)
        
        # Show the original frame and the mask
        cv2.imshow('Hand Detection', processed_frame)
        cv2.imshow('Mask', mask)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 