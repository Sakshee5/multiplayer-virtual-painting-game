import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skimage.feature import hog
import pickle
import os
import time
import traceback
import mediapipe as mp

class SimpleHandDetector:
    def __init__(self):
        # MediaPipe hand detection (only used for training data collection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7  # Increased confidence threshold
        )
        
        # HOG parameters - adjusted for better detail capture
        self.hog_pixels_per_cell = (8, 8)  # Smaller cells for more detail
        self.hog_cells_per_block = (3, 3)  # Larger blocks for better normalization
        self.hog_orientations = 12  # More orientations for better angle resolution
        self.target_size = (128, 128)  # Larger target size for more detail
        
        # Models
        self.models = None
        self.hand_present_classifier = None
        
        # Fingertip indices
        self.target_indices = [8, 12]  # index, middle
        
    def preprocess_frame(self, frame):
        """Enhanced preprocessing pipeline"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define two ranges for skin detection (handles different skin tones better)
        lower_skin1 = np.array([0, 30, 60])
        upper_skin1 = np.array([20, 150, 255])
        lower_skin2 = np.array([170, 30, 60])
        upper_skin2 = np.array([180, 150, 255])
        
        # Create masks and combine them
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (3,3), 0)
        
        # Find the largest contour (assumed to be the hand)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Create a mask with only the largest contour
            hand_mask = np.zeros_like(skin_mask)
            cv2.drawContours(hand_mask, [max_contour], -1, 255, -1)
            
            # Get bounding box of hand region
            x,y,w,h = cv2.boundingRect(max_contour)
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            
            # Extract hand region
            hand_region = frame[y:y+h, x:x+w]
            if hand_region.size == 0:
                return None, None, None
                
            # Resize to target size
            hand_region = cv2.resize(hand_region, self.target_size)
            return hand_region, hand_mask, (x,y,w,h)
        
        return None, None, None

    def extract_features(self, frame):
        """Enhanced feature extraction with multiple feature types"""
        try:
            # Get preprocessed hand region
            hand_region, hand_mask, bbox = self.preprocess_frame(frame)
            if hand_region is None:
                return None
            
            # Convert to different color spaces for robust feature extraction
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
            
            # 1. HOG features from grayscale
            hog_features = hog(gray, 
                             orientations=self.hog_orientations,
                             pixels_per_cell=self.hog_pixels_per_cell,
                             cells_per_block=self.hog_cells_per_block,
                             feature_vector=True)
            
            # 2. Color histogram features
            color_features = []
            for channel in cv2.split(hsv):
                hist = cv2.calcHist([channel], [0], None, [32], [0,256])
                color_features.extend(hist.flatten())
            
            # 3. Edge features using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_features = cv2.calcHist([edges], [0], None, [32], [0,256]).flatten()
            
            # Combine all features
            all_features = np.concatenate([hog_features, color_features, edge_features])
            
            # Normalize features
            all_features = (all_features - np.mean(all_features)) / (np.std(all_features) + 1e-7)
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            traceback.print_exc()
            return None
        
    def collect_training_data(self, num_samples=500): 
        print("Starting data collection...")
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return None, None, None
        
        features_list = []
        landmarks_list = []
        no_hand_features = []  # For negative samples
        count = 0
        no_hand_count = 0
        target_no_hand = num_samples // 5  # 20% negative samples
        
        while count < num_samples or no_hand_count < target_no_hand:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                features = self.extract_features(frame)
                if features is None:
                    continue
                
                if results.multi_hand_landmarks and count < num_samples:
                    # Get fingertip landmarks
                    landmarks = []
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Calculate hand center and size for normalization
                    all_x = [lm.x for lm in hand_landmarks.landmark]
                    all_y = [lm.y for lm in hand_landmarks.landmark]
                    hand_center_x = np.mean(all_x)
                    hand_center_y = np.mean(all_y)
                    hand_size = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
                    
                    for idx in self.target_indices:
                        # Normalize coordinates relative to hand center and size
                        norm_x = (hand_landmarks.landmark[idx].x - hand_center_x) / hand_size
                        norm_y = (hand_landmarks.landmark[idx].y - hand_center_y) / hand_size
                        landmarks.extend([norm_x, norm_y])
                    
                    features_list.append(features)
                    landmarks_list.append(landmarks)
                    count += 1
                    
                    # Visualization
                    h, w, _ = frame.shape
                    for i in range(0, len(landmarks), 2):
                        x = int((landmarks[i] * hand_size + hand_center_x) * w)
                        y = int((landmarks[i + 1] * hand_size + hand_center_y) * h)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    cv2.putText(frame, f'Samples with hand: {count}/{num_samples}', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif not results.multi_hand_landmarks and no_hand_count < target_no_hand:
                    no_hand_features.append(features)
                    no_hand_count += 1
                    cv2.putText(frame, f'Samples without hand: {no_hand_count}/{target_no_hand}', 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Collecting Training Data', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in collection loop: {str(e)}")
                traceback.print_exc()
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            print(f"Collected {count} hand samples and {no_hand_count} no-hand samples successfully!")
            return (np.array(features_list), np.array(landmarks_list), 
                   np.array(no_hand_features))
        else:
            print("No samples collected!")
            return None, None, None
            
    def train(self, features, landmarks, no_hand_features):
        print("Training models with validation...")
        
        # Combine positive and negative samples for hand detection
        X_hand_detection = np.vstack([features, no_hand_features])
        y_hand_detection = np.hstack([
            np.ones(len(features)),
            np.zeros(len(no_hand_features))
        ])
        
        # Split data into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_hand_detection, y_hand_detection, test_size=0.2, random_state=42
        )
        
        # Train hand presence classifier
        from sklearn.ensemble import GradientBoostingClassifier
        self.hand_present_classifier = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1  # Add verbose output
        )
        self.hand_present_classifier.fit(X_train, y_train)
        
        # Evaluate hand detector
        val_score = self.hand_present_classifier.score(X_val, y_val)
        print(f"Hand detector validation accuracy: {val_score:.3f}")
        
        # Train landmark predictors
        from sklearn.ensemble import GradientBoostingRegressor
        self.models = []
        
        # Split landmark data
        X_train, X_val, y_train, y_val = train_test_split(
            features, landmarks, test_size=0.2, random_state=42
        )
        
        for i in range(landmarks.shape[1]):
            print(f"Training landmark {i}...")
            model = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1,  # Add verbose output
                n_iter_no_change=10  # Add early stopping
            )
            model.fit(X_train, y_train[:, i])
            
            # Evaluate each landmark predictor
            val_score = model.score(X_val, y_val[:, i])
            print(f"Landmark {i} R² score: {val_score:.3f}")
            
            self.models.append(model)
            
        print("Training complete!")
        
    def save_models(self, filepath='hand_models.pkl'):
        with open(filepath, 'wb') as f:
            # Save both the landmark models and hand classifier
            model_package = {
                'landmark_models': self.models,
                'hand_classifier': self.hand_present_classifier
            }
            pickle.dump(model_package, f)
        print(f"Models saved to {filepath}")
        
    def load_models(self, filepath='hand_models.pkl'):
        try:
            with open(filepath, 'rb') as f:
                model_package = pickle.load(f)
                
                # Check if the loaded data is a dictionary (new format)
                if isinstance(model_package, dict):
                    self.models = model_package['landmark_models']
                    self.hand_present_classifier = model_package['hand_classifier']
                else:
                    # For backward compatibility with old saved files
                    self.models = model_package
                    print("Warning: Hand presence classifier not found in saved file.")
                    return False
                    
                print(f"Models loaded from {filepath}")
                print(f"Number of landmark models loaded: {len(self.models)}")
                return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            traceback.print_exc()
            return False
            
    def predict(self, frame):
        try:
            features = self.extract_features(frame)
            if features is None:
                print("No features extracted from frame")
                return frame, None
                
            # First check if hand is present
            if self.hand_present_classifier is not None:
                hand_confidence = self.hand_present_classifier.predict_proba([features])[0][1]
                # Only predict landmarks if confident hand is present
                if hand_confidence > 0.7:  # Confidence threshold
                    landmarks = []
                    print("Predicting landmarks...")
                    for i, model in enumerate(self.models):
                        pred = model.predict([features])[0]
                        landmarks.append(pred)
                        print(f"Landmark {i} prediction: {pred:.3f}")
                    
                    # Draw predictions with confidence
                    height, width = frame.shape[:2]
                    colors = [(0, 255, 0), (255, 0, 0)]  # Green for index, Blue for middle
                    
                    # Estimate hand size and center from the frame size
                    # Using similar proportions as during training
                    hand_size = min(width, height) * 0.3  # Assume hand takes up about 30% of frame
                    hand_center_x = width / 2
                    hand_center_y = height / 2
                    
                    for i in range(0, len(landmarks), 2):
                        # Denormalize coordinates using the same scheme as in training
                        x = int((landmarks[i] * hand_size + hand_center_x))
                        y = int((landmarks[i + 1] * hand_size + hand_center_y))
                        finger_idx = i // 2
                        print(f"Drawing {['Index', 'Middle'][finger_idx]} finger at ({x}, {y})")
                        
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame, (x, y), 8, colors[finger_idx], -1)
                            label = f"{'Index' if finger_idx == 0 else 'Middle'}"
                            cv2.putText(frame, label, (x + 10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[finger_idx], 2)
                        else:
                            print(f"Warning: Coordinates ({x}, {y}) out of frame bounds ({width}, {height})")
                    
                    cv2.putText(frame, f"Confidence: {hand_confidence:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    return frame, landmarks
                else:
                    cv2.putText(frame, "No hand detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame, None
            else:
                print("Hand classifier not loaded")
                cv2.putText(frame, "Models not properly loaded", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame, None
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            cv2.putText(frame, "Prediction error", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None
        

def main():
    try:
        detector = SimpleHandDetector()
        model_path = 'hand_models.pkl'
        
        if os.path.exists(model_path):
            print("Loading existing models...")
            if not detector.load_models(model_path):
                print("Failed to load models. Starting new training process...")
                features, landmarks, no_hand_features = detector.collect_training_data()
                if features is not None and landmarks is not None and no_hand_features is not None:
                    detector.train(features, landmarks, no_hand_features)
                    detector.save_models(model_path)
                else:
                    print("Training data collection failed.")
                    return
        else:
            print("No existing models found. Starting new training process...")
            features, landmarks, no_hand_features = detector.collect_training_data()
            if features is not None and landmarks is not None and no_hand_features is not None:
                detector.train(features, landmarks, no_hand_features)
                detector.save_models(model_path)
            else:
                print("Training data collection failed.")
                return
                
        # Run real-time detection with proper error handling
        print("Starting real-time detection. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera for detection!")
            return
            
        print("Camera opened successfully for detection.")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                    
                frame = cv2.flip(frame, 1)  # Mirror effect
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print FPS every 30 frames
                    fps = frame_count / (time.time() - start_time)
                    print(f"FPS: {fps:.1f}")
                
                processed_frame, landmarks = detector.predict(frame)
                
                # Add FPS counter
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                              (processed_frame.shape[1] - 120, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Hand Detection', processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Detection stopped by user")
                    break
                    
            except Exception as e:
                print(f"Error in detection loop: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Program ended successfully")
        
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()