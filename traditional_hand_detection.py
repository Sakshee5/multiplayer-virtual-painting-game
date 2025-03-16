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
    
        self.target_indices = [8]  # index finger only
        
    def preprocess_frame(self, frame):
        """Enhanced preprocessing pipeline for better hand segmentation"""
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # Improved HSV thresholds for skin detection
            lower_hsv1 = np.array([0, 30, 60])
            upper_hsv1 = np.array([20, 150, 255])
            lower_hsv2 = np.array([170, 30, 60])
            upper_hsv2 = np.array([180, 150, 255])
            
            # More accurate YCrCb thresholds
            lower_ycrcb = np.array([0, 135, 85])
            upper_ycrcb = np.array([255, 180, 135])
            
            # Create masks
            mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
            mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
            skin_mask = cv2.bitwise_and(skin_mask, mask_ycrcb)
            
            # More refined morphological operations
            kernel_small = np.ones((3,3), np.uint8)
            kernel_med = np.ones((5,5), np.uint8)
            kernel_large = np.ones((7,7), np.uint8)
            
            # Remove small noise
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Fill small holes
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_med)
            
            # Attempt to connect nearby skin regions
            skin_mask = cv2.dilate(skin_mask, kernel_small, iterations=1)
            skin_mask = cv2.erode(skin_mask, kernel_small, iterations=1)
            
            # Smooth edges
            skin_mask = cv2.GaussianBlur(skin_mask, (5,5), 0)
            
            # Find contours with better filtering
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Filter contours by area and aspect ratio
                min_area = 1000
                valid_contours = []
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue
                        
                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    
                    # Typical hand aspect ratios
                    if 0.5 <= aspect_ratio <= 2.0:
                        valid_contours.append((cnt, area))
                
                if valid_contours:
                    # Sort by area
                    valid_contours.sort(key=lambda x: x[1], reverse=True)
                    max_contour = valid_contours[0][0]
                    
                    # Create mask with only the largest contour
                    hand_mask = np.zeros_like(skin_mask)
                    cv2.drawContours(hand_mask, [max_contour], -1, 255, -1)
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(max_contour)
                    
                    # Add padding with bounds checking
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    # Extract hand region
                    hand_region = frame[y:y+h, x:x+w]
                    if hand_region.size == 0:
                        return None, None, None, None
                    
                    # Resize to target size
                    hand_region = cv2.resize(hand_region, self.target_size)
                    
                    # Create enhanced visualization
                    vis_mask = frame.copy()
                    overlay = np.zeros_like(frame)
                    overlay[hand_mask > 0] = [0, 255, 0]  # Green for skin
                    
                    # Draw contour outline
                    cv2.drawContours(overlay, [max_contour], -1, (0, 255, 255), 2)
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Blend visualization
                    alpha = 0.3
                    vis_mask = cv2.addWeighted(frame, 1, overlay, alpha, 0)
                    
                    return hand_region, hand_mask, (x,y,w,h), vis_mask
            
            return None, None, None, None
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            traceback.print_exc()
            return None, None, None, None

    def extract_features(self, frame):
        """Enhanced feature extraction with better features and normalization"""
        try:
            # Get preprocessed hand region
            hand_region, hand_mask, bbox, vis_mask = self.preprocess_frame(frame)
            
            # For negative samples or when no skin is detected, use the whole frame
            if hand_region is None:
                hand_region = cv2.resize(frame, self.target_size)
                
            # Convert to different color spaces for robust feature extraction
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
            
            # Enhance contrast for better feature detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            # 1. Multiple HOG features at different scales
            hog_features1 = hog(enhanced_gray, 
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            feature_vector=True)
            
            # Downsample for larger-scale features
            downsampled = cv2.resize(enhanced_gray, (self.target_size[0]//2, self.target_size[1]//2))
            hog_features2 = hog(downsampled,
                            orientations=6,
                            pixels_per_cell=(4, 4),
                            cells_per_block=(2, 2),
                            feature_vector=True)
            
            # 2. Improved color histograms - focus on hue and saturation for skin detection
            h, s, v = cv2.split(hsv)
            h_hist = cv2.calcHist([h], [0], None, [20], [0,180]).flatten()
            s_hist = cv2.calcHist([s], [0], None, [20], [0,256]).flatten()
            
            # 3. Gradient magnitude and direction features
            gx = cv2.Sobel(enhanced_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(enhanced_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            mag_hist = cv2.calcHist([cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)], 
                                [0], None, [20], [0,256]).flatten()
            ang_hist = cv2.calcHist([ang.astype(np.float32)], [0], None, [16], [0, np.pi*2]).flatten()
            
            # 4. Shape context - contour features if hand mask is available
            contour_features = []
            if hand_mask is not None:
                contours, _ = cv2.findContours(hand_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Compute basic shape features
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    shape_factor = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Add simple shape descriptors
                    contour_features = [area / (self.target_size[0] * self.target_size[1]), shape_factor]
                    
                    # Add convex hull features
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    contour_features.append(solidity)
            
            if not contour_features:  # If no contour features, add placeholders
                contour_features = [0, 0, 0]
            
            # Combine all features
            all_features = np.concatenate([
                hog_features1, 
                hog_features2,
                h_hist, 
                s_hist, 
                mag_hist,
                ang_hist,
                np.array(contour_features)
            ])
            
            # Normalize features
            all_features = (all_features - np.mean(all_features)) / (np.std(all_features) + 1e-7)
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            traceback.print_exc()
            return None
        
    def collect_training_data(self, num_samples=2000): 
        # Remove instruction frame as requested
        print("Starting data collection...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return None, None, None
        
        features_list = []
        landmarks_list = []
        no_hand_features = []
        count = 0
        no_hand_count = 0
        target_no_hand = num_samples // 5
        
        while count < num_samples or no_hand_count < target_no_hand:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Show skin detection visualization
                _, _, _, vis_mask = self.preprocess_frame(frame)
                if vis_mask is not None:
                    cv2.imshow('Skin Detection (Training)', vis_mask)
                else:
                    blank_mask = frame.copy()
                    cv2.putText(blank_mask, "No skin detected", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Skin Detection (Training)', blank_mask)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                features = self.extract_features(frame)
                if features is None:
                    continue
                
                # Add status indicators directly on main window
                cv2.putText(frame, f'Hand samples: {count}/{num_samples}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'No-hand samples: {no_hand_count}/{target_no_hand}', 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to stop collection", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if results.multi_hand_landmarks and count < num_samples:
                    # Get fingertip landmarks with improved normalization
                    landmarks = []
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Calculate hand center and size for normalization
                    all_x = [lm.x for lm in hand_landmarks.landmark]
                    all_y = [lm.y for lm in hand_landmarks.landmark]
                    hand_center_x = np.mean(all_x)
                    hand_center_y = np.mean(all_y)
                    
                    # Use wrist and middle finger MCP for better scale normalization
                    wrist = hand_landmarks.landmark[0]
                    middle_mcp = hand_landmarks.landmark[9]
                    hand_size = np.sqrt((middle_mcp.x - wrist.x)**2 + (middle_mcp.y - wrist.y)**2)
                    
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
                    
                elif not results.multi_hand_landmarks and count >= num_samples and no_hand_count < target_no_hand:
                    no_hand_features.append(features)
                    no_hand_count += 1
                
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
        print("Training improved models with validation...")
        
        # Combine positive and negative samples for hand detection
        X_hand_detection = np.vstack([features, no_hand_features])
        y_hand_detection = np.hstack([
            np.ones(len(features)),
            np.zeros(len(no_hand_features))
        ])
        
        # Split data with stratification
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_hand_detection, y_hand_detection, test_size=0.2, 
            random_state=42, stratify=y_hand_detection
        )
        
        # Use XGBoost for better performance
        try:
            import xgboost as xgb
            print("Using XGBoost for classification")
            self.hand_present_classifier = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='auc',
                early_stopping_rounds=20,
                verbosity=1
            )
        except ImportError:
            print("XGBoost not available, using GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier
            self.hand_present_classifier = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                verbose=1
            )
        
        print("Training hand detector...")
        # For XGBoost, provide validation set for early stopping
        if isinstance(self.hand_present_classifier, xgb.XGBClassifier):
            eval_set = [(X_val, y_val)]
            self.hand_present_classifier.fit(X_train, y_train, eval_set=eval_set)
        else:
            self.hand_present_classifier.fit(X_train, y_train)
        
        # Evaluate hand detector
        val_score = self.hand_present_classifier.score(X_val, y_val)
        print(f"Hand detector validation accuracy: {val_score:.3f}")
        
        # Try to use standardized features for landmark prediction
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(features)
        
        # Train landmark predictors with more robust models
        self.models = []
        
        # Split landmark data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_scaled, landmarks, test_size=0.2, random_state=42
        )
        
        for i in range(landmarks.shape[1]):
            print(f"Training landmark {i}...")
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    early_stopping_rounds=20,
                    verbosity=1
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42,
                    verbose=1,
                    n_iter_no_change=15
                )
            
            # For XGBoost, provide validation set for early stopping
            if isinstance(model, xgb.XGBRegressor):
                eval_set = [(X_val, y_val[:, i])]
                model.fit(X_train, y_train[:, i], eval_set=eval_set)
            else:
                model.fit(X_train, y_train[:, i])
            
            # Evaluate each landmark predictor
            val_score = model.score(X_val, y_val[:, i])
            print(f"Landmark {i} R² score: {val_score:.3f}")
            
            self.models.append(model)
        
        # Save the scaler for future use
        self.feature_scaler = scaler
        print("Training complete!")
        
    # Save the scaler too
    def save_models(self, filepath='hand_models.pkl'):
        with open(filepath, 'wb') as f:
            model_package = {
                'landmark_models': self.models,
                'hand_classifier': self.hand_present_classifier,
                'feature_scaler': self.feature_scaler if hasattr(self, 'feature_scaler') else None
            }
            pickle.dump(model_package, f)
        print(f"Models saved to {filepath}")
            
    def predict(self, frame):
        try:
            # Get the visualization mask
            _, _, _, vis_mask = self.preprocess_frame(frame)
            if vis_mask is not None:
                cv2.imshow('Skin Detection', vis_mask)
            else:
                blank_mask = frame.copy()
                cv2.putText(blank_mask, "No skin detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Skin Detection', blank_mask)

            # Extract features and scale if scaler is available
            features = self.extract_features(frame)
            if features is None:
                return frame, None
                
            # Apply scaling if available
            if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                features = self.feature_scaler.transform([features])[0]
                    
            # Check if hand is present
            if self.hand_present_classifier is not None:
                # Get probability of hand presence
                hand_confidence = self.hand_present_classifier.predict_proba([features])[0][1]
                
                # Use smoothing to avoid jitter
                if not hasattr(self, 'prev_confidence'):
                    self.prev_confidence = hand_confidence
                else:
                    # Exponential smoothing
                    alpha = 0.3  # Smoothing factor
                    hand_confidence = alpha * hand_confidence + (1 - alpha) * self.prev_confidence
                    self.prev_confidence = hand_confidence
                
                # Only predict landmarks if confident hand is present
                if hand_confidence > 0.7:  # Confidence threshold
                    landmarks = []
                    for model in self.models:
                        pred = model.predict([features])[0]
                        landmarks.append(pred)
                    
                    # Smoothing for landmark predictions
                    if not hasattr(self, 'prev_landmarks') or self.prev_landmarks is None:
                        self.prev_landmarks = landmarks
                    else:
                        # Apply smoothing to reduce jitter
                        alpha = 0.3  # Smoothing factor
                        for i in range(len(landmarks)):
                            landmarks[i] = alpha * landmarks[i] + (1 - alpha) * self.prev_landmarks[i]
                        self.prev_landmarks = landmarks
                    
                    # Draw predictions
                    height, width = frame.shape[:2]
                    
                    # Better estimation of hand size and center
                    hand_region, hand_mask, bbox, _ = self.preprocess_frame(frame)
                    
                    if bbox is not None:
                        # Use the actual bounding box for better hand center estimation
                        x, y, w, h = bbox
                        hand_center_x = x + w/2
                        hand_center_y = y + h/2
                        hand_size = max(w, h)
                    else:
                        # Fall back to frame-based estimation
                        hand_size = min(width, height) * 0.3
                        hand_center_x = width / 2
                        hand_center_y = height / 2
                    
                    # Draw index finger prediction
                    x = int((landmarks[0] * hand_size + hand_center_x))
                    y = int((landmarks[1] * hand_size + hand_center_y))
                    
                    # Draw both current position and prediction trajectory
                    if 0 <= x < width and 0 <= y < height:
                        # Draw point
                        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                        
                        # Draw trajectory/history if we have previous points
                        if hasattr(self, 'point_history'):
                            self.point_history.append((x, y))
                            # Keep only last 15 points
                            self.point_history = self.point_history[-15:]
                            
                            # Draw trajectory
                            for i in range(1, len(self.point_history)):
                                cv2.line(frame, 
                                    self.point_history[i-1], 
                                    self.point_history[i], 
                                    (0, 255, 255), 2)
                        else:
                            self.point_history = [(x, y)]
                        
                        cv2.putText(frame, "Index", (x + 10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        if hasattr(self, 'point_history'):
                            self.point_history = []
                    
                    # Draw hand confidence
                    cv2.putText(frame, f"Hand confidence: {hand_confidence:.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw bounding box if available
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
                    return frame, landmarks
                else:
                    cv2.putText(frame, f"No hand detected ({hand_confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Reset landmark history if hand is lost
                    if hasattr(self, 'point_history'):
                        self.point_history = []
                    if hasattr(self, 'prev_landmarks'):
                        self.prev_landmarks = None
                        
                    return frame, None
            else:
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