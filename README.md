# Multiplayer Drawing Game - Splashh

Splashh is a **Multiplayer Drawing Game** with real-time interactions powered by WebSockets, MediaPipe Hands, and JavaScript. Players can join a game, use hand gestures to draw on a shared canvas, and compete to fill the most canvas area while utilizing various power-ups.

Gameplay Snippet
![Gameplay Snippet](client/assets/video_snippet.gif)

[Complete Video Walkthrough](https://youtu.be/vPPS1YDccx0)

## Features
- **Real-Time Multiplayer Drawing**: Players can draw on a shared canvas using MediaPipe Hands for gesture tracking
- **WebSocket Communication**: Ensures smooth, real-time updates between the server and clients
- **Power-Ups**: Players can collect and use power-ups like erasers, paint buckets, and more to influence gameplay
- **Countdown Timer and Score Display**: Tracks the game duration and dynamically displays scores
- **Gesture-Based Controls**: Use hand gestures to start/reset the game
- **Username System**: Players can set custom usernames for better identification
- **Responsive Design**: Adapts to different screen sizes while maintaining aspect ratio
- **Sound Effects**: Audio feedback for game events, power-ups, and winner announcements
- **Canvas Synchronization**: Real-time canvas updates across all connected players
- **Score Tracking**: Percentage-based scoring system for fair competition

## Tech Stack
- **Frontend**:
  - HTML5 Canvas for drawing
  - MediaPipe Hands for hand gesture tracking
  - WebSocket client for real-time communication
  - Modern CSS3 for UI/UX
  - JavaScript for game logic and interactions

- **Backend**:
  - Python WebSocket server (aiohttp)
  - NumPy for canvas data management
  - Asyncio for asynchronous operations
  - Docker for containerization

## Installation & Setup

### Local Development
1. Clone the repository:
```bash
git clone https://github.com/Sakshee5/multiplayer-virtual-painting-game.git
cd multiplayer-virtual-painting-game
```

2. Install dependencies:
```bash
# Server dependencies
cd server
pip install -r requirements.txt

# Client dependencies (none required, uses CDN for libraries)
```

3. Start the server:
```bash
python server.py
```

4. Open `client/index.html` in a web browser with webcam access.

### Docker Deployment
Access the game at `https://vcm-47044.vm.duke.edu/`

## How It Works

### General Flow
- The **server** (`server.py`) manages:
  - Client connections and disconnections
  - Game state and synchronization
  - Power-up spawning and management
  - Score tracking and winner determination
  - Real-time canvas updates

- The **client** (`client.js`) handles:
  - Hand tracking using MediaPipe Hands
  - Canvas drawing and updates
  - WebSocket communication
  - Power-up effects and game state management
  - User interface and interactions

### Game Controls
- **Start Game**: Keep index and middle fingers upright and move over the "START" button
- **Reset Game**: Keep index and middle fingers upright and move over the "RESET" button
- **Drawing**: Use index finger to draw on the canvas
- **Power-up Collection**: Touch power-ups with index finger to collect them

### Real-Time Drawing
- Index finger tracking for precise drawing
- Real-time path broadcasting to all players
- Synchronized canvas updates across clients
- Brush thickness and color management

## Power-Ups and Downs
### Available Power-Ups and Downs 
1. <img src="server/assets/eraser.png" alt="Eraser Power-up" width="20"> **Eraser**: Temporarily (5 secs) allows erasing opponent drawings
2. <img src="server/assets/devil_face.png" alt="Devil Face Power-down" width="20"> **Devil Face**: Reduces opponent's brush size for 5 seconds
3. <img src="server/assets/paint_bucket.png" alt="Paint Bucket Power-up" width="20"> **Paint Bucket**: Fills a random area with player's color
4. <img src="server/assets/paint_brush.png" alt="Paint Brush Power-up" width="20"> **Paint Brush**: Temporarily increases brush size
4. <img src="server/assets/surprise.png" alt="Paint Brush Power-up" width="20"> **Surprise**: Can turn out to be any of the above 4

### Power-Up Mechanics
- Random spawning every 10-15 seconds
- Unique IDs for each power-up
- Visual indicators on the canvas
- Sound effects on collection
- Temporary effects (5-second duration)

## Ethics Statement

This project adheres to the following ethical principles:

1. **Privacy & Data Protection**:
   - No personal data is stored or transmitted beyond usernames
   - Webcam data is processed locally and never stored
   - All communications are encrypted using HTTPS/WSS

2. **Accessibility**:
   - Designed to work with standard web browsers
   - Clear visual feedback for all actions
   - Audio cues for important events

3. **Fair Play**:
   - Equal starting conditions for all players
   - Transparent scoring system
   - Random power-up distribution

## Hand Detection Approaches

### Traditional Approach (traditional_hand_detection.py)
The traditional approach implements a hand detection system using a combination of computer vision techniques and machine learning.

1. **Preprocessing Pipeline**:
   - Converts frames to multiple color spaces (HSV, YCrCb) for skin detection
   - Uses refined HSV thresholds for skin segmentation (The values have been set as per the repo owners skin color)
   - Applies morphological operations to clean up detection
   - Implements contour analysis with area and aspect ratio filtering
   - Adds padding and bounds checking for better hand region extraction

2. **Feature Extraction**:
   - Uses Histogram of Oriented Gradients (HOG) at multiple scales
   - Implements color histograms focusing on hue and saturation
   - Calculates gradient magnitude and direction features
   - Extracts shape context features from hand contours
   - Normalizes features for consistent model input

3. **Machine Learning Components**:
   - Uses XGBoost/GradientBoosting for both classification and regression
   - Implements a two-stage detection system:
     - Hand presence classifier
     - Landmark position regressors
   - Includes validation sets for model training
   - Applies feature scaling and standardization

4. **Prediction Pipeline**:
   - Real-time hand detection with confidence scoring
   - Smoothing algorithms to reduce jitter
   - Trajectory tracking for stable predictions
   - Visualization of detection results
   - Error handling and fallback mechanisms

### Naive Approach (naive_hand_detection.py)
The naive approach uses simpler computer vision techniques for hand detection:

1. **Color-Based Detection**:
   - Uses HSV color space for skin detection
   - Defines fixed ranges for skin color (lower_skin, upper_skin)
   - Applies basic morphological operations for noise reduction
   - Uses Gaussian blur for smoothing

2. **Contour Analysis**:
   - Finds the largest contour in the skin mask
   - Implements area thresholding to filter noise
   - Uses convex hull and convexity defects for finger detection
   - Calculates angles between fingers for counting

3. **Visualization**:
   - Draws hand contours and convex hull
   - Marks finger positions with circles
   - Displays finger count on screen
   - Shows binary mask for debugging

4. **Limitations**:
   - Less accurate in varying lighting conditions
   - May struggle with complex hand poses
   - More susceptible to false positives
   - Limited to basic hand detection
   - No confidence scoring


## Future Enhancements
- Improve hand-tracking accuracy for complex gestures
- Add more power-ups and effects
- Create a UI for lobby management
- Deploy to cloud platforms for wider accessibility
- Suite of Games Based on Virtual Drawing (e.g., Pictionary)
- Platform integrations (Zoom, Microsoft Teams)
- Enhanced accessibility features
- Mobile device support