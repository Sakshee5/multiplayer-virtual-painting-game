# Hand-Paint Multiplayer Game - Splashh

## Description

Splashh is a real-time multiplayer game that allows users to draw collaboratively on a virtual canvas using only their hand gestures. The game utilizes computer vision techniques (mediapipe) for hand tracking and gesture recognition, enabling an interactive and intuitive drawing experience. Players can connect through WebSockets and interact seamlessly in a shared environment.

## Features

1. Real-time hand tracking using OpenCV and MediaPipe
2. Gesture recognition to control drawing actions
3. Multiplayer support with WebSocket-based communication
4. Shared virtual canvas for collaborative gameplay
5. Power-ups and scoring system based on painted area
6. Real-time calculation of area covered on the virtual canvas for all players.
7. Other fun elements like background audio and artwork.
8. Webcam-based interaction requiring no additional hardware
9. Countdown timer (3,2,1) before enabling drawing
10. Reset and play endlessly.

## Tech Stack

1. Computer Vision: OpenCV, MediaPipe
2. Networking: WebSockets
3. Backend: Python
4. Temporary Server Deployment: ngrok

## Future Improvements

1. Enhance gesture recognition for additional interactions
2. Implement a lobby system for better player management. 
3. Improve real-time responsiveness and UI/UX
4. Robust deployment of client and server to handle multiple players.