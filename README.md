# Multiplayer Drawing Game - Splashh

Splashh is **Multiplayer Drawing Game** with real-time interactions powered by WebSockets, OpenCV, and Python. Players can join a game, use hand gestures to draw on a shared canvas, and compete to fill the most canvas area while utilizing various power-ups.

Gameplay Snippet
![Gameplay Snippet](video_snippet.gif)

[Complete Video Walkthrough](https://youtu.be/SlA19znMufY?si=TuDvYnA9aIcu4sZw)


## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Code Overview](#code-overview)
- [Power-Ups](#power-ups)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)


## Features
- **Real-Time Multiplayer Drawing**: Players can draw on a shared canvas via a webcam-based hand-tracking system.
- **WebSocket Communication**: Ensures smooth, real-time updates between the server and clients.
- **Power-Ups**: Players can collect and use power-ups like erasers, paint buckets, and more to influence gameplay.
- **Countdown Timer and Score Display**: Tracks the game duration and dynamically displays scores.
- **Gesture-Based Controls**: Use hand gestures to start/reset the game.


## How to Run (for local use)

### Server
1. Start the WebSocket server:
```python server.py```

### Client
2. Run the client application:
```python client.py```

Note: For the hackathon, I am deploying the server on ngrok for temporary basis to showcase multiplayer interactivity.

## How It Works

### General Flow
- The **server** (`server.py`) manages client connections and broadcasts updates about the shared canvas, power-ups, and game events.
- The **client** (`client.py`) facilitates interaction with the server using the webcam for hand tracking, displaying the drawn canvas, and distributing updates.

### Game Modes
- **Start Gestures**: Keep index and middle finger upright and move your hand over the "START" button to begin.
- **Reset Gestures**: Keep index and middle finger upright and move your hand over the "RESET" button to reset the game.

### Real-Time Drawing
- The index finger is used for drawing on the canvas.
- Draw paths are broadcast to all connected players via the WebSocket server.


## Code Overview

### `client.py`
Handles the following:
- Webcam-based **hand tracking** using OpenCV and a custom `hand_tracker` module.
- Real-time communication with the server via WebSockets (`websockets` library).
- Drawing implementation on a live feed overlayed with the shared canvas.
- **Power-Up Effects**: Enables temporary effects like changing brush size, using an eraser, etc.

### `server.py`
Manages:
- WebSocket connections and assigns unique colors to players.
- Shared canvas data and broadcasts drawing updates to all players.
- Power-up spawning logic and game events like start/reset/winner announcements.
- Game timer and score tracking.


## Power-Ups
### List of Power-Ups
1. <img src="eraser.png" alt="Eraser Power-up" width="20">  **Eraser**: Temporarily allows the player to erase opponent drawings.

2.  <img src="devil_face.png" alt="Eraser Power-up" width="20"> **Devil Face**: Reduces the opponent's brush size for a limited time.
3.  <img src="paint_bucket.png" alt="Eraser Power-up" width="20"> **Paint Bucket**: Fills a random area on the canvas with the player's color.
4.  <img src="paint_brush.png" alt="Eraser Power-up" width="20"> **Paint Brush**: Temporarily increases the player's brush size.

### Power-Up Spawning
- Power-ups are spawned at random intervals and locations on the canvas.
- Players collect them by interacting with their virtual hand near the power-up's position.


## Future Enhancements
- Improve hand-tracking accuracy for complex gestures.
- Add more power-ups and effects.
- Create a UI for lobby management (e.g., joining/leaving games).
- Implement scoring systems for better gamification.
- Deploy the WebSocket server to cloud-based platforms for wider accessibility.