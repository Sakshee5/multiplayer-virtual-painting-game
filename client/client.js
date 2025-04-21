// DOM Elements
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const drawingCanvas = document.getElementsByClassName('drawing_canvas')[0];
const drawingCtx = drawingCanvas.getContext('2d', { willReadFrequently: true });
const countdownElement = document.getElementById('countdown');
const startAreaElement = document.getElementById('start-area');
const resetAreaElement = document.getElementById('reset-area');
const winnerAnnouncementElement = document.getElementById('winner-announcement');
const timerElement = document.getElementById('timer');
const playerScoresElement = document.getElementById('player-scores');
const gameInfoElement = document.querySelector('.game-info');
const powerUpsContainer = document.getElementById('power-ups-container');
const usernameModal = document.getElementById('username-modal');
const usernameInput = document.getElementById('username-input');
const submitUsernameButton = document.getElementById('submit-username');

// Apply responsive styling for smaller screens
document.addEventListener('DOMContentLoaded', function() {
  const videoContainer = document.querySelector('.video-container');
  const container = document.querySelector('.container');
  
  // Function to check screen size and apply scrolling if needed
  function checkScreenSize() {
    if (window.innerHeight < 768) { // Adjust this value based on your content height
      container.style.height = 'auto';
      container.style.overflowY = 'auto';
      document.body.style.overflowY = 'auto';
    } else {
      container.style.height = '';
      container.style.overflowY = '';
      document.body.style.overflowY = '';
    }
  }
  
  // Check on load and resize
  checkScreenSize();
  window.addEventListener('resize', checkScreenSize);
});

// Create drawing area border element
const drawingAreaBorder = document.createElement('div');
drawingAreaBorder.className = 'drawing-area-border';
document.querySelector('.video-container').appendChild(drawingAreaBorder);

// Audio elements
const beepSound = document.getElementById('beepSound');
const goSound = document.getElementById('goSound');
const gameplaySound = document.getElementById('gameplaySound');
const powerUpSound = document.getElementById('powerUpSound');
const winnerSound = document.getElementById('winnerSound');

// Set the canvas dimensions to match the video dimensions
canvasElement.width = 1280;
canvasElement.height = 720;
drawingCanvas.width = 1280;
drawingCanvas.height = 522;

// Game state variables
let gameActive = false;
let gameReset = false;
let gameCountdown = false;
let selfId = null;
let username = '';
let DRAW_COLOR = [255, 0, 0]; // Default color (will be assigned by server)
let DRAW_COLOR_TEMP = null; // Temporary color for power-ups

// Drawing variables
let xp = 0;
let yp = 0;
let brushThickness = 35;
let isDrawing = false;
let isErasing = false;

// Power-ups
let powerUpsAvailable = [];
let activePowerUpElements = [];

// Client data
let clientData = {};

// WebSocket connection
const getWebSocketUrl = () => {
    // If running from file:// protocol, use localhost
    if (window.location.protocol === 'file:') {
        return 'ws://localhost:5000/ws';
    } else {
        // For HTTPS, use wss:// protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.hostname}/ws`;
    }
};

let websocket = null;
let countdownActive = false;

// Position playerScoresElement
playerScoresElement.style.position = 'absolute';
playerScoresElement.style.bottom = '220px';
playerScoresElement.style.right = '10px';
playerScoresElement.style.backgroundColor = 'rgba(0,0,0,0.7)';
playerScoresElement.style.color = 'white';
playerScoresElement.style.padding = '10px';
playerScoresElement.style.borderRadius = '5px';
playerScoresElement.style.zIndex = '100';
playerScoresElement.style.maxHeight = '200px';
playerScoresElement.style.overflowY = 'auto';

// Helper function to robustly parse a color string
function parseColor(colorStr) {
  // If string starts with '[' then JSON parse
  if (colorStr.startsWith('[')) {
    try {
      return JSON.parse(colorStr);
    } catch (e) {
      return [0, 0, 0];
    }
  }
  // If string starts with '(' then remove parentheses and split
  if (colorStr.startsWith('(')) {
    let trimmed = colorStr.slice(1, -1);
    return trimmed.split(',').map(num => parseInt(num.trim()));
  }
  // Fallback: try JSON.parse
  try {
    return JSON.parse(colorStr);
  } catch (e) {
    return [0, 0, 0];
  }
}

// Handle username submission
submitUsernameButton.addEventListener('click', () => {
  username = usernameInput.value.trim();
  if (username) {
    usernameModal.style.display = 'none';
    connectWebSocket();
  } else {
    alert('Please enter a username');
  }
});

// Connect to WebSocket server
function connectWebSocket() {
  websocket = new WebSocket(getWebSocketUrl());

  websocket.onopen = () => {
    console.log("Connected to server");
    gameInfoElement.textContent = "Connected, waiting for color assignment...";

    // Send username to server
    if (username) {
      websocket.send(JSON.stringify({ type: "username", username: username }));
    }
  };

  websocket.onclose = (event) => {
    console.log("Disconnected from server:", event.code, event.reason);
    gameInfoElement.textContent = "Connection lost. Reconnecting...";
    setTimeout(connectWebSocket, 3000); // Try to reconnect after 3 seconds
  };

  websocket.onerror = (error) => {
    console.error("WebSocket error:", error);
    gameInfoElement.textContent = "Connection error. Reconnecting...";
  };

  websocket.onmessage = (event) => {
    try {
      // Try to parse as JSON first
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (e) {
        // If not JSON, it might be the initial color assignment as a string
        if (typeof event.data === 'string' && event.data.startsWith('[')) {
          DRAW_COLOR = JSON.parse(event.data);
          console.log("Assigned color:", DRAW_COLOR);
          gameInfoElement.textContent = `Your color: RGB(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
          return;
        }
      }

      // First message - color assignment
      if (Array.isArray(data) && data.length === 3) {
        DRAW_COLOR = data;
        console.log("Assigned color:", DRAW_COLOR);
        gameInfoElement.textContent = `Your color: RGB(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
        return;
      }

      // Client list update
      if (data.type === "client_list") {
        if (!gameActive && !gameCountdown && !gameReset) {
          const connectedClients = data.clients;
          selfId = data.self_id;
          clientData = {}; // Clear client data

          // Store client list for use in updateScores
          websocket.lastClientList = connectedClients;

          // Update clientData based on client list
          for (const [clientId, clientInfo] of Object.entries(connectedClients)) {
            if (clientInfo.color) {
              let colorKey;
              if (Array.isArray(clientInfo.color)) {
                colorKey = `${clientInfo.color[0]},${clientInfo.color[1]},${clientInfo.color[2]}`;
              } else if (typeof clientInfo.color === 'string' && clientInfo.color.startsWith('[')) {
                const colorArray = JSON.parse(clientInfo.color);
                colorKey = `${colorArray[0]},${colorArray[1]},${colorArray[2]}`;
              } else {
                colorKey = clientInfo.color.toString();
              }
              clientData[colorKey] = clientData[colorKey] || 0;
            }
          }

          clearDrawingCanvas();
          let i = 40;
          for (const [clientId, clientInfo] of Object.entries(connectedClients)) {
            const clientColor = Array.isArray(clientInfo.color) 
              ? clientInfo.color 
              : JSON.parse(clientInfo.color);

            const clientName = clientInfo.username || `Player ${clientId}`;

            if (parseInt(clientId) === selfId) {
              const text = `Connected - YOU (${clientName}): Color (${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
              drawText(drawingCtx, text, 650, 85, `rgb(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`, 20);
            } else {
              const text = `Connected - ${clientName}: Color (${clientColor[0]}, ${clientColor[1]}, ${clientColor[2]})`;
              drawText(drawingCtx, text, 650, 85 + i, `rgb(${clientColor[0]}, ${clientColor[1]}, ${clientColor[2]})`, 20);
              i += 40;
            }
          }
          updateScores();
        }
      }

      // Countdown
      if (data.type === "countdown") {
        if (!countdownActive) {
          countdownActive = true;
          gameActive = false;
          gameCountdown = true;
          const count = data.count;

          if (count === "GO") {
            goSound.play();
          } else {
            beepSound.play();
          }

          countdownElement.textContent = count;
          countdownElement.style.display = "block";

          // Hide countdown after a short delay
          if (count === "GO") {
            setTimeout(() => {
              countdownElement.style.display = "none";
              countdownActive = false;
            }, 1000);
          } else {
            setTimeout(() => {
              countdownActive = false;
            }, 900); // Slightly less than 1 second to prevent overlap
          }
        }
      }

      // Game start
      if (data.type === "start") {
        gameplaySound.play();
        clearDrawingCanvas();
        gameActive = true;
        gameReset = false;
        gameCountdown = false;
        countdownElement.style.display = "none";
        startAreaElement.style.display = "none";
        timerElement.style.display = "block";
        resetAreaElement.style.display = "none";
        winnerAnnouncementElement.style.display = "none";

        // Reset all power-up effects
        isErasing = false;
        DRAW_COLOR_TEMP = null;
        brushThickness = 35;

        // Clear all power-ups
        powerUpsAvailable = [];
        updatePowerUpDisplay();

        // Initialize scores to zero for all clients
        updateScores();
      }

      // Power-up spawn
      if (data.type === "power_up_spawn") {
        console.log("Power-up spawned:", data.power_up);  // Debug log
        powerUpsAvailable.push(data.power_up);
        updatePowerUpDisplay();
      }

      // Game reset
      if (data.type === "reset") {
        gameActive = false;
        gameReset = false;
        gameCountdown = false;
        countdownActive = false;
        clearDrawingCanvas();
        resetAreaElement.style.display = "none";
        startAreaElement.style.display = "flex";
        timerElement.style.display = "none";
        winnerAnnouncementElement.style.display = "none";

        // Reset all power-up effects
        isErasing = false;
        DRAW_COLOR_TEMP = null;
        brushThickness = 35;

        // Clear power-ups
        powerUpsAvailable = [];
        updatePowerUpDisplay();

        // Clear client data
        clientData = {};
        updateScores();

        // Stop music
        gameplaySound.pause();
        gameplaySound.currentTime = 0;
      }

      // Winner announcement
      if (data.type === "winner") {
        gameActive = false;
        gameReset = true;

        // Play winner sound
        winnerSound.play();

        // Pause gameplay sound
        gameplaySound.pause();
        gameplaySound.currentTime = 0;

        const winner = data.winner;
        console.log("Winner is", winner);

        let winnerColor, winnerName;
        if (typeof winner === 'object') {
          if (Array.isArray(winner.color)) {
            winnerColor = winner.color;
          } else if (typeof winner.color === 'string') {
            // Use helper function to parse color whether in [r,g,b] or (r, g, b) format
            winnerColor = winner.color.startsWith('[') ? JSON.parse(winner.color) : parseColor(winner.color);
          }
          winnerName = winner.username || "Player";
        } else if (typeof winner === 'string') {
          winnerColor = winner.startsWith('[') ? JSON.parse(winner) : parseColor(winner);
          winnerName = "Player";
        }

        // Display winner announcement
        winnerAnnouncementElement.textContent = `Winner: ${winnerName} (${winnerColor[0]}, ${winnerColor[1]}, ${winnerColor[2]})`;
        winnerAnnouncementElement.style.color = `rgb(${winnerColor[0]}, ${winnerColor[1]}, ${winnerColor[2]})`;
        winnerAnnouncementElement.style.display = "block";

        // Show reset area
        resetAreaElement.style.display = "flex";

        // Clear power-ups
        powerUpsAvailable = [];
        updatePowerUpDisplay();
      }

      // Timer update
      if (data.type === "timer") {
        const timeLeft = data.time_left;
        timerElement.textContent = `Time Left: ${timeLeft}s`;

        // Update scores regularly
        if (gameActive) {
          updateScores();
        }

        // Game has ended when timer reaches 0
        if (timeLeft === 0) {
          gameActive = false;
        }
      }

      // Drawing data received
      if (data.hasOwnProperty('x1')) {
        const { x1, y1, x2, y2, color, brush_thickness, pixel_perc, power_up_id, client_id, is_eraser } = data;

        // Handle power-up collection
        if (power_up_id) {
          for (let i = 0; i < powerUpsAvailable.length; i++) {
            if (powerUpsAvailable[i].id === power_up_id) {
              // Play power-up sound
              if (client_id === selfId) {
                powerUpSound.play();
              }

              const powerUpType = powerUpsAvailable[i].type;
              powerUpsAvailable.splice(i, 1);
              updatePowerUpDisplay();

              if (client_id === selfId) {
                if (powerUpType === "paint_bucket") {
                  applyPaintBucket();
                }

                if (powerUpType === "paint_brush") {
                  applyPaintBrush();
                }

                if (powerUpType === "eraser") {
                  applyEraser();
                }

                if (powerUpType === "devil_face") {
                  applyDevilFace();
                }

                // Add surprise power-up handling
                if (powerUpType === "surprise") {
                  // Randomly select one of the other power-ups
                  const powerUps = ["paint_bucket", "paint_brush", "eraser", "devil_face"];
                  const randomPowerUp = powerUps[Math.floor(Math.random() * powerUps.length)];
                  
                  // Apply the randomly selected power-up
                  if (randomPowerUp === "paint_bucket") {
                    applyPaintBucket();
                  } else if (randomPowerUp === "paint_brush") {
                    applyPaintBrush();
                  } else if (randomPowerUp === "eraser") {
                    applyEraser();
                  } else if (randomPowerUp === "devil_face") {
                    applyDevilFace();
                  }
                }
              }

              break;
            }
          }
        }

        // Draw on canvas if it's not our own drawing
        if (client_id !== selfId && gameActive) {
          if (is_eraser) {
            // Handle eraser drawing
            drawingCtx.globalCompositeOperation = 'destination-out';
            drawingCtx.beginPath();
            drawingCtx.moveTo(x1, y1);
            drawingCtx.lineTo(x2, y2);
            drawingCtx.strokeStyle = 'rgba(255,255,255,1)';
            drawingCtx.lineWidth = brush_thickness;
            drawingCtx.lineCap = 'round';
            drawingCtx.stroke();
            drawingCtx.globalCompositeOperation = 'source-over';
          } else {
            // Handle normal drawing
            drawingCtx.beginPath();
            drawingCtx.moveTo(x1, y1);
            drawingCtx.lineTo(x2, y2);
            drawingCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            drawingCtx.lineWidth = brush_thickness;
            drawingCtx.lineCap = 'round';
            drawingCtx.stroke();
          }
        }

        // Update client data for score display
        if (pixel_perc !== undefined) {
          const colorKey = Array.isArray(color) ? `${color[0]},${color[1]},${color[2]}` : color;
          clientData[colorKey] = pixel_perc;
          updateScores();
        }
      }

      // Handle pixel updates from other players
      if (data.type === "pixel_update") {
        const color = data.color;
        const pixel_perc = data.pixel_perc;
        if (color && pixel_perc !== undefined) {
          const colorKey = Array.isArray(color) ? `${color[0]},${color[1]},${color[2]}` : color;
          clientData[colorKey] = pixel_perc;
          updateScores();
        }
      }

      // Handle paint bucket action from other clients
      if (data.type === "paint_bucket" && data.client_id !== selfId) {
        drawPaintBucket(data.x, data.y, data.color);
      }

    } catch (error) {
      console.error("Error processing message:", error, event.data);
    }
  };
}

// Function to update power-up display
function updatePowerUpDisplay() {
  // Clear existing power-up elements
  powerUpsContainer.innerHTML = '';

  // Create elements for each power-up
  powerUpsAvailable.forEach(powerUp => {
    const powerUpElement = document.createElement('div');
    powerUpElement.className = 'power-up';
    powerUpElement.style.position = 'absolute';
    powerUpElement.style.left = `${powerUp.x}px`;
    powerUpElement.style.top = `${powerUp.y}px`;

    // Create an image element
    const img = document.createElement('img');
    // Use URL constructor to properly resolve the path
    img.src = new URL(powerUp.image, window.location.origin).href;
    img.style.width = '30px';  // Adjust size as needed
    img.style.height = '30px';

    // Append the image to the power-up element
    powerUpElement.appendChild(img);
    powerUpsContainer.appendChild(powerUpElement);
  });
}

// Function to clear the drawing canvas
function clearDrawingCanvas() {
  drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
}

// Draw text helper
function drawText(context, text, x, y, color, fontSize) {
  context.font = `${fontSize}px Arial`;
  context.fillStyle = color;
  context.fillText(text, x, y);
}

// Update scores display
function updateScores() {
  if (!clientData || Object.keys(clientData).length === 0) {
    playerScoresElement.innerHTML = "<strong>Player Scores:</strong><br>No scores yet";
    return;
  }

  let scoresHTML = "<strong>Player Scores:</strong><br>";

  // Sort client data by percentage (descending)
  const sortedEntries = Object.entries(clientData).sort((a, b) => b[1] - a[1]);

  // Find player names from the most recent client list
  for (const [colorStr, percentage] of sortedEntries) {
    let color;
    if (colorStr.includes(',')) {
      color = colorStr.split(",").map(Number);
    } else {
      continue; // Skip invalid entries
    }

    const colorCSS = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    let playerName = "";
    
    // Find player name matching this color
    if (color[0] === DRAW_COLOR[0] && color[1] === DRAW_COLOR[1] && color[2] === DRAW_COLOR[2]) {
      playerName = "YOU";
    } else {
      // Try to find the name in the client list
      for (const [clientId, clientInfo] of Object.entries(websocket.lastClientList || {})) {
        const clientColor = Array.isArray(clientInfo.color) ? clientInfo.color : 
                          (typeof clientInfo.color === 'string' && clientInfo.color.startsWith('[')) ? 
                          JSON.parse(clientInfo.color) : 
                          clientInfo.color;
        
        if (clientColor[0] === color[0] && clientColor[1] === color[1] && clientColor[2] === color[2]) {
          playerName = clientInfo.username || `Player ${clientId}`;
          break;
        }
      }
      
      // If we couldn't find the name, fall back to showing the color
      if (!playerName) {
        playerName = `(${color[0]}, ${color[1]}, ${color[2]})`;
      }
    }

    if (color[0] === 0 && color[1] === 0 && color[2] === 0) {
      // Skip black color
      continue;
    }

    scoresHTML += `<div style="color: ${colorCSS}; ${color[0] === DRAW_COLOR[0] && color[1] === DRAW_COLOR[1] && color[2] === DRAW_COLOR[2] ? 'font-weight: bold;' : ''} margin-bottom: 5px;">
                    ${playerName}: ${percentage.toFixed(2)}%
                  </div>`;
  }

  playerScoresElement.innerHTML = scoresHTML;
  playerScoresElement.style.display = 'block';
}

// Update and send pixel percentage
async function updateAndSendPixelPercentage() {
  if (!gameActive || !websocket || websocket.readyState !== WebSocket.OPEN) return;

  const percentage = await getPixelPercent(DRAW_COLOR);

  websocket.send(JSON.stringify({
    type: "pixel_update",
    color: DRAW_COLOR,
    pixel_perc: percentage,
    client_id: selfId
  }));

  // Update local data too
  const colorKey = `${DRAW_COLOR[0]},${DRAW_COLOR[1]},${DRAW_COLOR[2]}`;
  clientData[colorKey] = percentage;
  updateScores();
}

// Function to check if index finger is up and others are down
function isIndexFingerUp(landmarks) {
  // Get the y-coordinates of important finger landmarks
  const wristY = landmarks[0].y;
  const indexMCP = landmarks[5].y;
  const indexPIP = landmarks[6].y;
  const indexDIP = landmarks[7].y;
  const indexTip = landmarks[8].y;

  const middleMCP = landmarks[9].y;
  const middleTip = landmarks[12].y;

  const ringMCP = landmarks[13].y;
  const ringTip = landmarks[16].y;

  const pinkyMCP = landmarks[17].y;
  const pinkyTip = landmarks[20].y;

  const thumbTip = landmarks[4].y;

  // Check if index finger is extended up
  const isIndexUp = indexTip < indexDIP && indexDIP < indexPIP && indexPIP < indexMCP;

  // Check if other fingers are closed (tips below MCPs)
  const isMiddleClosed = middleTip > middleMCP;
  const isRingClosed = ringTip > ringMCP;
  const isPinkyClosed = pinkyTip > pinkyMCP;

  // For thumb we just check if it's generally higher than the wrist
  const isThumbNotUp = thumbTip > wristY;

  return isIndexUp && isMiddleClosed && isRingClosed && isPinkyClosed;
}

// Function to check if index and middle fingers are up (for selection)
function isTwoFingersUp(landmarks) {
  const indexMCP = landmarks[5].y;
  const indexPIP = landmarks[6].y;
  const indexTIP = landmarks[8].y;

  const middleMCP = landmarks[9].y;
  const middlePIP = landmarks[10].y;
  const middleTIP = landmarks[12].y;

  const ringTIP = landmarks[16].y;
  const pinkyTIP = landmarks[20].y;

  // Check if index and middle fingers are up
  const isIndexUp = indexTIP < indexPIP && indexPIP < indexMCP;
  const isMiddleUp = middleTIP < middlePIP && middlePIP < middleMCP;

  // Check if other fingers are down
  const isRingDown = ringTIP > landmarks[13].y;
  const isPinkyDown = pinkyTIP > landmarks[17].y;

  return isIndexUp && isMiddleUp && isRingDown && isPinkyDown;
}

// Check if selection is in a specific area
function isInArea(x, y, areaX, areaY, areaWidth, areaHeight) {
  return x >= areaX && x <= areaX + areaWidth && y >= areaY && y <= areaY + areaHeight;
}

// Function to scale coordinates from camera space to drawing space
function scaleCoordinates(x, y) {
  // Flip x coordinate
  const scaledX = canvasElement.width - x * canvasElement.width;

  const absoluteY = y * canvasElement.height;  // Get position in 720px space

  return { x: scaledX, y: absoluteY };
}

// Check for power-up collection
function checkPowerUpCollection(x, y) {
  for (let i = 0; i < powerUpsAvailable.length; i++) {
    const powerUp = powerUpsAvailable[i];
    // Power-up coordinates are already in drawing canvas space
    if (Math.abs(x - powerUp.x) < 20 && Math.abs(y - powerUp.y) < 20) {
      return powerUp.id;
    }
  }
  return null;
}

// Calculate pixel percentage covered by a specific color
async function getPixelPercent(color) {
  const imageData = drawingCtx.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);
  const data = imageData.data;
  let matchingPixels = 0;
  const totalPixels = drawingCanvas.width * drawingCanvas.height;

  for (let i = 0; i < data.length; i += 4) {
    if (data[i] === color[0] && data[i + 1] === color[1] && data[i + 2] === color[2] && data[i + 3] !== 0) {
      matchingPixels++;
    }
  }

  return (matchingPixels / totalPixels) * 100;
}

// Send drawing data to server
async function sendDrawData(x1, y1, x2, y2, powerUpId) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN || !gameActive) return;

  // Use temporary color if a power-up effect is active
  const colorToSend = DRAW_COLOR_TEMP || DRAW_COLOR;

  const data = {
    x1: x1,
    y1: y1,
    x2: x2,
    y2: y2,
    color: colorToSend,
    brush_thickness: brushThickness,
    client_id: selfId,
    is_eraser: isErasing  // Add eraser flag
  };

  // Add power-up ID if collecting one
  if (powerUpId) {
    data.power_up_id = powerUpId;
  }

  websocket.send(JSON.stringify(data));

  // Update percentage if it's been more than 500ms since last update
  if (!data.lastUpdate || Date.now() - data.lastUpdate > 500) {
    data.lastUpdate = Date.now();
    updateAndSendPixelPercentage();
  }
}

// Apply paint bucket power-up effect
function applyPaintBucket() {
  const x = Math.floor(Math.random() * 1160) + 50;
  const y = Math.floor(Math.random() * 400) + 50;

  // Send paint bucket action to server
  websocket.send(JSON.stringify({
    type: "paint_bucket",
    x: x,
    y: y,
    color: DRAW_COLOR,
    client_id: selfId
  }));

  // Draw locally
  drawPaintBucket(x, y, DRAW_COLOR);
}

// Function to draw paint bucket effect
function drawPaintBucket(x, y, color) {
  drawingCtx.beginPath();
  drawingCtx.arc(x, y, 100, 0, 2 * Math.PI);
  drawingCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  drawingCtx.fill();
}

// Apply paint brush power-up effect
function applyPaintBrush() {
  brushThickness = 70; // Larger brush

  // Reset to normal brush after 5 seconds
  setTimeout(() => {
    brushThickness = 35;
  }, 5000);
}

// Apply eraser power-up effect
function applyEraser() {
  // Enable erasing mode
  isErasing = true;
  brushThickness = 50; // Larger brush for erasing

  // Reset to normal brush after 5 seconds
  setTimeout(() => {
    isErasing = false;
    brushThickness = 35;
  }, 5000);
}

// Apply devil face power-up effect (broken brush)
function applyDevilFace() {
  brushThickness = 1; // Broken brush

  // Reset to normal brush after 5 seconds
  setTimeout(() => {
    brushThickness = 35;
  }, 5000);
}

// MediaPipe Hands setup
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Process hand landmarks for drawing
hands.onResults((results) => {
  // Draw camera feed
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Draw the camera feed with mirrored view consistently
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  // Process hands
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Convert to flipped coordinate system for drawing
    canvasCtx.lineWidth = 0.5;
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: 'rgba(255, 255, 255, 0.4)', lineWidth: 0.5});
    drawLandmarks(canvasCtx, landmarks, {color: 'rgba(255, 255, 255, 0.5)', lineWidth: 0.5, radius: 2});

    // Get index finger position and scale coordinates
    const indexFinger = landmarks[8];
    const { x, y } = scaleCoordinates(indexFinger.x, indexFinger.y);

    // Only process drawing if y is within the drawing canvas bounds
    if (y <= 522) {
      // Handle two fingers up (selection mode)
      if (isTwoFingersUp(landmarks)) {
        // Check if in start area (coordinates are in drawing canvas space)
        if (!gameActive && !gameCountdown && isInArea(x, y, 340, 300, 200, 100)) {
          startAreaElement.style.backgroundColor = "rgba(0,255,0,0.3)";
          // On click effect - send start command
          if (!gameReset) {
            websocket.send(JSON.stringify({ type: "start" }));
          }
        } else {
          startAreaElement.style.backgroundColor = "transparent";
        }

        // Check if in reset area (coordinates are in drawing canvas space)
        if (gameReset && isInArea(x, y, 740, 300, 200, 100)) {
          resetAreaElement.style.backgroundColor = "rgba(255,0,0,0.3)";
          // On click effect - send reset command
          websocket.send(JSON.stringify({ type: "reset" }));
        } else {
          resetAreaElement.style.backgroundColor = "transparent";
        }
      }

      // Handle index finger for drawing
      if (isIndexFingerUp(landmarks) && gameActive) {
        if (!isDrawing) {
          // Start drawing
          isDrawing = true;
          xp = x;
          yp = y;
        } else {
          // Check for power-up collection
          const powerUpId = checkPowerUpCollection(x, y);

          // Continue drawing
          if (isErasing) {
            // Eraser functionality - use destination-out compositing operation
            drawingCtx.globalCompositeOperation = 'destination-out';
            drawingCtx.beginPath();
            drawingCtx.moveTo(xp, yp);
            drawingCtx.lineTo(x, y);
            drawingCtx.strokeStyle = 'rgba(255,255,255,1)';
            drawingCtx.lineWidth = brushThickness;
            drawingCtx.lineCap = 'round';
            drawingCtx.stroke();
            drawingCtx.globalCompositeOperation = 'source-over';
          } else {
            // Normal drawing
            drawingCtx.beginPath();
            drawingCtx.moveTo(xp, yp);
            drawingCtx.lineTo(x, y);
            drawingCtx.strokeStyle = `rgb(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
            drawingCtx.lineWidth = brushThickness;
            drawingCtx.lineCap = 'round';
            drawingCtx.stroke();
          }

          // Send drawing data to server
          sendDrawData(xp, yp, x, y, powerUpId);

          // Update position
          xp = x;
          yp = y;
        }
      } else {
        isDrawing = false;
      }
    } else {
      isDrawing = false;
    }
  } else {
    isDrawing = false;
  }

  canvasCtx.restore();
});

// Initialize camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720  // Back to 720 for better hand detection
});

// Add camera initialization with error handling
async function initializeCamera() {
  try {
    // Check if we're in a secure context
    if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
      throw new Error('Camera access requires HTTPS or localhost');
    }

    // Check if mediaDevices API is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not available in this browser');
    }

    // Request camera permission
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    
    // Start camera processing
    await camera.start();
  } catch (error) {
    console.error('Camera initialization error:', error);
    // Display error message to user
    const errorMessage = document.createElement('div');
    errorMessage.style.color = 'red';
    errorMessage.style.padding = '20px';
    errorMessage.style.textAlign = 'center';
    errorMessage.innerHTML = `
      <h2>Camera Access Error</h2>
      <p>${error.message}</p>
      <p>Please ensure:</p>
      <ul style="text-align: left; display: inline-block;">
        <li>You're using HTTPS or localhost</li>
        <li>You've granted camera permissions</li>
        <li>Your browser supports camera access</li>
      </ul>
    `;
    document.querySelector('.container').prepend(errorMessage);
  }
}

// Call initializeCamera instead of camera.start()
initializeCamera();

// Event listeners
resetAreaElement.addEventListener('click', () => {
  if (gameReset) {
    websocket.send(JSON.stringify({ type: "reset" }));
  }
});

startAreaElement.addEventListener('click', () => {
  if (!gameActive && !gameCountdown && !gameReset) {
    websocket.send(JSON.stringify({ type: "start" }));
  }
});

// Handle window resize
window.addEventListener('resize', () => {
  const container = document.querySelector('.container');
  const videoContainer = document.querySelector('.video-container');

  // Keep aspect ratio consistent
  const maxWidth = Math.min(window.innerWidth, 1280);
  container.style.width = `${maxWidth}px`;
  videoContainer.style.height = `${(maxWidth / 1280) * 720}px`;
});

// Handle page visibility change (pause/resume game sounds)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    gameplaySound.pause();
  } else if (gameActive) {
    gameplaySound.play();
  }
});