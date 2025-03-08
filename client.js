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

// Audio elements
const beepSound = document.getElementById('beepSound');
const goSound = document.getElementById('goSound');
const gameplaySound = document.getElementById('gameplaySound');
const powerUpSound = document.getElementById('powerUpSound');
const winnerSound = document.getElementById('winnerSound');

// Set the canvas dimensions to match the video dimensions
canvasElement.width = 1280;
canvasElement.height = 522;
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
let brushThickness = 25;
let isDrawing = false;

// Power-ups
let powerUpsAvailable = [];
let activePowerUpElements = [];

// Client data
let clientData = {};

// WebSocket connection
const SERVER = "ws://localhost:8765";
let websocket = null;
let countdownActive = false;

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
  websocket = new WebSocket(SERVER);
  
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
        
        // Clear all power-ups
        powerUpsAvailable = [];
        updatePowerUpDisplay();
      }
      
      // Power-up spawn
      if (data.type === "power_up_spawn") {
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
        
        // Clear power-ups
        powerUpsAvailable = [];
        updatePowerUpDisplay();
        
        // Reset client data
        clientData = {};
        updateScores();
        
        // Stop music
        gameplaySound.pause();
        gameplaySound.currentTime = 0;
      }
      
      // Winner announcement
      if (data.type === "winner") {
        winnerSound.play();
        gameplaySound.pause();
        gameplaySound.currentTime = 0;
        gameActive = false;
        gameReset = true;
        const winner = data.winner;
        console.log("Winner is", winner);
        
        let winnerColor, winnerName;
        
        if (typeof winner === 'string' && winner.startsWith('[')) {
          winnerColor = JSON.parse(winner);
          winnerName = "Player";
        } else if (typeof winner === 'object') {
          winnerColor = winner.color;
          winnerName = winner.username || "Player";
        }
        
        winnerAnnouncementElement.textContent = `Winner: ${winnerName} (${winnerColor[0]}, ${winnerColor[1]}, ${winnerColor[2]})`;
        winnerAnnouncementElement.style.color = `rgb(${winnerColor[0]}, ${winnerColor[1]}, ${winnerColor[2]})`;
        winnerAnnouncementElement.style.display = "block";
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
      }
      
      // Drawing data received
      if (data.hasOwnProperty('x1')) {
        const { x1, y1, x2, y2, color, brush_thickness, pixel_perc, power_up_id, client_id } = data;
        
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
                  const x = Math.floor(Math.random() * 1160) + 50;
                  const y = Math.floor(Math.random() * 400) + 50;
                  drawingCtx.beginPath();
                  drawingCtx.arc(x, y, 100, 0, 2 * Math.PI);
                  drawingCtx.fillStyle = `rgb(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
                  drawingCtx.fill();
                  
                  // Send area update
                  setTimeout(() => updateAndSendPixelPercentage(), 100);
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
              }
              
              break;
            }
          }
        }
        
        // Draw on canvas if it's not our own drawing (we'll handle our own drawing separately)
        if (client_id !== selfId) {
          drawingCtx.beginPath();
          drawingCtx.moveTo(x1, y1);
          drawingCtx.lineTo(x2, y2);
          drawingCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          drawingCtx.lineWidth = brush_thickness;
          drawingCtx.lineCap = 'round';
          drawingCtx.stroke();
        }
        
        // Update client data for score display
        if (pixel_perc !== undefined) {
          const colorKey = Array.isArray(color) ? `${color[0]},${color[1]},${color[2]}` : color;
          clientData[colorKey] = pixel_perc;
          updateScores();
        }
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
    powerUpElement.style.left = `${powerUp.x}px`;
    powerUpElement.style.top = `${powerUp.y}px`;
    
    // Add different colors/icons based on power-up type
    switch(powerUp.type) {
      case 'eraser':
        powerUpElement.textContent = 'E';
        powerUpElement.style.backgroundColor = 'white';
        powerUpElement.style.color = 'black';
        break;
      case 'devil_face':
        powerUpElement.textContent = 'D';
        powerUpElement.style.backgroundColor = 'red';
        break;
      case 'paint_bucket':
        powerUpElement.textContent = 'B';
        powerUpElement.style.backgroundColor = 'blue';
        break;
      case 'paint_brush':
        powerUpElement.textContent = 'P';
        powerUpElement.style.backgroundColor = 'green';
        break;
      default:
        powerUpElement.textContent = '?';
    }
    
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
  let scoresHTML = "<strong>Player Scores:</strong><br>";
  
  // Sort client data by percentage (descending)
  const sortedEntries = Object.entries(clientData).sort((a, b) => b[1] - a[1]);
  
  for (const [colorStr, percentage] of sortedEntries) {
    let color;
    if (colorStr.includes(',')) {
      color = colorStr.split(",").map(Number);
    } else {
      continue; // Skip invalid entries
    }
    
    const colorCSS = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    
    if (color[0] === DRAW_COLOR[0] && color[1] === DRAW_COLOR[1] && color[2] === DRAW_COLOR[2]) {
      scoresHTML += `<div style="color: ${colorCSS}; font-weight: bold; margin-bottom: 5px;">
                      YOU: ${percentage.toFixed(2)}%
                    </div>`;
    } else if (color[0] === 0 && color[1] === 0 && color[2] === 0) {
      // Skip black color
      continue;
    } else {
      scoresHTML += `<div style="color: ${colorCSS}; margin-bottom: 5px;">
                      Player (${color[0]}, ${color[1]}, ${color[2]}): ${percentage.toFixed(2)}%
                    </div>`;
    }
  }
  
  playerScoresElement.innerHTML = scoresHTML;
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

// Check for power-up collection
function checkPowerUpCollection(x, y) {
  for (let i = 0; i < powerUpsAvailable.length; i++) {
    const powerUp = powerUpsAvailable[i];
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
    client_id: selfId
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

// Apply paint brush power-up effect
function applyPaintBrush() {
  brushThickness = 60; // Larger brush
  
  // Reset to normal brush after 5 seconds
  setTimeout(() => {
    brushThickness = 25;
  }, 5000);
}

// Apply eraser power-up effect
function applyEraser() {
  // Store original color
  DRAW_COLOR_TEMP = [255, 255, 255]; // White for eraser
  
  // Reset to original color after 5 seconds
  setTimeout(() => {
    DRAW_COLOR_TEMP = null;
  }, 5000);
}

// Apply paint brush power-up effect
function applyDevilFace() {
  brushThickness = 1; // Broken brush
  
  // Reset to normal brush after 5 seconds
  setTimeout(() => {
    brushThickness = 25;
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
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,{color: 'rgba(255, 255, 255, 0.4)', lineWidth: 0.5});
    drawLandmarks(canvasCtx, landmarks, {color: 'rgba(255, 255, 255, 0.5)', lineWidth: 0.5, radius: 2});

    // Get index finger position
    const indexFinger = landmarks[8];
    const x = canvasElement.width - indexFinger.x * canvasElement.width;
    const y = indexFinger.y * canvasElement.height;

    // Handle two fingers up (selection mode)
    if (isTwoFingersUp(landmarks)) {
      // Check if in start area
      if (!gameActive && !gameCountdown && isInArea(x, y, 340, 300, 200, 100)) {
        startAreaElement.style.backgroundColor = "rgba(0,255,0,0.3)";
        // On click effect - send start command
        if (!gameReset) {
          websocket.send(JSON.stringify({ type: "start" }));
        }
      } else {
        startAreaElement.style.backgroundColor = "transparent";
      }
      
      // Check if in reset area
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
        drawingCtx.beginPath();
        drawingCtx.moveTo(xp, yp);
        drawingCtx.lineTo(x, y);
        drawingCtx.strokeStyle = DRAW_COLOR_TEMP 
          ? `rgb(${DRAW_COLOR_TEMP[0]}, ${DRAW_COLOR_TEMP[1]}, ${DRAW_COLOR_TEMP[2]})` 
          : `rgb(${DRAW_COLOR[0]}, ${DRAW_COLOR[1]}, ${DRAW_COLOR[2]})`;
        drawingCtx.lineWidth = brushThickness;
        drawingCtx.lineCap = 'round';
        drawingCtx.stroke();
        
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
  
  canvasCtx.restore();
});

// Initialize camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 522
});

camera.start();

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
  const splash = document.querySelector('.splash');
  
  // Keep aspect ratio consistent
  const maxWidth = Math.min(window.innerWidth, 1280);
  container.style.width = `${maxWidth}px`;
  videoContainer.style.height = `${(maxWidth / 1280) * 522}px`;
  splash.style.height = `${(maxWidth / 1280) * 198}px`;
});

// Handle page visibility change (pause/resume game sounds)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    gameplaySound.pause();
  } else if (gameActive) {
    gameplaySound.play();
  }
});