import asyncio
import websockets
import cv2
import numpy as np
import json
import hand_tracker as ht
import ast
import random

import pygame

pygame.mixer.init()

# SERVER = "wss://0c41-2603-6080-65f0-2c0-4c00-32a5-cf6-6923.ngrok-free.app"
SERVER = "ws://localhost:8765"

# Load audio files
countdown_audio_beep = 'countdown_beep.mp3'
countdown_audio_go = 'countdown_go.mp3'
gameplay_audio = 'gameplay.mp3'  # Background music

def play_countdown_audio_go():
    pygame.mixer.music.load(countdown_audio_go)
    pygame.mixer.music.play()

def play_countdown_audio_beep():
    pygame.mixer.music.load(countdown_audio_beep)
    pygame.mixer.music.play()

def play_gameplay_audio():
    pygame.mixer.music.load(gameplay_audio)
    pygame.mixer.music.play()

def stop_audio():
    pygame.mixer.music.stop()

brush_thickness = 25

overlay_image = cv2.imread('splashh.png')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
img_canvas = np.zeros((500, 1260, 3), np.uint8)  # Shared canvas for drawing
game_active = False
game_reset = False
game_countdown = False
self_id = None
DRAW_COLOR_TEMP = None

power_ups_available = []  # Track power-ups on the client

async def check_power_up_collection(lm_list):
    """
    Check if the player's index finger tip touches a power-up.
    """
    global power_ups_available, self_id
    x1, y1 = lm_list[8][1:]  # Tip of the index finger

    for power_up in power_ups_available:
        if abs(x1 - power_up["x"]) < 20 and abs(y1 - power_up["y"]) < 20:
            return power_up["id"], self_id
        
    return None, None

async def get_pixel_percent(color):
    global img_canvas
    # Count the pixels that match the target color
    matching_pixels = np.all(img_canvas == color, axis=-1)
    matching_pixel_count = np.sum(matching_pixels)

    # Calculate total number of pixels
    total_pixels = img_canvas.shape[0] * img_canvas.shape[1]

    # Calculate percentage
    percentage_covered = (matching_pixel_count / total_pixels) * 100

    return percentage_covered

async def send_draw_data(websocket, x1, y1, x2, y2, power_up_id, client_id):
    global img_canvas

    if DRAW_COLOR_TEMP:
        data = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "color": DRAW_COLOR_TEMP,
            "brush_thickness": brush_thickness,
            "pixel_perc": await get_pixel_percent(DRAW_COLOR),
            "power_up_id": power_up_id,
            "client_id": client_id
        }
    else:
        data = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "color": DRAW_COLOR,
            "brush_thickness": brush_thickness,
            "pixel_perc": await get_pixel_percent(DRAW_COLOR),
            "power_up_id": power_up_id,
            "client_id": client_id
        }
    try:
        await websocket.send(json.dumps(data))
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed while sending data: {e}")

async def check_start_gesture(lm_list):
    global game_active
    fingers = detector.fingersUp()
    if fingers[1] and fingers[2]:
        x, y = lm_list[8][1:]
        if 300 < y < 400 and 340 < x < 540:
            game_active = True
            return True
    return False

async def check_reset_gesture(lm_list):
    global game_reset
    fingers = detector.fingersUp()
    if fingers[1] and fingers[2]:
        x, y = lm_list[8][1:]
        if 300 < y < 400 and 740 < x < 940:
            game_reset = True
            return True
    return False

async def process_frame(img, websocket):
    global xp, yp, img_canvas, DRAW_COLOR, game_active, game_reset, game_countdown, power_ups_available, DRAW_COLOR_TEMP
    img = detector.findHands(img)
    lm_list, _ = detector.findPosition(img, draw=False)

    if not game_active and not game_reset and not game_countdown:
        power_ups_available = []
        # Show start button
        cv2.rectangle(img, (340, 300), (540, 400), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "START", (350, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        if len(lm_list) != 0 and await check_start_gesture(lm_list):
            await websocket.send(json.dumps({"type": "start"}))

    if not game_active and game_countdown:
        pass

    if game_active and len(lm_list) != 0 and not game_reset:
    # if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]  # Tip of index finger
        fingers = detector.fingersUp()

        # Check for power-up collection
        power_up_id, client_id = await check_power_up_collection(lm_list)

        # If in drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, DRAW_COLOR, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on the shared canvas
            if DRAW_COLOR_TEMP:
                cv2.line(img_canvas, (xp, yp), (x1, y1), DRAW_COLOR_TEMP, brush_thickness)
            else:
                cv2.line(img_canvas, (xp, yp), (x1, y1), DRAW_COLOR, brush_thickness)

            # Send drawing data to the server
            await send_draw_data(websocket, xp, yp, x1, y1, power_up_id, client_id)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    elif not game_active and game_reset:
        power_ups_available = []
        cv2.rectangle(img_canvas, (740, 300), (940, 400), (255, 255, 255), cv2.FILLED)
        cv2.putText(img_canvas, "RESET", (750, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        if len(lm_list) != 0 and await check_reset_gesture(lm_list):
            await websocket.send(json.dumps({"type": "reset"}))

    await asyncio.sleep(0)
    return img


async def apply_devil_face():
    global brush_thickness
    old_brush_thickness = brush_thickness
    brush_thickness -= 24  # Increase brush thickness
    await asyncio.sleep(5)  # Effect lasts 5 seconds
    brush_thickness = old_brush_thickness  # Revert back to the original thickness

async def apply_paint_brush():
    global brush_thickness
    old_brush_thickness = brush_thickness
    brush_thickness += 20  # Increase brush thickness
    await asyncio.sleep(5)  # Effect lasts 5 seconds
    brush_thickness = old_brush_thickness  # Revert back to the original thickness

# Function to handle the eraser effect
async def apply_eraser():
    global DRAW_COLOR_TEMP, brush_thickness
    old_brush_thickness = brush_thickness
    brush_thickness += 20
    DRAW_COLOR_TEMP = (0, 0, 0)
    await asyncio.sleep(5)  # Effect lasts 5 seconds
    DRAW_COLOR_TEMP = None  # Revert to the original color
    brush_thickness = old_brush_thickness

async def main_client():
    global img_canvas, DRAW_COLOR, self_id
    try:
        async with websockets.connect(SERVER) as websocket:
            color_data = await websocket.recv()
            DRAW_COLOR = ast.literal_eval(color_data)
            print(f"Assigned color: {DRAW_COLOR}")

            # Start listening for incoming draw data
            client_data = {}
            async def receive_draw_data(websocket):
                nonlocal client_data
                global game_active, game_reset, img_canvas, game_countdown, self_id, power_ups_available, DRAW_COLOR_TEMP
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if "type" in data and data["type"] == "client_list" and not game_active and not game_countdown and not game_reset:
                            img_canvas = np.zeros((500, 1260, 3), dtype=np.uint8)
                            connected_clients = data['clients']
                            self_id = data.get("self_id")  # Get client's own ID

                            i = 40
                            for client, color in connected_clients.items():
                                if str(client) == str(self_id):
                                    text = f"Connected - YOU: Color {DRAW_COLOR}"
                                    cv2.putText(img_canvas, text, (650, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW_COLOR, 2)
                                else:
                                    text = f"Connected - Player {ast.literal_eval(color)}"
                                    cv2.putText(img_canvas, text, (650, 85 + i), cv2.FONT_HERSHEY_SIMPLEX, 1, ast.literal_eval(color), 2)
                                    i += 40
                            
                        
                        if "type" in data and data["type"] == "countdown":
                            if data['count'] == "GO":
                                play_countdown_audio_go()
                            else:
                                play_countdown_audio_beep()

                            # Handle countdown display
                            game_active = False
                            game_countdown = True
                            count = data["count"]
                            img_canvas = np.zeros((500, 1260, 3), dtype=np.uint8)  # Clear the canvas for countdown
                            cv2.putText(img_canvas, str(count), (580, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
                            cv2.waitKey(100)  # Wait 100ms to ensure update

                        if "type" in data and data["type"] == "start":
                            play_gameplay_audio()
                            img_canvas = np.zeros((500, 1260, 3), dtype=np.uint8)  # Clear the canvas for countdown
                            game_active = True
                            game_reset = False
                            game_countdown = False

                        if "type" in data and data["type"] == "power_up_spawn":
                            # power_ups_available store {"type": power_up["type"], "x": x, "y": y, "image": power_up["image"], "id": power_up_id}
                            power_ups_available.append(data["power_up"])

                        if "type" in data and data["type"] == "reset":
                            game_active = False
                            game_reset = False
                            img_canvas = np.zeros((500, 1260, 3), np.uint8)  # Reset canvas
                            for color in client_data:
                                client_data[color] = 0

                        if "type" in data and data["type"] == "winner":
                            stop_audio()
                            game_active = False
                            game_reset = True
                            winner = data["winner"]
                            print(f"Winner is {winner}")

                            cv2.putText(img_canvas, f"Winner: {winner}", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)

                            # Display winner on screen
                            cv2.putText(img_canvas, f"Winner: {winner}", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, ast.literal_eval(winner), 3)


                        if "type" in data and data["type"] == "timer":
                            time_left = data["time_left"]
                            # Display timer on screen
                            cv2.rectangle(img_canvas, (1000, 10), (1250, 60), (255, 255, 255), cv2.FILLED)
                            cv2.putText(img_canvas, f"Time Left: {time_left}s", (1010, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                        if "x1" in data:
                            x1, y1, x2, y2, pixel_perc, power_up_id, client_id = data["x1"], data["y1"], data["x2"], data["y2"], data["pixel_perc"], data["power_up_id"], data["client_id"]

                            for pu in power_ups_available:
                                if pu["id"] == power_up_id:
                                    power_up_type = pu["type"]
                                    power_ups_available.remove(pu)

                                if self_id == client_id:
                                    if power_up_type == "paint_bucket":
                                        x = random.randint(50, 1210)  # Random x-coordinate within canvas bounds
                                        y = random.randint(50, 450)
                                        cv2.circle(img_canvas, (x, y), 100, DRAW_COLOR, -1)

                                    if power_up_type == "paint_brush":
                                        asyncio.create_task(apply_paint_brush())

                                    if power_up_type == "eraser":
                                        asyncio.create_task(apply_eraser())

                                    if power_up_type == "devil_face":
                                        asyncio.create_task(apply_devil_face())

                            color = tuple(data["color"])
                            thickness = data["brush_thickness"]

                            # Draw on the shared canvas
                            if client_id == self_id:
                                pass
                            else:
                                cv2.line(img_canvas, (x1, y1), (x2, y2), color, thickness)

                            # Update received data
                            client_data[color] = pixel_perc

                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"Connection closed while receiving data: {e}")

            receive_task = asyncio.create_task(receive_draw_data(websocket))

            while True:
                success, img = cap.read()
                img = cv2.flip(img, 1)
                cv2.rectangle(img, (10, 10), (1270, 510), (0, 0, 0), 3)
                cv2.putText(img, "DRAW HERE", (0 + 20, 0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                webcam_top_part = img[10:510, 10:1270]

                # Process frame and send data
                img = await process_frame(img, websocket)

                # Combine the canvas with the live feed
                img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                
                canvas_part = cv2.bitwise_and(webcam_top_part, img_inv)
                canvas_part = cv2.bitwise_or(canvas_part, img_canvas)

                img[10:510, 10:1270] = canvas_part
                
                overlay_image = cv2.imread('splashh.png')
                img[522:720, 0:1280] = overlay_image

                pixel_perc = await get_pixel_percent(DRAW_COLOR)
            
                i = 40
                for color, pixel_perc in client_data.items():
                    if color == DRAW_COLOR:
                        pixel_count_text = f"YOU: Color {DRAW_COLOR}: {pixel_perc:.2f}%"
                        cv2.putText(img, pixel_count_text, (650, 555), cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW_COLOR, 2)

                    elif color == (0, 0, 0):
                        pass
                    
                    else:
                        pixel_count_text = f"Player {color}: {pixel_perc:.2f}%"
                        cv2.putText(img, pixel_count_text, (650, 555 + i), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        i += 40

                for power_up in power_ups_available:
                    power_up_img = cv2.imread(power_up["image"], cv2.IMREAD_UNCHANGED)
                    # Check if the image has an alpha channel (4 channels)
                    if power_up_img.shape[2] == 4:
                        power_up_img = cv2.cvtColor(power_up_img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

                    power_up_img = cv2.resize(power_up_img, (32, 32))

                    x, y = power_up["x"], power_up["y"]
                    img[y:y + power_up_img.shape[0], x:x + power_up_img.shape[1]] = power_up_img


                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up tasks
            receive_task.cancel()

    except Exception as e:
        print(f"Error in WebSocket connection: {e}")


if __name__ == "__main__":
    asyncio.run(main_client())
