import asyncio
import websockets
import cv2
import numpy as np
import json
import hand_tracker as ht
import ast

SERVER = "ws://localhost:8765"
brush_thickness = 25

overlay_image = cv2.imread('splashh.png')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)  # Shared canvas for drawing
game_active = False
game_reset = False


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

async def send_draw_data(websocket, x1, y1, x2, y2):
    global img_canvas
    data = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "color": DRAW_COLOR,
        "brush_thickness": brush_thickness,
        "pixel_perc": await get_pixel_percent(DRAW_COLOR),
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
    global xp, yp, img_canvas, DRAW_COLOR, game_active, game_reset
    img = detector.findHands(img)
    lm_list, _ = detector.findPosition(img, draw=False)

    if not game_active and not game_reset:
        # Show start button
        cv2.rectangle(img, (340, 300), (540, 400), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "START", (350, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        if len(lm_list) != 0 and await check_start_gesture(lm_list):
            await websocket.send(json.dumps({"type": "start"}))

    if game_active and len(lm_list) != 0 and not game_reset:
    # if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]  # Tip of index finger
        fingers = detector.fingersUp()

        # If in drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, DRAW_COLOR, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on the local canvas
            # cv2.line(img_canvas, (xp, yp), (x1, y1), DRAW_COLOR, brush_thickness)

            # Send drawing data to the server
            await send_draw_data(websocket, xp, yp, x1, y1)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    elif not game_active and game_reset:
        cv2.rectangle(img_canvas, (740, 300), (940, 400), (255, 255, 255), cv2.FILLED)
        cv2.putText(img_canvas, "RESET", (750, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        if len(lm_list) != 0 and await check_reset_gesture(lm_list):
            await websocket.send(json.dumps({"type": "reset"}))

    await asyncio.sleep(0)
    return img


async def main_client():
    global img_canvas, DRAW_COLOR
    try:
        async with websockets.connect(SERVER) as websocket:
            color_data = await websocket.recv()
            DRAW_COLOR = ast.literal_eval(color_data)
            print(f"Assigned color: {DRAW_COLOR}")

            # Start listening for incoming draw data
            client_data = {}
            async def receive_draw_data(websocket):
                nonlocal client_data
                global game_active, game_reset, img_canvas
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if "type" in data and data["type"] == "start":
                            game_active = True
                            game_reset = False

                        if "type" in data and data["type"] == "reset":
                            game_active = False
                            game_reset = False
                            img_canvas = np.zeros((720, 1280, 3), np.uint8)  # Reset canvas
                            for color in client_data:
                                client_data[color] = 0

                        if "type" in data and data["type"] == "winner":
                            game_active = False
                            game_reset = True
                            winner = data["winner"]
                            print(f"Winner is {winner}")

                            # Display winner on screen
                            cv2.putText(img_canvas, f"Winner: {winner}", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


                        if "type" in data and data["type"] == "timer":
                            time_left = data["time_left"]
                            # Display timer on screen
                            cv2.rectangle(img_canvas, (40, 10), (280, 60), (255, 255, 255), cv2.FILLED)
                            cv2.putText(img_canvas, f"Time Left: {time_left}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

                        if "x1" in data:
                            x1, y1, x2, y2, pixel_perc = data["x1"], data["y1"], data["x2"], data["y2"], data["pixel_perc"]
                            color = tuple(data["color"])
                            thickness = data["brush_thickness"]

                            # Draw on the shared canvas
                            cv2.line(img_canvas, (x1, y1), (x2, y2), color, thickness)

                            # Update received data
                            client_data[color] = pixel_perc

                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"Connection closed while receiving data: {e}")

            receive_task = asyncio.create_task(receive_draw_data(websocket))

            while True:
                success, img = cap.read()
                img = cv2.flip(img, 1)

                # Process frame and send data
                img = await process_frame(img, websocket)

                # Combine the canvas with the live feed
                img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, img_canvas)

                overlay_image = cv2.imread('splashh.png')
                img[522:720, 0:1280] = overlay_image

                pixel_perc = await get_pixel_percent(DRAW_COLOR)
                pixel_count_text = f"YOU: Color {DRAW_COLOR}: {pixel_perc:.2f}%"
                cv2.putText(img, pixel_count_text, (650, 555), cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW_COLOR, 2)

                i = 40
                for color, pixel_perc in client_data.items():
                    pixel_count_text = f"Player {color}: {pixel_perc:.2f}%"
                    cv2.putText(img, pixel_count_text, (650, 555 + i), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    i += 40

                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Clean up tasks
            receive_task.cancel()

    except Exception as e:
        print(f"Error in WebSocket connection: {e}")


if __name__ == "__main__":
    asyncio.run(main_client())