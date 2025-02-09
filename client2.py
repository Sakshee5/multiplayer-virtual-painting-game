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

cap = cv2.VideoCapture('client2.mp4')
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)  # Shared canvas for drawing


def get_pixel_percent(color):
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
        "pixel_perc": get_pixel_percent(DRAW_COLOR),
    }
    try:
        await websocket.send(json.dumps(data))
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed while sending data: {e}")


async def process_frame(img, websocket):
    global xp, yp, img_canvas, DRAW_COLOR
    img = detector.findHands(img)
    lm_list, _ = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]  # Tip of index finger
        fingers = detector.fingersUp()

        # If in drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, DRAW_COLOR, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on the local canvas
            cv2.line(img_canvas, (xp, yp), (x1, y1), DRAW_COLOR, brush_thickness)

            # Send drawing data to the server
            await send_draw_data(websocket, xp, yp, x1, y1)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

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
                try:
                    async for message in websocket:
                        data = json.loads(message)
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
                img = cv2.resize(img, (1280, 720))
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

                pixel_perc = get_pixel_percent(DRAW_COLOR)
                pixel_count_text = f"Color {DRAW_COLOR}: {pixel_perc:.2f}%"
                cv2.putText(img, pixel_count_text, (650, 555), cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW_COLOR, 2)

                i = 40
                for color, pixel_perc in client_data.items():
                    pixel_count_text = f"Color {color}: {pixel_perc:.2f}%"
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