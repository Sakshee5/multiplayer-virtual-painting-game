import asyncio
import websockets
import json
import numpy as np
import cv2

connected_clients = {}
draw_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
color_pixel_perc = {str(color): 0 for color in draw_colors}
game_duration = 100  # seconds
img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Initialize the shared canvas

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

async def handler(websocket):
    global connected_clients, color_pixel_perc
    # Assign a color to the new client
    if len(connected_clients) < len(draw_colors):
        assigned_color = draw_colors[len(connected_clients)]
    else:
        assigned_color = draw_colors[len(connected_clients) % len(draw_colors)]
    
    await websocket.send(str(assigned_color))
    
    connected_clients[websocket] = str(assigned_color)
    print(f"New client {websocket.remote_address} assigned color {assigned_color}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
            color = tuple(data["color"])

            # Update the canvas with the new drawing
            cv2.line(img_canvas, (x1, y1), (x2, y2), color, data["brush_thickness"])

            # Update pixel counts for the drawn color
            pixel_perc = get_pixel_percent(color)
            data["pixel_perc"] = pixel_perc
            
            # Broadcast draw data to all clients
            for client in connected_clients:
                if client != websocket:
                    await client.send(message)
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client {websocket.remote_address} disconnected")
    finally:
        del connected_clients[websocket]

async def game_timer():
    global color_pixel_perc
    await asyncio.sleep(game_duration)
    # Calculate the winner
    winner_color = max(color_pixel_perc, key=color_pixel_perc.get)
    print(f"Winner is color {winner_color} with {color_pixel_perc[winner_color]} pixels!")
    # Broadcast the winner to all clients
    for client in connected_clients:
        try:
            await client.send(json.dumps({"type": "winner", "winner": winner_color}))
        except websockets.exceptions.ConnectionClosedError:
            continue

async def main():
    global connected_clients
    print("Starting server at ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.gather(game_timer())

if __name__ == "__main__":
    asyncio.run(main())