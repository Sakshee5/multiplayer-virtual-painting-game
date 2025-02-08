import asyncio
import websockets
import json
import numpy as np
import cv2

# Global variables
connected_clients = {}
draw_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
color_pixel_perc = {str(color): 0 for color in draw_colors}
img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Initialize the shared canvas
game_duration = 10
game_active = False  # To track whether the game is active
connected_clients_lock = asyncio.Lock()  # To manage client connections safely

async def handle_start_game():
    """
    Start the game and reset the canvas and scores.
    """
    global color_pixel_perc, img_canvas, game_active
    print("Game started!")
    img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Reset canvas
    color_pixel_perc = {str(color): 0 for color in color_pixel_perc}  # Reset scores
    game_active = True  # Mark game as active

    async with connected_clients_lock:
        for client in connected_clients:
            try:
                await client.send(json.dumps({"type": "start"}))
            except websockets.exceptions.ConnectionClosedError:
                print(f"Client disconnected during game start.")
                await remove_client(client)

async def handle_reset_game():
    """
    Reset the game without stopping the server.
    """
    global game_active
    game_active = False  # Stop any ongoing game
    print("Game reset!")
    async with connected_clients_lock:
        for client in list(connected_clients):
            try:
                await client.send(json.dumps({"type": "reset"}))
            except websockets.exceptions.ConnectionClosedError:
                print(f"Client disconnected during reset.")
                await remove_client(client)

async def game_timer():
    """
    Timer for the game duration. Ends the game when the time runs out.
    """
    global game_active, color_pixel_perc
    for t in range(game_duration, -1, -1):
        if not game_active:  # Stop the timer if reset occurs
            print("Game timer stopped due to reset.")
            return

        async with connected_clients_lock:
            for client in connected_clients:
                try:
                    await client.send(json.dumps({"type": "timer", "time_left": t}))
                except websockets.exceptions.ConnectionClosedError:
                    print("Client disconnected during timer.")
                    await remove_client(client)
        await asyncio.sleep(1)

    # End game logic
    game_active = False
    winner_color = max(color_pixel_perc, key=color_pixel_perc.get)
    print(f"Winner is color {winner_color} with {color_pixel_perc[winner_color]} pixels!")
    async with connected_clients_lock:
        for client in connected_clients:
            try:
                await client.send(json.dumps({"type": "winner", "winner": winner_color}))
            except websockets.exceptions.ConnectionClosedError:
                print("Client disconnected during winner announcement.")
                await remove_client(client)

async def remove_client(client):
    """
    Remove a disconnected client from the connected clients list.
    """
    if client in connected_clients:
        print(f"Removing client {connected_clients[client]}")
        del connected_clients[client]

async def handler(websocket):
    """
    Handle incoming client connections and manage their interactions.
    """
    global connected_clients, color_pixel_perc
    async with connected_clients_lock:
        if len(connected_clients) < len(draw_colors):
            assigned_color = draw_colors[len(connected_clients)]
        else:
            assigned_color = draw_colors[len(connected_clients) % len(draw_colors)]

        await websocket.send(str(assigned_color))
        connected_clients[websocket] = str(assigned_color)

    try:
        async for message in websocket:
            data = json.loads(message)

            # Start game signal
            if data.get("type") == "start":
                asyncio.create_task(game_timer())
                await handle_start_game()

            # Reset game signal
            elif data.get("type") == "reset":
                await handle_reset_game()

            # Drawing logic
            elif "x1" in data:  # Allow drawing only if game is active
                if game_active:
                    color = tuple(data["color"])
                    pixel_perc = data["pixel_perc"]
                    color_pixel_perc[str(color)] = pixel_perc
                    print(data)

                    async with connected_clients_lock:
                        for client in connected_clients:
                            await client.send(json.dumps(data))

    except websockets.exceptions.ConnectionClosedError:
        print("Client disconnected unexpectedly.")
    finally:
        async with connected_clients_lock:
            await remove_client(websocket)

async def main():
    print("Starting server at ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Keeps the server running indefinitely

if __name__ == "__main__":
    asyncio.run(main())