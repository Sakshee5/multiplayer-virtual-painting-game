import asyncio
import websockets
import json
import numpy as np
import random
import itertools

# Global variables
connected_clients = {}
draw_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
color_pixel_perc = {str(color): 0 for color in draw_colors}
img_canvas = np.zeros((500, 1260, 3), dtype=np.uint8)  # Initialize the shared canvas
game_duration = 10
game_active = False  # To track whether the game is active
connected_clients_lock = asyncio.Lock()  # To manage client connections safely

power_up_counter = itertools.count(start=1)  # Creates an incremental counter

def generate_power_up_id():
    return next(power_up_counter) * 100 + random.randint(0, 99)  # Ensures uniqueness with a mix of incremental and random values

# Constants for power-ups
POWER_UPS = [
    {"type": "eraser", "image": "eraser.png", "id": None},
    {"type": "devil_face", "image": "devil_face.png", "id": None},
    {"type": "paint_bucket", "image": "paint_bucket.png", "id": None},
    {"type": "paint_brush", "image": "paint_brush.png", "id": None}
]

async def spawn_power_ups():
    """
    Spawn power-ups randomly on the canvas at intervals during the game.
    """
    global game_active
    while game_active:
        power_up_id = generate_power_up_id()
        power_up = random.choice(POWER_UPS)
        x = random.randint(50, 1210)  # Random x-coordinate
        y = random.randint(50, 450)   # Random y-coordinate
        active_power_up = {"type": power_up["type"], "x": x, "y": y, "image": power_up["image"], "id": power_up_id}

        # Broadcast the new power-up to all clients
        async with connected_clients_lock:
            for client in connected_clients:
                try:
                    await client.send(json.dumps({
                        "type": "power_up_spawn",
                        "power_up": active_power_up
                    }))
                except websockets.exceptions.ConnectionClosedError:
                    await remove_client(client)

        await asyncio.sleep(random.randint(10, 20))  # Spawn every 5-10 seconds

async def handle_countdown():
    global color_pixel_perc, img_canvas, game_active
    print("Game starting countdown!")
    async with connected_clients_lock:
        # Countdown logic
        for count in [3, 2, 1, "GO"]:
            for client in connected_clients:
                try:
                    await client.send(json.dumps({"type": "countdown", "count": count}))
                except websockets.exceptions.ConnectionClosedError:
                    print(f"Client disconnected during countdown.")
                    await remove_client(client)
            await asyncio.sleep(1)  # Wait 1 second between countdowns

async def handle_start_game():
    """
    Start the game and reset the canvas and scores.
    """
    global color_pixel_perc, img_canvas, game_active
    print("Game started!")
    img_canvas = np.zeros((500, 1260, 3), dtype=np.uint8)  # Reset canvas
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


async def broadcast_client_list():
    """
    Sends an updated list of connected clients and their assigned colors to all clients.
    """
    async with connected_clients_lock:
        client_list = {str(client.remote_address): color for client, color in connected_clients.items()}
     
        for i, client in enumerate(connected_clients):
            try:
                await client.send(json.dumps({
                    "type": "client_list",
                    "clients": client_list,
                    "self_id": i+1  # Send each client their own identifier
                }))
            except websockets.exceptions.ConnectionClosedError:
                print(f"Client disconnected while sending client list.")
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

        await websocket.send([str(assigned_color)])
        connected_clients[websocket] = str(assigned_color)

    await broadcast_client_list()

    try:
        async for message in websocket:
            data = json.loads(message)

            # Start game signal
            if data.get("type") == "start":
                await handle_countdown()
                asyncio.create_task(game_timer())
                await handle_start_game()
                asyncio.create_task(spawn_power_ups())

            # Reset game signal
            elif data.get("type") == "reset":
                await handle_reset_game()

            # Drawing logic
            elif "x1" in data:  # Allow drawing only if game is active
                if game_active:
                    color = tuple(data["color"])
                    pixel_perc = data["pixel_perc"]
                    color_pixel_perc[str(color)] = pixel_perc

                    async with connected_clients_lock:
                        for client in connected_clients:
                            # if client != websocket:
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
