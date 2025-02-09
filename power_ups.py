import cv2
import numpy as np
import random
import time

# Load power-up images
powerup_icons = {
    "devil": cv2.imread("devil.png", cv2.IMREAD_UNCHANGED),
    "paint_bucket": cv2.imread("paint-bucket.png", cv2.IMREAD_UNCHANGED),
    "brush": cv2.imread("paint-brush.png", cv2.IMREAD_UNCHANGED)
}

# Power-up effects
effects = {
    "devil": {"active": False, "end_time": 0},
    "paint_bucket": False,
    "brush": {"active": False, "end_time": 0}
}

# Randomly spawn power-ups
powerups = []
def spawn_powerup():
    x, y = random.randint(50, 600), random.randint(50, 400)
    type = random.choice(list(powerup_icons.keys()))
    powerups.append({"type": type, "pos": (x, y), "collected": False})

# Detect if fingertip touches power-up
def check_powerup_collection(fingertip_pos):
    global effects
    for powerup in powerups:
        if not powerup["collected"]:
            px, py = powerup["pos"]
            if abs(fingertip_pos[0] - px) < 20 and abs(fingertip_pos[1] - py) < 20:
                powerup["collected"] = True
                if powerup["type"] == "devil":
                    effects["devil"]["active"] = True
                    effects["devil"]["end_time"] = time.time() + 5
                elif powerup["type"] == "paint_bucket":
                    effects["paint_bucket"] = True
                elif powerup["type"] == "brush":
                    effects["brush"]["active"] = True
                    effects["brush"]["end_time"] = time.time() + 5

# Apply power-up effects
def apply_effects():
    global effects
    if effects["devil"]["active"] and time.time() > effects["devil"]["end_time"]:
        effects["devil"]["active"] = False
    if effects["brush"]["active"] and time.time() > effects["brush"]["end_time"]:
        effects["brush"]["active"] = False
    
    return effects

# Main game loop
while True:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Example canvas
    
    # Spawn a power-up occasionally
    if random.random() < 0.01:  # 1% chance per frame
        spawn_powerup()
    
    # Draw power-ups
    for powerup in powerups:
        if not powerup["collected"]:
            px, py = powerup["pos"]
            icon = powerup_icons[powerup["type"]]
            h, w, _ = icon.shape
            frame[py:py+h, px:px+w] = icon[:, :, :3]  # Display power-up
    
    # Example fingertip detection (replace with actual tracking logic)
    fingertip_pos = (random.randint(0, 640), random.randint(0, 480))
    check_powerup_collection(fingertip_pos)
    
    # Apply effects
    active_effects = apply_effects()
    
    # Simulate brush effect
    brush_thickness = 5 + (5 if active_effects["brush"]["active"] else 0)
    
    cv2.imshow("Game", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
