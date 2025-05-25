import cv2
import mediapipe as mp
import numpy as np
import socket
import joblib
import threading
import time
import pygame
import sys
import math

# ==================== SETTING ====================
MODEL_PATH = "rf_pose_model.pkl"
# ==================================================
ESP32_HOST = "192.168.1.2"
ESP32_PORT = 1234

# Shared variables
current_pose = "stop"
running = True
pose_lock = threading.Lock()

# Variabel untuk status koneksi ESP
esp_connected = False
client_socket = None

# Inisialisasi pygame untuk simulator
pygame.init()

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
SKY_BLUE = (135, 206, 235)

# Ukuran window
SIMULATOR_WIDTH = 800
SIMULATOR_HEIGHT = 600
ALTITUDE_WIDTH = 300
ALTITUDE_HEIGHT = 500

# Buat window utama
screen = pygame.display.set_mode((SIMULATOR_WIDTH + ALTITUDE_WIDTH, max(SIMULATOR_HEIGHT, ALTITUDE_HEIGHT)))
pygame.display.set_caption("UAV Simulator")

# Parameter UAV
uav_size = 40
uav_x = SIMULATOR_WIDTH // 2
uav_y = SIMULATOR_HEIGHT // 2
uav_z = 0  # Ketinggian (0 = ground, 100 = max)
uav_yaw = 0  # Orientasi (dalam derajat)
uav_speed_x = 0
uav_speed_y = 0
uav_speed_z = 0

# Parameter gerakan
base_speed = 2
altitude_speed = 1
rotation_speed = 3

def draw_uav(surface, x, y, size, yaw):
    # Gambar badan utama UAV (persegi)
    body_rect = pygame.Rect(x - size//2, y - size//2, size, size)
    pygame.draw.rect(surface, BLUE, body_rect)
    
    # Gambar baling-baling (4 garis dari pusat)
    for angle in [0, 90, 180, 270]:
        end_x = x + (size//2 + 10) * math.cos(math.radians(angle + yaw))
        end_y = y + (size//2 + 10) * math.sin(math.radians(angle + yaw))
        pygame.draw.line(surface, RED, (x, y), (end_x, end_y), 2)
    
    # Gambar indikator arah depan (segitiga)
    front_x = x + (size//2 + 5) * math.cos(math.radians(yaw))
    front_y = y + (size//2 + 5) * math.sin(math.radians(yaw))
    pygame.draw.polygon(surface, GREEN, [
        (front_x, front_y),
        (x + (size//4) * math.cos(math.radians(yaw + 120)), 
        y + (size//4) * math.sin(math.radians(yaw + 120))),
        (x + (size//4) * math.cos(math.radians(yaw - 120)), 
        y + (size//4) * math.sin(math.radians(yaw - 120)))
    ])

def draw_altitude_indicator(altitude, z_speed):
    # Buat surface untuk altitude indicator
    indicator_surface = pygame.Surface((ALTITUDE_WIDTH, ALTITUDE_HEIGHT))
    indicator_surface.fill(WHITE)
    
    # Gambar latar belakang gradien (biru langit ke putih)
    for y in range(ALTITUDE_HEIGHT):
        ratio = y / ALTITUDE_HEIGHT
        r = int(SKY_BLUE[0] * (1 - ratio) + WHITE[0] * ratio)
        g = int(SKY_BLUE[1] * (1 - ratio) + WHITE[1] * ratio)
        b = int(SKY_BLUE[2] * (1 - ratio) + WHITE[2] * ratio)
        pygame.draw.line(indicator_surface, (r, g, b), (0, y), (ALTITUDE_WIDTH, y))
    
    # Gambar garis-garis horizontal untuk indikator ketinggian
    for alt in range(0, 101, 10):
        y_pos = ALTITUDE_HEIGHT - (alt * ALTITUDE_HEIGHT // 100)
        pygame.draw.line(indicator_surface, BLACK, (50, y_pos), (ALTITUDE_WIDTH, y_pos), 1)
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"{alt}m", True, BLACK)
        indicator_surface.blit(text, (10, y_pos - 10))
    
    # Gambar UAV kecil di indikator ketinggian
    uav_indicator_y = ALTITUDE_HEIGHT - (altitude * ALTITUDE_HEIGHT // 100)
    pygame.draw.circle(indicator_surface, RED, (30, uav_indicator_y), 15)
    
    # Tambahkan label
    font = pygame.font.SysFont(None, 36)
    title = font.render("ALTITUDE", True, BLACK)
    indicator_surface.blit(title, (ALTITUDE_WIDTH//2 - 50, 20))
    
    # Gambar panah naik/turun berdasarkan gerakan
    if z_speed > 0:  # Naik
        pygame.draw.polygon(indicator_surface, GREEN, [
            (ALTITUDE_WIDTH - 40, uav_indicator_y - 30),
            (ALTITUDE_WIDTH - 20, uav_indicator_y - 30),
            (ALTITUDE_WIDTH - 30, uav_indicator_y - 50)
        ])
    elif z_speed < 0:  # Turun
        pygame.draw.polygon(indicator_surface, RED, [
            (ALTITUDE_WIDTH - 40, uav_indicator_y + 30),
            (ALTITUDE_WIDTH - 20, uav_indicator_y + 30),
            (ALTITUDE_WIDTH - 30, uav_indicator_y + 50)
        ])
    
    return indicator_surface

def uav_simulator():
    global uav_x, uav_y, uav_z, uav_yaw, uav_speed_x, uav_speed_y, uav_speed_z
    
    clock = pygame.time.Clock()
    
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Get current pose from shared variable
        with pose_lock:
            pose = current_pose
        
        # Set UAV movement based on pose
        if pose == "maju":
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw))
        elif pose == "mundur":
            uav_speed_x = -base_speed * math.cos(math.radians(uav_yaw))
            uav_speed_y = -base_speed * math.sin(math.radians(uav_yaw))
        elif pose == "kiri":
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw - 90))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw - 90))
        elif pose == "kanan":
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw + 90))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw + 90))
        elif pose == "rotasikanan":
            uav_yaw += rotation_speed
            uav_speed_x = 0
            uav_speed_y = 0
        elif pose == "rotasikiri":
            uav_yaw -= rotation_speed
            uav_speed_x = 0
            uav_speed_y = 0
        elif pose == "naik":
            uav_speed_z = altitude_speed
        elif pose == "turun":
            uav_speed_z = -altitude_speed
        elif pose == "hover":
            uav_speed_x = 0
            uav_speed_y = 0
            uav_speed_z = 0
        else:  # stop
            uav_speed_x = 0
            uav_speed_y = 0
            uav_speed_z = 0
        
        # Update position
        uav_x += uav_speed_x
        uav_y += uav_speed_y
        uav_z += uav_speed_z
        
        # Normalize yaw angle
        uav_yaw %= 360
        
        # Boundary checks
        uav_x = max(uav_size//2, min(SIMULATOR_WIDTH - uav_size//2, uav_x))
        uav_y = max(uav_size//2, min(SIMULATOR_HEIGHT - uav_size//2, uav_y))
        uav_z = max(0, min(100, uav_z))  # Ketinggian antara 0-100
        
        # Bersihkan layar
        screen.fill(WHITE)
        
        # Buat surface untuk simulator utama
        simulator_surface = pygame.Surface((SIMULATOR_WIDTH, SIMULATOR_HEIGHT))
        simulator_surface.fill(WHITE)
        
        # Draw grid (ground)
        for x in range(0, SIMULATOR_WIDTH, 50):
            pygame.draw.line(simulator_surface, GRAY, (x, 0), (x, SIMULATOR_HEIGHT))
        for y in range(0, SIMULATOR_HEIGHT, 50):
            pygame.draw.line(simulator_surface, GRAY, (0, y), (SIMULATOR_WIDTH, y))
        
        # Draw UAV
        draw_uav(simulator_surface, uav_x, uav_y, uav_size, uav_yaw)
        
        # Display status information
        font = pygame.font.SysFont(None, 36)
        info_texts = [
            f"Pose: {pose}",
            f"Position: X={uav_x:.1f}, Y={uav_y:.1f}",
            f"Altitude: {uav_z:.1f}m",
            f"Yaw: {uav_yaw:.1f}°",
            f"Speed: X={uav_speed_x:.1f}, Y={uav_speed_y:.1f}, Z={uav_speed_z:.1f}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, BLACK)
            simulator_surface.blit(text_surface, (10, 10 + i * 40))
        
        # Draw altitude indicator
        altitude_surface = draw_altitude_indicator(uav_z, uav_speed_z)
        
        # Gabungkan kedua surface ke layar utama
        screen.blit(simulator_surface, (0, 0))
        screen.blit(altitude_surface, (SIMULATOR_WIDTH, 0))
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

# Load model
model = joblib.load(MODEL_PATH)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Setup kamera
cap = cv2.VideoCapture(0)

def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        keypoints = [0] * 132
    return np.array(keypoints)

# Mulai thread simulator
sim_thread = threading.Thread(target=uav_simulator)
sim_thread.daemon = True
sim_thread.start()

prev_label = None
try:
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Add bounding box
        height, width = frame.shape[:2]
        box_width, box_height = int(width * 0.7), int(height * 0.7)
        box_x = (width - box_width) // 2
        box_y = (height - box_height) // 2
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 2)
        cv2.putText(frame, "Posisikan diri dalam kotak", (box_x, box_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Pose detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        keypoints = extract_keypoints(results)
        X = keypoints.reshape(1, -1)
        label = model.predict(X)[0]

        valid_poses = ["maju", "mundur", "kiri", "kanan", "rotasikanan", "rotasikiri", 
                      "naik", "turun", "hover", "stop"]
        if label not in valid_poses:
            label = "hover"

        cv2.putText(frame, f"Pose: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not results.pose_landmarks:
            cv2.putText(frame, "Pose tidak terdeteksi!", (width // 4, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Pose Detection", frame)

        if label != prev_label:
            with pose_lock:
                current_pose = label
            
            if esp_connected and client_socket:
                try:
                    message = label + "\n"
                    client_socket.sendall(message.encode())
                    print(f"Mengirim ke ESP32: {label}")
                except Exception as e:
                    print(f"Error kirim data: {e}")
                    esp_connected = False
                    if client_socket:
                        client_socket.close()
                    client_socket = None
            
            prev_label = label

        if cv2.waitKey(1) & 0xFF == ord('q'):
            with pose_lock:
                running = False
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if client_socket:
        client_socket.close()
    with pose_lock:
        running = False
    sim_thread.join()
    pygame.quit()
    print("Program berhenti.")