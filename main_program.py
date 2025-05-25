import cv2
import mediapipe as mp
import numpy as np
import socket
import joblib
import threading
import time
from shared_vars import current_pose, running, pose_lock
import pygame
import sys
import math

# ==================== SETTING ====================
MODEL_PATH = "rf_pose_model.pkl"
# ==================================================
ESP32_HOST = "192.168.1.2"
ESP32_PORT = 1234

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

# Ukuran window simulator
SIMULATOR_WIDTH = 800
SIMULATOR_HEIGHT = 600

# Buat window simulator
simulator_screen = pygame.display.set_mode((SIMULATOR_WIDTH, SIMULATOR_HEIGHT))
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

# Fungsi untuk menggambar UAV
def draw_uav():
    # Gambar badan utama UAV (persegi)
    body_rect = pygame.Rect(uav_x - uav_size//2, uav_y - uav_size//2, uav_size, uav_size)
    pygame.draw.rect(simulator_screen, BLUE, body_rect)
    
    # Gambar baling-baling (4 garis dari pusat)
    for angle in [0, 90, 180, 270]:
        end_x = uav_x + (uav_size//2 + 10) * math.cos(math.radians(angle + uav_yaw))
        end_y = uav_y + (uav_size//2 + 10) * math.sin(math.radians(angle + uav_yaw))
        pygame.draw.line(simulator_screen, RED, (uav_x, uav_y), (end_x, end_y), 2)
    
    # Gambar indikator arah depan (segitiga)
    front_x = uav_x + (uav_size//2 + 5) * math.cos(math.radians(uav_yaw))
    front_y = uav_y + (uav_size//2 + 5) * math.sin(math.radians(uav_yaw))
    pygame.draw.polygon(simulator_screen, GREEN, [
        (front_x, front_y),
        (uav_x + (uav_size//4) * math.cos(math.radians(uav_yaw + 120)), 
        uav_y + (uav_size//4) * math.sin(math.radians(uav_yaw + 120))),
        (uav_x + (uav_size//4) * math.cos(math.radians(uav_yaw - 120)), 
        uav_y + (uav_size//4) * math.sin(math.radians(uav_yaw - 120)))
    ])
    
    # Gambar indikator ketinggian (bar vertikal)
    altitude_bar_height = uav_z / 100 * 100  # Skala 0-100
    pygame.draw.rect(simulator_screen, YELLOW, 
                    (uav_x - uav_size//2 - 15, uav_y - 50, 
                    10, -altitude_bar_height))

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
            # Maju relatif terhadap orientasi UAV
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw))
        elif pose == "mundur":
            # Mundur relatif terhadap orientasi UAV
            uav_speed_x = -base_speed * math.cos(math.radians(uav_yaw))
            uav_speed_y = -base_speed * math.sin(math.radians(uav_yaw))
        elif pose == "kiri":
            # Geser kiri relatif terhadap orientasi UAV
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw - 90))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw - 90))
        elif pose == "kanan":
            # Geser kanan relatif terhadap orientasi UAV
            uav_speed_x = base_speed * math.cos(math.radians(uav_yaw + 90))
            uav_speed_y = base_speed * math.sin(math.radians(uav_yaw + 90))
        elif pose == "rotasikanan":
            # Putar kanan (yaw)
            uav_yaw += rotation_speed
            uav_speed_x = 0
            uav_speed_y = 0
        elif pose == "rotasikiri":
            # Putar kiri (yaw)
            uav_yaw -= rotation_speed
            uav_speed_x = 0
            uav_speed_y = 0
        elif pose == "naik":
            # Naik (tambah ketinggian)
            uav_speed_z = altitude_speed
        elif pose == "turun":
            # Turun (kurangi ketinggian)
            uav_speed_z = -altitude_speed
        elif pose == "hover":
            # Hover (pertahankan posisi)
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
        
        # Draw everything
        simulator_screen.fill(WHITE)
        
        # Draw grid (ground)
        for x in range(0, SIMULATOR_WIDTH, 50):
            pygame.draw.line(simulator_screen, GRAY, (x, 0), (x, SIMULATOR_HEIGHT))
        for y in range(0, SIMULATOR_HEIGHT, 50):
            pygame.draw.line(simulator_screen, GRAY, (0, y), (SIMULATOR_WIDTH, y))
        
        # Draw UAV
        draw_uav()
        
        # Display status information
        font = pygame.font.SysFont(None, 36)
        info_texts = [
            f"Pose: {pose}",
            f"Position: X={uav_x:.1f}, Y={uav_y:.1f}, Z={uav_z:.1f}",
            f"Yaw: {uav_yaw:.1f}°",
            f"Speed: X={uav_speed_x:.1f}, Y={uav_speed_y:.1f}, Z={uav_speed_z:.1f}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, BLACK)
            simulator_screen.blit(text_surface, (10, 10 + i * 40))
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

try:
    ESP32_IP = socket.gethostbyname(ESP32_HOST)
    print(f"🌐 mDNS {ESP32_HOST} -> {ESP32_IP}")
    
    # Setup koneksi socket ke ESP32
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.settimeout(10)  # Timeout 2 detik untuk koneksi
        client_socket.connect((ESP32_IP, ESP32_PORT))
        esp_connected = True
        print("\u2705 Terhubung ke ESP32!")
    except Exception as e:
        print(f"\u274C Gagal konek ke ESP32: {e}")
        print("\u2139 Simulasi akan tetap berjalan tanpa koneksi ESP32")
        if client_socket:
            client_socket.close()
        client_socket = None
except socket.gaierror:
    print(f"❌ Gagal resolve {ESP32_HOST}. Simulasi akan tetap berjalan tanpa koneksi ESP32")
    client_socket = None

# Load model
model = joblib.load(MODEL_PATH)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Setup kamera
cap = cv2.VideoCapture(0)

# ==================== LOOP UTAMA ====================
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        keypoints = [0] * 132
    return np.array(keypoints)

sim_thread = threading.Thread(target=uav_simulator)
sim_thread.daemon = True  # Jadikan sebagai daemon thread
sim_thread.start()

prev_label = None
try:
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Add bounding box (rectangle) to guide user positioning
        height, width = frame.shape[:2]
        box_width, box_height = int(width * 0.7), int(height * 0.7)  # Box size is 70% of frame
        box_x = (width - box_width) // 2
        box_y = (height - box_height) // 2
        
        # Draw the bounding box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 2)
        
        # Add text instructions
        cv2.putText(frame, "Posisikan diri dalam kotak", (box_x, box_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process the image for pose detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        keypoints = extract_keypoints(results)
        X = keypoints.reshape(1, -1)
        label = model.predict(X)[0]

        # Update valid poses for UAV
        valid_poses = ["maju", "mundur", "kiri", "kanan", "rotasikanan", "rotasikiri", 
                      "naik", "turun", "hover", "stop"]
        if label not in valid_poses:
            label = "hover"  # Default to hover if pose not recognized

        # Display the detected pose
        cv2.putText(frame, f"Pose: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display warning if pose is not properly detected
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

# ==================== BERSIH-BERSIH ====================
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