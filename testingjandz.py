import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

previous_landmarks = []

# Tkinter window
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("800x600")

# Video feed label
video_label = tk.Label(root)
video_label.pack()

# Detected letter label
detected_label = tk.Label(root, text="Detected Letter:", font=("Arial", 20))
detected_label.pack()

# Capture video
cap = cv2.VideoCapture(0)

def detect_sign_language():
    ret, frame = cap.read()
    if not ret:
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    detected_text = ""  # Text to display

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_coords, y_coords = [], []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_coords.append(x)
                y_coords.append(y)

            # Movement pattern detection
            if previous_landmarks:
                movement_x = [x - px for x, px in zip(x_coords, previous_landmarks[-1][0])]
                movement_y = [y - py for y, py in zip(y_coords, previous_landmarks[-1][1])]

                avg_movement_x = np.mean(movement_x)
                avg_movement_y = np.mean(movement_y)

                # Check for "J" pattern
                if avg_movement_y > 0.01 and abs(avg_movement_x) > 0.01:
                    detected_text = "J"

                # Check for "Z" pattern
                if len(previous_landmarks) >= 3:
                    x1, y1 = previous_landmarks[0][0][0], previous_landmarks[0][1][0]
                    x2, y2 = previous_landmarks[1][0][0], previous_landmarks[1][1][0]
                    x3, y3 = x_coords[0], y_coords[0]

                    if x1 < x2 and x2 > x3 and y2 < y3 and x3 > x1:
                        detected_text = "Z"

            previous_landmarks.append((x_coords, y_coords))
            if len(previous_landmarks) > 10:
                previous_landmarks.pop(0)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # Update detected label text
    detected_label.config(text=f"Detected Letter: {detected_text}")

    # Update video feed in Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    # Run the function again after 10ms
    root.after(10, detect_sign_language)

# Start detection
detect_sign_language()

# Run Tkinter main loop
root.mainloop()

# Release the capture and close windows
cap.release()
