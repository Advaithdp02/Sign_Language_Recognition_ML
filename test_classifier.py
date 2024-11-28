import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Label dictionary
labels_dict = {
    0: 'next', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
    10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q',
    18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z',
    27: 'option1', 28: 'option2', 29: 'option3', 30: 'backspace'
}

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Capture video from the camera
cap = cv2.VideoCapture(0)
predicted_text = ''
before = ''
predicted_character = ''
last_prediction_time = time.time()  # Timestamp for the last prediction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data_aux = []
    x_ = []
    y_ = []
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark data for normalization
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x - min_x, lm.y - min_y])

        # Handle single-hand input by padding
        if len(result.multi_hand_landmarks) == 1:
            data_aux.extend([0] * 42)  # Padding for the second hand

        # Ensure feature vector size matches the model's expectation
        if len(data_aux) != 84:
            print(f"Feature vector size mismatch: {len(data_aux)} instead of 84")
            continue

        # Prediction delay
        current_time = time.time()
        prediction = model.predict([np.asarray(data_aux)])
        if current_time - last_prediction_time >= 5:  # 5-second delay
            
            predicted_character = labels_dict[int(prediction[0])]
            if predicted_character == 'next':
                predicted_text += before
            elif predicted_character == 'backspace':
                predicted_text = predicted_text[:-1]  # Remove the last character
            else:
                before = predicted_character
            last_prediction_time = current_time  # Reset the timer

        # Draw prediction results on the frame
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()

