import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


data_dir='./Data'
#detect all landmarks and draw on top of the images
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data=[]
labels=[]


for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir,dir_)):
        data_aux=[]
        x_=[]
        y_=[]
        img=cv2.imread(os.path.join(data_dir,dir_,img_path))
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)      #have to convert to rgb for mediapipe to read
        
        result=hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            data.append(data_aux)
            labels.append(dir_)

           
           
            #    mp_drawing.draw_landmarks(
            #         img_rgb,  # Image to draw
            #         hand_landmarks,  # Model output
            #         mp_hands.HAND_CONNECTIONS,  # Hand connections
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style()  # Different style for connections
            #     )
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(labels)
print(data)